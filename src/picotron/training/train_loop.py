"""Minimal single-device next-token pretraining loop."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import AdamW

from picotron.config.config import PicotronConfig
from picotron.logging.display import TrainingDisplay
from picotron.nn.triton_kernels.adamw import AdamWStepWithFallback
from picotron.nn.triton_kernels.cross_entropy import CrossEntropyWithFallback
from picotron.parallel.ddp import initialize_distributed, wrap_model
from picotron.parallel.zero import ZeroOptimizer
from picotron.serialize.checkpoint import load_checkpoint, save_checkpoint


def train(
    model: nn.Module,
    data_loader: Iterable[Tensor],
    config: PicotronConfig,
    *,
    device: torch.device | str = torch.device("cpu"),
    max_steps: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    checkpoint_path: str | None = None,
    resume_from: str | None = None,
) -> list[float]:
    """Train tensor-only pretraining batches for ``config.num_epochs`` or ``max_steps``.

    The tensor-only next-token batch format is this pretraining API's required
    contract. Label-aware or model-specific inputs are handled by picotron_sft.
    """

    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    target_device = torch.device(device)
    distributed_info = initialize_distributed()
    model = wrap_model(model, distributed_info, device=target_device)
    model.train()
    if config.zero_stage > 0 and optimizer is not None:
        raise ValueError("Pass no optimizer when ZeRO is enabled; it creates its sharded AdamW.")
    if config.zero_stage > 0 and distributed_info.is_distributed and (
        checkpoint_path is not None or resume_from is not None
    ):
        raise NotImplementedError(
            "Distributed ZeRO checkpointing requires rank-sharded checkpoint files."
        )
    active_optimizer = (
        ZeroOptimizer(
            model.parameters(),
            learning_rate=config.learning_rate,
            stage=config.zero_stage,
        )
        if config.zero_stage > 0
        else optimizer or AdamW(model.parameters(), lr=config.learning_rate)
    )
    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(model, active_optimizer, resume_from)
    losses: list[float] = []
    loss_function = CrossEntropyWithFallback(
        use_triton=bool(config.model_kwargs.get("use_triton_cross_entropy", False))
    )
    optimizer_step = AdamWStepWithFallback(
        use_triton=bool(config.model_kwargs.get("use_triton_adamw", False))
    )

    with TrainingDisplay(config, total_steps=max_steps) as display:
        for _ in range(config.num_epochs):
            for batch in data_loader:
                if max_steps is not None and len(losses) >= max_steps:
                    return losses
                input_ids = _batch_to_input_ids(batch, target_device)
                if input_ids.ndim != 2 or input_ids.size(1) < 2:
                    raise ValueError("Training batches must have shape (batch, sequence >= 2).")

                active_optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids)
                loss = loss_function(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                )
                auxiliary_loss = _model_auxiliary_loss(model)
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss
                if isinstance(active_optimizer, ZeroOptimizer):
                    active_optimizer.backward(loss, model)
                else:
                    loss.backward()
                if isinstance(active_optimizer, ZeroOptimizer):
                    active_optimizer.step()
                else:
                    optimizer_step.step(active_optimizer)

                loss_value = loss.detach().cpu().item()
                losses.append(loss_value)
                step = start_step + len(losses)
                display.update(
                    step=step,
                    loss=loss_value,
                    learning_rate=active_optimizer.param_groups[0]["lr"],
                    tokens_seen=step * input_ids.numel(),
                )
                if (
                    checkpoint_path is not None
                    and step % config.checkpoint_interval == 0
                ):
                    save_checkpoint(model, active_optimizer, step, checkpoint_path)

    return losses


def _batch_to_input_ids(batch: Tensor, device: torch.device) -> Tensor:
    """Move a tensor batch to the target device and validate token dtype."""

    if not isinstance(batch, Tensor):
        raise TypeError("The data loader must yield token-id tensors.")
    if batch.is_floating_point() or batch.is_complex():
        raise TypeError("Token ids must use an integer tensor dtype.")
    return batch.to(device=device, dtype=torch.long)


def _model_auxiliary_loss(model: nn.Module) -> Tensor | None:
    """Retrieve optional MoE routing loss from a plain or DDP-wrapped model."""

    auxiliary_loss = getattr(model, "auxiliary_loss", None)
    if auxiliary_loss is None and hasattr(model, "module"):
        auxiliary_loss = getattr(model.module, "auxiliary_loss", None)
    return auxiliary_loss if isinstance(auxiliary_loss, Tensor) else None
