"""Config-driven next-token pretraining loop."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import AdamW

from picotron.config.config import PicotronConfig
from picotron.logging.display import TrainingDisplay
from picotron.logging.file_logger import FileLogger
from picotron.nn.triton_kernels.adamw import AdamWStepWithFallback
from picotron.nn.triton_kernels.cross_entropy import CrossEntropyWithFallback
from picotron.parallel.ddp import DistributedInfo, initialize_distributed, wrap_model
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
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> list[float]:
    """Train an exact number of next-token steps from the nested run config.

    ``max_steps`` is an explicit override for ``config.tokens.train_steps``.
    Finite, re-iterable loaders are restarted as needed; an empty or exhausted
    one-shot loader fails loudly instead of silently training fewer steps.
    """

    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    _configure_runtime(config)
    target_device = torch.device(device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable.")
        if target_device.index is not None:
            torch.cuda.set_device(target_device)

    training_dtype = config.model.resolve_dtype(target_device)
    if target_device.type == "cpu" and training_dtype != torch.float32:
        raise ValueError(
            "CPU training supports model.dtype='auto' or 'float32'; "
            "float16 and bfloat16 require CUDA in Picotron."
        )
    # Keep parameters/master optimizer state in fp32. CUDA autocast below
    # applies the configured compute dtype without unsafe fp16 AdamW state.
    model.to(device=target_device)

    distributed_info = initialize_distributed(
        expected_world_size=config.parallelism.dp
    )
    model = wrap_model(model, distributed_info, device=target_device)
    model.train()

    zero_stage = config.parallelism.zero_stage
    if zero_stage > 0 and optimizer is not None:
        raise ValueError("Pass no optimizer when ZeRO is enabled; it creates sharded AdamW.")
    if target_device.type == "cuda" and training_dtype == torch.float16 and zero_stage > 0:
        raise NotImplementedError(
            "fp16 with ZeRO requires a distributed GradScaler path that is not implemented; "
            "use zero_stage=0 or float32 until it is added."
        )

    save_path = checkpoint_path or config.checkpoints.checkpoints_path
    resume_path = resume_from or config.checkpoints.resume_checkpoint_path
    if zero_stage > 0 and distributed_info.is_distributed and (
        save_path is not None or resume_path is not None
    ):
        raise NotImplementedError(
            "Distributed ZeRO checkpointing requires rank-sharded checkpoint files."
        )

    active_optimizer = _build_optimizer(model, config, optimizer)
    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(
            model,
            active_optimizer,
            resume_path,
            load_optimizer=config.checkpoints.load_optimizer,
        )

    target_steps = max_steps or config.tokens.train_steps
    losses: list[float] = []
    last_saved_step: int | None = None
    loss_function = CrossEntropyWithFallback(
        use_triton=config.model.triton_kernels.cross_entropy
    )
    optimizer_step = AdamWStepWithFallback(
        use_triton=config.model.triton_kernels.adamw
    )
    grad_scaler = _create_grad_scaler(target_device, training_dtype)

    with TrainingDisplay(
        config,
        total_steps=start_step + target_steps,
        plain_interval=config.logging.iteration_step_info_interval,
    ) as display, FileLogger(config, method="pretraining") as file_logger:
        while len(losses) < target_steps:
            batches_this_pass = 0
            for batch in data_loader:
                batches_this_pass += 1
                absolute_step = start_step + len(losses)
                learning_rate = config.optimizer.learning_rate_scheduler.learning_rate_at(
                    absolute_step
                )
                _set_learning_rate(active_optimizer, learning_rate)

                input_ids = _batch_to_input_ids(batch, target_device)
                if input_ids.ndim != 2 or input_ids.size(1) < 2:
                    raise ValueError(
                        "Training batches must have shape (batch, sequence >= 2)."
                    )

                active_optimizer.zero_grad(set_to_none=True)
                with _autocast_context(target_device, training_dtype):
                    logits = model(input_ids)
                    # Cross entropy accepts (batch, classes, sequence) directly.
                    # This avoids the multi-gigabyte contiguous logits copy created by
                    # flattening a sliced (batch, sequence, vocabulary) tensor.
                    loss = loss_function(
                        logits[:, :-1, :].transpose(1, 2),
                        input_ids[:, 1:],
                    )
                    auxiliary_loss = _model_auxiliary_loss(model)
                    if auxiliary_loss is not None:
                        loss = loss + auxiliary_loss
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(active_optimizer)
                    _clip_gradients(model, active_optimizer, config.optimizer.clip_grad)
                    grad_scaler.step(active_optimizer)
                    grad_scaler.update()
                elif isinstance(active_optimizer, ZeroOptimizer):
                    active_optimizer.backward(loss, model)
                    _clip_gradients(model, active_optimizer, config.optimizer.clip_grad)
                    active_optimizer.step()
                else:
                    loss.backward()
                    _clip_gradients(model, active_optimizer, config.optimizer.clip_grad)
                    optimizer_step.step(active_optimizer)

                loss_value = loss.detach().float().cpu().item()
                losses.append(loss_value)
                completed_step = start_step + len(losses)
                display.update(
                    step=completed_step,
                    loss=loss_value,
                    learning_rate=learning_rate,
                    tokens_seen=completed_step * input_ids.numel(),
                )
                file_logger.log_step(
                    step=completed_step,
                    loss=loss_value,
                    learning_rate=learning_rate,
                    tokens_seen=completed_step * input_ids.numel(),
                )
                if (
                    save_path is not None
                    and completed_step % config.checkpoints.checkpoint_interval == 0
                ):
                    _save_checkpoint_on_primary_rank(
                        model,
                        active_optimizer,
                        completed_step,
                        save_path,
                        distributed_info,
                    )
                    last_saved_step = completed_step

                if len(losses) >= target_steps:
                    break
            if batches_this_pass == 0:
                raise ValueError(
                    "The data loader yielded no batches before the configured train-step "
                    "budget was reached."
                )

    final_step = start_step + len(losses)
    if (
        save_path is not None
        and config.checkpoints.save_final_state
        and final_step != last_saved_step
    ):
        _save_checkpoint_on_primary_rank(
            model,
            active_optimizer,
            final_step,
            save_path,
            distributed_info,
        )
    return losses


def _configure_runtime(config: PicotronConfig) -> None:
    """Apply the implemented seed and standard-log-level configuration."""

    torch.manual_seed(config.general.seed)
    logging.getLogger("picotron").setLevel(config.logging.log_level)


def _autocast_context(
    device: torch.device,
    dtype: torch.dtype,
) -> AbstractContextManager[object]:
    """Return CUDA autocast only for a configured reduced-precision run."""

    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _create_grad_scaler(
    device: torch.device,
    dtype: torch.dtype,
) -> torch.amp.GradScaler | None:
    """Create a scaler for CUDA fp16; bf16 and fp32 do not need scaling."""

    if device.type != "cuda" or dtype != torch.float16:
        return None
    return torch.amp.GradScaler("cuda")


def _build_optimizer(
    model: nn.Module,
    config: PicotronConfig,
    supplied_optimizer: torch.optim.Optimizer | None,
) -> torch.optim.Optimizer | ZeroOptimizer:
    """Construct the configured AdamW implementation unless one was supplied."""

    if supplied_optimizer is not None:
        return supplied_optimizer
    factory = config.optimizer.optimizer_factory
    scheduler = config.optimizer.learning_rate_scheduler
    adamw_kwargs = {
        "lr": scheduler.learning_rate,
        "betas": (factory.adam_beta1, factory.adam_beta2),
        "eps": factory.adam_eps,
        "weight_decay": config.optimizer.weight_decay,
    }
    if config.parallelism.zero_stage > 0:
        return ZeroOptimizer(
            model.parameters(),
            learning_rate=scheduler.learning_rate,
            stage=config.parallelism.zero_stage,
            betas=adamw_kwargs["betas"],
            eps=adamw_kwargs["eps"],
            weight_decay=adamw_kwargs["weight_decay"],
        )
    return AdamW(model.parameters(), **adamw_kwargs)


def _set_learning_rate(
    optimizer: torch.optim.Optimizer | ZeroOptimizer,
    learning_rate: float,
) -> None:
    """Set a schedule-selected rate on every active optimizer parameter group."""

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _clip_gradients(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | ZeroOptimizer,
    max_norm: float | None,
) -> None:
    """Apply configured gradient clipping without breaking ZeRO stage semantics."""

    if max_norm is None:
        return
    if isinstance(optimizer, ZeroOptimizer):
        optimizer.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _save_checkpoint_on_primary_rank(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | ZeroOptimizer,
    step: int,
    path: str | Path,
    distributed_info: DistributedInfo,
) -> None:
    """Persist one checkpoint safely when replicated DDP ranks share a path."""

    if distributed_info.rank == 0:
        save_checkpoint(model, optimizer, step, path)
    if distributed_info.is_distributed:
        dist.barrier()


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
