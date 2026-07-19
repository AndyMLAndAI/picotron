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
from picotron.parallel.zero import DistributedGradScaler, ZeroOptimizer
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
        # Training uses one fixed sequence length, so cuDNN's shape autotuning
        # can safely select its fastest kernels without re-tuning every step.
        torch.backends.cudnn.benchmark = True

    training_dtype = config.model.resolve_dtype(target_device)
    if target_device.type == "cpu" and training_dtype != torch.float32:
        raise ValueError(
            "CPU training supports model.dtype='auto' or 'float32'; "
            "float16 and bfloat16 require CUDA in Picotron."
        )
    # Keep parameters/master optimizer state in fp32. CUDA autocast below
    # applies the configured compute dtype without unsafe fp16 AdamW state.
    model.to(device=target_device)
    # Compile before DDP: DDP must register hooks on the module that actually
    # runs forward/backward, while checkpointing unwraps the compiled module.
    model = _maybe_compile_model(model, config)

    distributed_info = initialize_distributed(
        expected_world_size=config.parallelism.dp
    )
    model = wrap_model(model, distributed_info, device=target_device)
    model.train()

    zero_stage = config.parallelism.zero_stage
    if zero_stage > 0 and optimizer is not None:
        raise ValueError("Pass no optimizer when ZeRO is enabled; it creates sharded AdamW.")
    save_path = checkpoint_path or config.checkpoints.checkpoints_path
    resume_path = resume_from or config.checkpoints.resume_checkpoint_path

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
    grad_scaler = _create_grad_scaler(target_device, training_dtype, zero_stage)

    with TrainingDisplay(
        config,
        total_steps=start_step + target_steps,
        plain_interval=config.logging.iteration_step_info_interval,
        enabled=distributed_info.rank == 0,
    ) as display, FileLogger(
        config,
        method="pretraining",
        # One primary transcript avoids two DDP ranks concurrently appending
        # to the same CSV and log paths.
        enabled=distributed_info.rank == 0,
    ) as file_logger:
        data_epoch = 0
        while len(losses) < target_steps:
            _set_data_epoch(data_loader, data_epoch)
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
                if isinstance(grad_scaler, DistributedGradScaler):
                    active_optimizer.backward(grad_scaler.scale(loss), model)
                    grad_scaler.step(
                        active_optimizer,
                        model,
                        max_grad_norm=config.optimizer.clip_grad,
                    )
                elif grad_scaler is not None:
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
                    _save_checkpoint(
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
            data_epoch += 1

    final_step = start_step + len(losses)
    if (
        save_path is not None
        and config.checkpoints.save_final_state
        and final_step != last_saved_step
    ):
        _save_checkpoint(
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


def _maybe_compile_model(model: nn.Module, config: PicotronConfig) -> nn.Module:
    """Compile the model only when explicitly requested, with eager fallback."""

    if not config.model.compile_model:
        return model
    try:
        return _CompileFallbackModule(model, torch.compile(model))
    except Exception as error:  # pragma: no cover - backend-specific failures.
        logging.getLogger("picotron").warning(
            "torch.compile failed; using eager model instead: %s", error
        )
        return model


class _CompileFallbackModule(nn.Module):
    """Run a compiled model until its lazy compilation fails, then use eager.

    ``torch.compile`` commonly defers backend work to its first forward. This
    wrapper catches those deferred failures, logs them once, and permanently
    selects the original module. The original module is deliberately exposed
    as ``_orig_mod`` so checkpoint serialization can unwrap it just like
    PyTorch's own compiled module.
    """

    def __init__(self, eager_model: nn.Module, compiled_model: nn.Module) -> None:
        super().__init__()
        self._orig_mod = eager_model
        # Keep the compiled wrapper out of Module registration: it already
        # references _orig_mod, and double registration would duplicate keys.
        object.__setattr__(self, "_compiled_model", compiled_model)
        self._compiled_active = True

    def forward(self, *args: object, **kwargs: object) -> object:
        if self._compiled_active:
            try:
                return self._compiled_model(*args, **kwargs)
            except Exception as error:  # pragma: no cover - backend-specific failures.
                logging.getLogger("picotron").warning(
                    "torch.compile execution failed; using eager model instead: %s", error
                )
                self._compiled_active = False
        return self._orig_mod(*args, **kwargs)


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
    zero_stage: int,
) -> torch.amp.GradScaler | DistributedGradScaler | None:
    """Create a scaler for CUDA fp16; bf16 and fp32 do not need scaling."""

    if device.type != "cuda" or dtype != torch.float16:
        return None
    if zero_stage > 0:
        return DistributedGradScaler("cuda")
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


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | ZeroOptimizer,
    step: int,
    path: str | Path,
    distributed_info: DistributedInfo,
) -> None:
    """Persist portable weights and per-rank ZeRO shards when needed."""

    if isinstance(optimizer, ZeroOptimizer) and distributed_info.is_distributed:
        save_checkpoint(model, optimizer, step, path)
    elif distributed_info.rank == 0:
        save_checkpoint(model, optimizer, step, path)
    if distributed_info.is_distributed:
        dist.barrier()


def _batch_to_input_ids(batch: Tensor, device: torch.device) -> Tensor:
    """Move a tensor batch to the target device and validate token dtype."""

    if not isinstance(batch, Tensor):
        raise TypeError("The data loader must yield token-id tensors.")
    if batch.is_floating_point() or batch.is_complex():
        raise TypeError("Token ids must use an integer tensor dtype.")
    return batch.to(device=device, dtype=torch.long, non_blocking=True)


def _set_data_epoch(data_loader: Iterable[Tensor], epoch: int) -> None:
    """Advance DistributedSampler shuffles when a finite loader is restarted."""

    sampler = getattr(data_loader, "sampler", None)
    if isinstance(sampler, torch.utils.data.DistributedSampler):
        sampler.set_epoch(epoch)


def _model_auxiliary_loss(model: nn.Module) -> Tensor | None:
    """Retrieve optional MoE routing loss from a plain or DDP-wrapped model."""

    auxiliary_loss = getattr(model, "auxiliary_loss", None)
    if auxiliary_loss is None and hasattr(model, "module"):
        auxiliary_loss = getattr(model.module, "auxiliary_loss", None)
    return auxiliary_loss if isinstance(auxiliary_loss, Tensor) else None
