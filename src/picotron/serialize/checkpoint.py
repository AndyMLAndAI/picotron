"""Safetensors model checkpoints with a PyTorch optimizer sidecar."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from picotron.parallel.zero import ZeroOptimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | ZeroOptimizer,
    step: int,
    path: str | Path,
) -> None:
    """Save model weights as safetensors and optimizer metadata in sidecars.

    ``path`` remains a compatibility stem: ``run.pt`` resolves to
    ``run.safetensors`` and ``run.optimizer.pt``. Passing a ``.safetensors``
    path uses it directly and writes the matching optimizer sidecar.
    """

    if step < 0:
        raise ValueError("step must be non-negative.")
    weights_path, metadata_path = _checkpoint_paths(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_distributed_zero(optimizer):
        _save_distributed_zero_checkpoint(
            model, optimizer, step, weights_path, metadata_path
        )
        return
    checkpoint_model = _unwrap_ddp_model(model)
    state_dict = {
        name: value.detach().cpu().contiguous()
        for name, value in checkpoint_model.state_dict().items()
    }
    save_file(state_dict, str(weights_path))
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "step": step}, metadata_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | ZeroOptimizer,
    path: str | Path,
    *,
    load_optimizer: bool = True,
) -> int:
    """Load safetensors weights and optionally restore optimizer state in place."""

    weights_path, metadata_path = _checkpoint_paths(path)
    checkpoint_model = _unwrap_ddp_model(model)
    try:
        device = next(checkpoint_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint optimizer metadata not found: {metadata_path}")

    checkpoint_model.load_state_dict(load_file(str(weights_path), device=str(device)))
    payload: Any = torch.load(metadata_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint metadata must contain a mapping payload.")
    required_keys = {"step"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(f"Checkpoint metadata is missing keys: {sorted(missing_keys)}.")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("Checkpoint step must be a non-negative integer.")

    if _is_distributed_zero(optimizer):
        _load_distributed_zero_optimizer(optimizer, payload, weights_path, load_optimizer)
    else:
        if "optimizer_state_dict" not in payload:
            raise ValueError("Checkpoint metadata is missing optimizer_state_dict.")
        if load_optimizer:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload["step"]


def _unwrap_ddp_model(model: nn.Module) -> nn.Module:
    """Use the underlying model so checkpoints are portable across DDP runs."""

    return model.module if isinstance(model, DistributedDataParallel) else model


def _checkpoint_paths(path: str | Path) -> tuple[Path, Path]:
    """Resolve a compatibility path to weights and optimizer sidecar files."""

    requested_path = Path(path)
    weights_path = (
        requested_path
        if requested_path.suffix.lower() == ".safetensors"
        else requested_path.with_suffix(".safetensors")
    )
    metadata_path = weights_path.with_suffix(".optimizer.pt")
    return weights_path, metadata_path


def _is_distributed_zero(optimizer: Optimizer | ZeroOptimizer) -> bool:
    return (
        isinstance(optimizer, ZeroOptimizer)
        and optimizer.world_size > 1
        and dist.is_available()
        and dist.is_initialized()
    )


def _save_distributed_zero_checkpoint(
    model: nn.Module,
    optimizer: ZeroOptimizer,
    step: int,
    weights_path: Path,
    metadata_path: Path,
) -> None:
    """Save replicated weights once and one optimizer-state shard per rank."""

    if optimizer.rank == 0:
        checkpoint_model = _unwrap_ddp_model(model)
        state_dict = {
            name: value.detach().cpu().contiguous()
            for name, value in checkpoint_model.state_dict().items()
        }
        save_file(state_dict, str(weights_path))
        torch.save(
            {"step": step, "zero_world_size": optimizer.world_size}, metadata_path
        )
    torch.save(optimizer.state_dict(), _zero_optimizer_path(weights_path, optimizer.rank))


def _load_distributed_zero_optimizer(
    optimizer: ZeroOptimizer,
    payload: Mapping[str, Any],
    weights_path: Path,
    load_optimizer: bool,
) -> None:
    saved_world_size = payload.get("zero_world_size")
    if saved_world_size != optimizer.world_size:
        raise ValueError(
            "ZeRO checkpoint world size does not match the active process group."
        )
    if not load_optimizer:
        return
    shard_path = _zero_optimizer_path(weights_path, optimizer.rank)
    if not shard_path.exists():
        raise FileNotFoundError(f"ZeRO optimizer shard not found: {shard_path}")
    shard: Any = torch.load(shard_path, map_location="cpu", weights_only=False)
    if not isinstance(shard, dict):
        raise ValueError("ZeRO optimizer shard must contain a mapping payload.")
    optimizer.load_state_dict(shard)


def _zero_optimizer_path(weights_path: Path, rank: int) -> Path:
    return weights_path.with_suffix(f".optimizer.rank{rank}.pt")
