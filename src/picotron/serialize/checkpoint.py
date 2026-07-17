"""Safetensors model checkpoints with a PyTorch optimizer sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    path: str | Path,
) -> None:
    """Save model weights as safetensors and optimizer metadata in a sidecar.

    ``path`` remains a compatibility stem: ``run.pt`` resolves to
    ``run.safetensors`` and ``run.optimizer.pt``. Passing a ``.safetensors``
    path uses it directly and writes the matching optimizer sidecar.
    """

    if step < 0:
        raise ValueError("step must be non-negative.")
    weights_path, metadata_path = _checkpoint_paths(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_model = _unwrap_ddp_model(model)
    state_dict = {
        name: value.detach().cpu().contiguous()
        for name, value in checkpoint_model.state_dict().items()
    }
    save_file(state_dict, str(weights_path))
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        metadata_path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
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
    required_keys = {"optimizer_state_dict", "step"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(f"Checkpoint metadata is missing keys: {sorted(missing_keys)}.")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("Checkpoint step must be a non-negative integer.")

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
