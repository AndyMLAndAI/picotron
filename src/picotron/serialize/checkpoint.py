"""Atomic model and optimizer checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    path: str | Path,
) -> None:
    """Save model weights, optimizer state, and global step to one file."""

    if step < 0:
        raise ValueError("step must be non-negative.")
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        checkpoint_path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    path: str | Path,
) -> int:
    """Load checkpoint state into the supplied model and optimizer in place."""

    checkpoint_path = Path(path)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    payload: Any = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a mapping payload.")
    required_keys = {"model_state_dict", "optimizer_state_dict", "step"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing_keys)}.")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("Checkpoint step must be a non-negative integer.")

    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload["step"]

