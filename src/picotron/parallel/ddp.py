"""DistributedDataParallel setup for torchrun and CPU tests."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class DistributedInfo:
    """Process-group identity and selected backend."""

    rank: int
    world_size: int
    local_rank: int
    backend: str | None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def initialize_distributed(
    backend: str | None = None,
    *,
    expected_world_size: int | None = None,
) -> DistributedInfo:
    """Initialize torchrun DDP and validate its configured data-parallel size."""

    try:
        configured_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        configured_rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise RuntimeError("torchrun rank environment variables must be integers.") from error

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = configured_world_size
        rank = configured_rank
    if expected_world_size is not None and expected_world_size != world_size:
        raise RuntimeError(
            "Configured parallelism.dp does not match the active process-group size: "
            f"expected {expected_world_size}, got {world_size}."
        )
    if world_size <= 1:
        return DistributedInfo(rank=0, world_size=1, local_rank=0, backend=None)

    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if not dist.is_initialized():
        dist.init_process_group(
            backend=selected_backend,
            rank=rank,
            world_size=world_size,
            init_method="env://",
        )
    return DistributedInfo(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        backend=dist.get_backend(),
    )


def wrap_model(
    model: nn.Module,
    info: DistributedInfo,
    *,
    device: torch.device | str,
) -> nn.Module:
    """Move and wrap a model when running with more than one process."""

    target_device = torch.device(device)
    model.to(target_device)
    if not info.is_distributed:
        return model
    if target_device.type == "cuda":
        device_index = (
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        )
        return DistributedDataParallel(model, device_ids=[device_index])
    return DistributedDataParallel(model)


def cleanup_distributed() -> None:
    """Tear down an initialized process group."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
