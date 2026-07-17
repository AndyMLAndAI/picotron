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


def initialize_distributed(backend: str | None = None) -> DistributedInfo:
    """Initialize a torchrun process group, or return single-process metadata."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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
        return DistributedDataParallel(model, device_ids=[target_device.index])
    return DistributedDataParallel(model)


def cleanup_distributed() -> None:
    """Tear down an initialized process group."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

