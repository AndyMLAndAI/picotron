"""Config-driven, rank-aware DataLoader construction for token datasets."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler

from picotron.config.config import PicotronConfig
from picotron.data.dataset import MemmapTokenDataset, SyntheticTokenDataset


def create_synthetic_dataloader(
    config: PicotronConfig,
    num_sequences: int,
    *,
    shuffle: bool = True,
    seed: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader[torch.Tensor]:
    """Build a synthetic-data loader, sharding examples when DDP is active."""

    dataset = SyntheticTokenDataset(config, num_sequences, seed=seed)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    sampler = _distributed_sampler(
        dataset,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        seed=config.general.seed,
    )
    return _create_loader(
        config,
        dataset,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        generator=generator,
    )


def create_memmap_dataloader(
    config: PicotronConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int | None = None,
) -> DataLoader[torch.Tensor]:
    """Build a token-cache loader, sharding examples across DDP ranks."""

    token_path = config.data.dataset_token_path
    if token_path is None:
        raise ValueError("config.data.dataset_token_path is required for memmap loading.")
    dataset = MemmapTokenDataset(config, token_path)
    sampler = _distributed_sampler(
        dataset,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=config.general.seed,
    )
    return _create_loader(
        config,
        dataset,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
    )


def _distributed_sampler(
    dataset: torch.utils.data.Dataset[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    shuffle: bool,
    seed: int,
) -> DistributedSampler | None:
    """Return one deterministic, non-overlapping sampler partition per DDP rank."""

    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in the range [0, world_size).")
    if world_size == 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        # Padding would repeat examples across ranks when the dataset length is
        # not divisible by world size. Dropping the short tail preserves the
        # strict no-overlap DDP invariant.
        drop_last=True,
    )


def _create_loader(
    config: PicotronConfig,
    dataset: torch.utils.data.Dataset[torch.Tensor],
    *,
    sampler: DistributedSampler | None,
    shuffle: bool,
    generator: torch.Generator | None = None,
    num_workers: int | None = None,
) -> DataLoader[torch.Tensor]:
    """Apply the common throughput settings to one rank's DataLoader."""

    active_workers = config.data.num_workers if num_workers is None else num_workers
    if active_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    # Worker processes overlap CPU memmap reads/token collation with model work.
    # Pinned batches make CUDA's non-blocking copy effective; persistence avoids
    # respawning workers at every epoch. PyTorch requires it disabled for zero workers.
    loader_kwargs: dict[str, object] = {
        "batch_size": config.tokens.micro_batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "generator": generator,
        "num_workers": active_workers,
        "pin_memory": True,
        "persistent_workers": active_workers > 0,
    }
    if active_workers > 0:
        loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
    return DataLoader(
        dataset,
        **loader_kwargs,
    )
