"""Config-driven DataLoader construction for synthetic token data."""

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
) -> DataLoader[torch.Tensor]:
    """Build a batched loader using ``config.tokens.micro_batch_size``."""

    dataset = SyntheticTokenDataset(config, num_sequences, seed=seed)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.tokens.micro_batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def create_memmap_dataloader(
    config: PicotronConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
) -> DataLoader[torch.Tensor]:
    """Build a token-cache loader, sharding examples across DDP ranks."""

    token_path = config.data.dataset_token_path
    if token_path is None:
        raise ValueError("config.data.dataset_token_path is required for memmap loading.")
    dataset = MemmapTokenDataset(config, token_path)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.general.seed,
        )
        if world_size > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=config.tokens.micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
