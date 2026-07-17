"""Config-driven DataLoader construction for synthetic token data."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from picotron.config.config import PicotronConfig
from picotron.data.dataset import SyntheticTokenDataset


def create_synthetic_dataloader(
    config: PicotronConfig,
    num_sequences: int,
    *,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader[torch.Tensor]:
    """Build a batched loader using ``config.batch_size``."""

    dataset = SyntheticTokenDataset(config, num_sequences, seed=seed)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
    )

