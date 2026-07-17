"""Synthetic token dataset used until the streaming data pipeline exists."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from picotron.config.config import PicotronConfig


class SyntheticTokenDataset(Dataset[Tensor]):
    """Generate fixed-length random token sequences for pretraining smoke tests."""

    def __init__(
        self,
        config: PicotronConfig,
        num_sequences: int,
        *,
        seed: int | None = None,
    ) -> None:
        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive.")
        self.vocab_size = config.model.model_config.vocab_size
        self.sequence_length = config.tokens.sequence_length
        self.num_sequences = num_sequences
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Tensor:
        if index < 0 or index >= self.num_sequences:
            raise IndexError(f"Synthetic token index out of range: {index}.")
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.sequence_length,),
            dtype=torch.long,
            generator=self._generator,
        )
