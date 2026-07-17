"""Synthetic token dataset used until the streaming data pipeline exists."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


class MemmapTokenDataset(Dataset[Tensor]):
    """Read fixed-length token examples lazily from a uint16 memmap file."""

    def __init__(self, config: PicotronConfig, path: str | Path) -> None:
        token_path = Path(path)
        if not token_path.is_file():
            raise FileNotFoundError(f"Token cache does not exist: {token_path}")
        if token_path.stat().st_size % np.dtype(np.uint16).itemsize:
            raise ValueError("Token cache size must be divisible by uint16 item size.")
        self.sequence_length = config.tokens.sequence_length
        self._tokens = np.memmap(token_path, mode="r", dtype=np.uint16)
        self._num_sequences = len(self._tokens) // self.sequence_length
        if self._num_sequences <= 0:
            raise ValueError("Token cache does not contain a complete sequence.")

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, index: int) -> Tensor:
        if index < 0 or index >= self._num_sequences:
            raise IndexError(f"Memmap token index out of range: {index}.")
        start = index * self.sequence_length
        values = np.asarray(
            self._tokens[start : start + self.sequence_length], dtype=np.int64
        )
        return torch.from_numpy(values.copy())
