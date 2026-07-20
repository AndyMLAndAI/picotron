"""Synthetic token dataset used until the streaming data pipeline exists."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

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
        token_path = _decompress_gzip_cache(token_path)
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

    def get_batch(self, indices: Tensor) -> Tensor:
        """Fetch a same-source batch without Python-level sequence collation."""

        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be a one-dimensional integer tensor.")
        if torch.any(indices < 0) or torch.any(indices >= self._num_sequences):
            raise IndexError("Memmap token index out of range.")
        positions = indices.cpu().numpy().astype(np.int64, copy=False)
        offsets = positions[:, None] * self.sequence_length + np.arange(self.sequence_length)
        values = np.asarray(self._tokens[offsets], dtype=np.int64)
        return torch.from_numpy(values.copy())


class WeightedInterleavedBatchDataset(IterableDataset[Tensor]):
    """Yield full batches sampled from token caches by configurable weights.

    A batch is drawn from one source, so a 0.7/0.3 configuration produces an
    actual 70/30 *batch* mix rather than a concatenated corpus. Each DDP rank
    samples only its modulo partition of every source cache.
    """

    def __init__(
        self,
        datasets: tuple[MemmapTokenDataset, ...],
        weights: tuple[float, ...],
        *,
        batch_size: int,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        if len(datasets) < 2 or len(datasets) != len(weights):
            raise ValueError("Weighted interleaving requires at least two datasets and matching weights.")
        if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid batch_size, rank, or world_size for weighted interleaving.")
        if any(weight <= 0 for weight in weights):
            raise ValueError("All dataset weights must be positive.")
        if any(len(dataset) <= rank for dataset in datasets):
            raise ValueError("Every dataset must contain at least one sequence per DDP rank.")
        self._datasets = datasets
        self._weights = torch.tensor(weights, dtype=torch.double)
        self._batch_size = batch_size
        self._seed = seed
        self._rank = rank
        self._world_size = world_size

    def __iter__(self):  # type: ignore[override]
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        generator = torch.Generator()
        generator.manual_seed(self._seed + 10_007 * self._rank + 1_000_003 * worker_id)

        while True:
            source_index = int(torch.multinomial(self._weights, 1, generator=generator).item())
            dataset = self._datasets[source_index]
            local_length = (len(dataset) - self._rank + self._world_size - 1) // self._world_size
            local_indices = torch.randint(
                local_length, (self._batch_size,), generator=generator, dtype=torch.long
            )
            global_indices = self._rank + local_indices * self._world_size
            yield dataset.get_batch(global_indices)


def _decompress_gzip_cache(token_path: Path) -> Path:
    """Materialize an opt-in gzip token cache once before memmapping it."""

    if token_path.suffix.lower() != ".gz":
        return token_path
    cache_key = f"{token_path.resolve()}:{token_path.stat().st_mtime_ns}:{token_path.stat().st_size}"
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    cache_dir = Path(tempfile.gettempdir()) / "picotron-token-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / f"{digest}.uint16"
    if raw_path.exists():
        return raw_path
    temporary_path = cache_dir / f"{raw_path.name}.{uuid.uuid4().hex}.partial"
    with gzip.open(token_path, "rb") as compressed_file, temporary_path.open("wb") as raw_file:
        shutil.copyfileobj(compressed_file, raw_file, length=1024 * 1024)
    if raw_path.exists():
        temporary_path.unlink(missing_ok=True)
    else:
        temporary_path.replace(raw_path)
    return raw_path
