"""CPU checks for DataLoader throughput settings and DDP data partitions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from picotron.config.config import DataConfig
from picotron.data.dataloader import create_memmap_dataloader, create_synthetic_dataloader
from tests.config_factory import make_test_config


def _shard_worker(rank: int, token_path: str, output_dir: str) -> None:
    """Collect first-token identifiers from one independently constructed rank loader."""

    base_config = make_test_config(sequence_length=4, micro_batch_size=2)
    config = replace(
        base_config,
        data=DataConfig(
            dataset_token_path=token_path,
            num_workers=0,
            prefetch_factor=2,
        ),
    )
    loader = create_memmap_dataloader(config, rank=rank, world_size=2)
    identifiers = torch.cat([batch[:, 0] for batch in loader]).tolist()
    torch.save(identifiers, Path(output_dir) / f"rank_{rank}.pt")


def test_dataloader_throughput_settings_are_config_driven() -> None:
    config = make_test_config(
        data=DataConfig(num_workers=3, prefetch_factor=4),
    )

    loader = create_synthetic_dataloader(config, num_sequences=12, seed=7)

    assert loader.num_workers == 3
    assert loader.persistent_workers
    assert loader.pin_memory
    assert loader.prefetch_factor == 4


def test_two_rank_memmap_loaders_receive_disjoint_dataset_examples(tmp_path: Path) -> None:
    """A two-process proxy for DDP must prove each rank sees a unique shard."""

    token_path = tmp_path / "tokens.uint16"
    np.arange(16 * 4, dtype=np.uint16).tofile(token_path)
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_shard_worker, args=(rank, str(token_path), str(tmp_path)))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0

    rank_zero = set(torch.load(tmp_path / "rank_0.pt", map_location="cpu"))
    rank_one = set(torch.load(tmp_path / "rank_1.pt", map_location="cpu"))
    expected = set(range(0, 16 * 4, 4))
    assert rank_zero.isdisjoint(rank_one)
    assert rank_zero | rank_one == expected
