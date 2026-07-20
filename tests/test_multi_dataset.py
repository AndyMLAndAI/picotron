"""CPU checks for weighted token-cache interleaving."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from picotron.config.config import DataConfig, DatasetSourceConfig, load_config
from picotron.data.dataloader import create_memmap_dataloader
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.training.train_loop import train
from tests.config_factory import make_test_config


def _write_repeated_sequences(path: Path, sequence: list[int], count: int = 96) -> None:
    """Write a token cache containing recognizable fixed-length examples."""

    np.tile(np.asarray(sequence, dtype=np.uint16), count).tofile(path)


def _write_indexed_sequences(path: Path, *, start: int, count: int = 32) -> None:
    """Write uniquely identifiable four-token examples for rank-shard checks."""

    starts = np.arange(start, start + count * 4, 4, dtype=np.uint16)
    sequences = starts[:, None] + np.arange(4, dtype=np.uint16)
    sequences.tofile(path)


def test_weighted_sources_load_from_yaml_and_legacy_path_still_works(tmp_path: Path) -> None:
    first = tmp_path / "first.uint16"
    second = tmp_path / "second.uint16"
    _write_repeated_sequences(first, [1, 2, 3, 4])
    _write_repeated_sequences(second, [10, 11, 12, 13])
    config_path = tmp_path / "multi.yaml"
    config_path.write_text(
        f"""\
checkpoints:
  checkpoint_interval: 10
model:
  model_config:
    vocab_size: 32
    hidden_size: 16
    intermediate_size: 32
    num_hidden_layers: 1
    num_attention_heads: 4
optimizer:
  learning_rate_scheduler:
    learning_rate: 0.001
parallelism: {{}}
tokens:
  sequence_length: 4
  micro_batch_size: 2
  train_steps: 1
data:
  datasets:
    - path: {first.as_posix()}
      weight: 0.7
    - path: {second.as_posix()}
      weight: 0.3
  num_workers: 0
logging: {{}}
general: {{}}
""",
        encoding="utf-8",
    )

    multi_config = load_config(config_path)
    assert [(source.path, source.weight) for source in multi_config.data.dataset_sources] == [
        (str(first).replace("\\", "/"), 0.7),
        (str(second).replace("\\", "/"), 0.3),
    ]

    legacy_config = make_test_config(
        sequence_length=4,
        micro_batch_size=2,
        data=DataConfig(dataset_token_path=str(first), num_workers=0),
    )
    legacy_batch = next(iter(create_memmap_dataloader(legacy_config)))
    assert legacy_batch.shape == (2, 4)
    assert torch.all(legacy_batch[0] == torch.tensor([1, 2, 3, 4]))
    assert legacy_config.data.dataset_sources == (DatasetSourceConfig(path=str(first)),)


def test_weighted_interleaving_matches_configured_mix(tmp_path: Path) -> None:
    first = tmp_path / "first.uint16"
    second = tmp_path / "second.uint16"
    _write_repeated_sequences(first, [1, 1, 1, 1])
    _write_repeated_sequences(second, [2, 2, 2, 2])
    config = make_test_config(
        sequence_length=4,
        micro_batch_size=3,
        data=DataConfig(
            datasets=(
                DatasetSourceConfig(path=str(first), weight=0.7),
                DatasetSourceConfig(path=str(second), weight=0.3),
            ),
            num_workers=0,
        ),
    )

    iterator = iter(create_memmap_dataloader(config))
    observed_sources = [int(next(iterator)[0, 0]) for _ in range(1_000)]
    first_share = observed_sources.count(1) / len(observed_sources)

    print(f"first_source_share={first_share:.3f}")
    assert 0.64 <= first_share <= 0.76


def test_weighted_interleaving_keeps_ddp_rank_partitions_disjoint(tmp_path: Path) -> None:
    first = tmp_path / "first.uint16"
    second = tmp_path / "second.uint16"
    _write_indexed_sequences(first, start=0)
    _write_indexed_sequences(second, start=10_000)
    config = make_test_config(
        sequence_length=4,
        micro_batch_size=2,
        data=DataConfig(
            datasets=(
                DatasetSourceConfig(path=str(first), weight=0.5),
                DatasetSourceConfig(path=str(second), weight=0.5),
            ),
            num_workers=0,
        ),
    )

    rank_zero = iter(create_memmap_dataloader(config, rank=0, world_size=2))
    rank_one = iter(create_memmap_dataloader(config, rank=1, world_size=2))
    zero_ids = {int(token) for _ in range(100) for token in next(rank_zero)[:, 0]}
    one_ids = {int(token) for _ in range(100) for token in next(rank_one)[:, 0]}

    assert zero_ids.isdisjoint(one_ids)


def test_training_loss_decreases_with_weighted_interleaving(tmp_path: Path) -> None:
    first = tmp_path / "first.uint16"
    second = tmp_path / "second.uint16"
    _write_repeated_sequences(first, list(range(8)))
    _write_repeated_sequences(second, list(range(1, 9)))
    config = make_test_config(
        vocab_size=32,
        sequence_length=8,
        micro_batch_size=2,
        train_steps=50,
        learning_rate=0.01,
        data=DataConfig(
            datasets=(
                DatasetSourceConfig(path=str(first), weight=0.7),
                DatasetSourceConfig(path=str(second), weight=0.3),
            ),
            num_workers=0,
        ),
    )
    config = replace(config, logging=replace(config.logging, file_logging=False))
    model = PicotronDecoderModel(config)
    losses = train(model, create_memmap_dataloader(config), config, max_steps=50)

    first_average = sum(losses[:10]) / 10
    last_average = sum(losses[-10:]) / 10
    print(f"first_10_avg={first_average:.6f} last_10_avg={last_average:.6f}")
    assert last_average < first_average
