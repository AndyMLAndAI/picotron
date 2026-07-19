"""CPU checks for the synthetic pretraining data pipeline."""

from pathlib import Path
from dataclasses import replace
import gzip

import numpy as np
import torch

from picotron.config.config import load_config
from picotron.data.dataloader import (
    create_memmap_dataloader,
    create_synthetic_dataloader,
)


def test_synthetic_dataloader_batch_shapes_and_dtype() -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/picotron_decoder.yaml"
    config = load_config(config_path)
    loader = create_synthetic_dataloader(
        config, num_sequences=config.tokens.micro_batch_size * 3, seed=7
    )

    batches = list(loader)

    assert len(batches) == 3
    for batch in batches:
        assert batch.shape == (
            config.tokens.micro_batch_size,
            config.tokens.sequence_length,
        )
        assert batch.dtype == torch.long
        assert torch.all((batch >= 0) & (batch < config.model.model_config.vocab_size))


def test_memmap_dataloader_reads_configured_token_cache(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/picotron_decoder.yaml"
    loaded_config = load_config(config_path)
    token_path = tmp_path / "tokens.uint16"
    token_count = loaded_config.tokens.sequence_length * loaded_config.tokens.micro_batch_size
    np.arange(token_count, dtype=np.uint16).tofile(token_path)
    config = replace(
        loaded_config,
        data=replace(loaded_config.data, dataset_token_path=str(token_path)),
    )

    batch = next(iter(create_memmap_dataloader(config)))

    assert batch.shape == (
        config.tokens.micro_batch_size,
        config.tokens.sequence_length,
    )
    assert batch.dtype == torch.long


def test_memmap_dataloader_decompresses_gzip_token_cache(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/picotron_decoder.yaml"
    loaded_config = load_config(config_path)
    token_count = loaded_config.tokens.sequence_length * loaded_config.tokens.micro_batch_size
    token_path = tmp_path / "tokens.uint16.gz"
    with gzip.open(token_path, "wb") as compressed_file:
        compressed_file.write(np.arange(token_count, dtype=np.uint16).tobytes())
    config = replace(
        loaded_config,
        data=replace(loaded_config.data, dataset_token_path=str(token_path)),
    )

    batch = next(iter(create_memmap_dataloader(config)))

    assert batch.shape == (
        config.tokens.micro_batch_size,
        config.tokens.sequence_length,
    )
    assert batch.dtype == torch.long
