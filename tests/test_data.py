"""CPU checks for the synthetic pretraining data pipeline."""

from pathlib import Path

import torch

from picotron.config.config import load_config
from picotron.data.dataloader import create_synthetic_dataloader


def test_synthetic_dataloader_batch_shapes_and_dtype() -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/toy_model.yaml"
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
