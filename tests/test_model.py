"""CPU sanity checks for the toy decoder model."""

from pathlib import Path

import torch

from picotron.config.config import load_config
from picotron.models.picotron_decoder import PicotronDecoderModel


def test_picotron_decoder_forward_shape() -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/picotron_decoder.yaml"
    config = load_config(config_path)
    model_config = config.model.model_config
    model = PicotronDecoderModel(config)
    input_ids = torch.randint(
        low=0,
        high=model_config.vocab_size,
        size=(config.tokens.micro_batch_size, config.tokens.sequence_length),
    )

    logits = model(input_ids)

    assert logits.shape == (
        config.tokens.micro_batch_size,
        config.tokens.sequence_length,
        model_config.vocab_size,
    )
