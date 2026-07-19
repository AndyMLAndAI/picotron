"""Simplified MLA shape, cache-compression, and learning checks."""

import torch

from config_factory import make_test_config
from picotron.config.config import PicotronConfig
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.nn.mla import MultiHeadLatentAttention
from picotron.training.train_loop import train


def _config() -> PicotronConfig:
    return make_test_config(
        learning_rate=0.003,
        attention_type="mla",
        kv_lora_rank=8,
    )


def test_mla_model_forward_shape_and_latent_cache_compression() -> None:
    config = _config()
    model = PicotronDecoderModel(config)
    input_ids = torch.randint(
        0,
        config.model.model_config.vocab_size,
        (config.tokens.micro_batch_size, config.tokens.sequence_length),
    )

    logits = model(input_ids)
    attention = model.layers[0].attention

    assert isinstance(attention, MultiHeadLatentAttention)
    assert logits.shape == (
        config.tokens.micro_batch_size,
        config.tokens.sequence_length,
        config.model.model_config.vocab_size,
    )
    assert attention.last_kv_cache is not None
    assert attention.last_kv_cache.numel == (
        config.tokens.micro_batch_size * config.tokens.sequence_length * 8
    )
    assert attention.last_kv_cache.numel < attention.full_kv_cache_numel(
        config.tokens.micro_batch_size, config.tokens.sequence_length
    )


def test_mla_training_loss_decreases() -> None:
    config = _config()
    model = PicotronDecoderModel(config)
    tokens = torch.arange(config.tokens.sequence_length).unsqueeze(0).repeat(
        config.tokens.micro_batch_size, 1
    )

    losses = train(model, [tokens] * 40, config, max_steps=40)

    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
