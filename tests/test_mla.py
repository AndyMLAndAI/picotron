"""Simplified MLA shape, cache-compression, and learning checks."""

import torch

from picotron.config.config import PicotronConfig
from picotron.models.toy_model import ToyDecoderModel
from picotron.nn.mla import MultiHeadLatentAttention
from picotron.training.train_loop import train


def _config() -> PicotronConfig:
    return PicotronConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_seq_len=8,
        learning_rate=0.003,
        batch_size=2,
        num_epochs=1,
        checkpoint_interval=100,
        attention_type="mla",
        kv_lora_rank=8,
    )


def test_mla_model_forward_shape_and_latent_cache_compression() -> None:
    config = _config()
    model = ToyDecoderModel(config)
    input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len))

    logits = model(input_ids)
    attention = model.layers[0].attention

    assert isinstance(attention, MultiHeadLatentAttention)
    assert logits.shape == (config.batch_size, config.max_seq_len, config.vocab_size)
    assert attention.last_kv_cache is not None
    assert attention.last_kv_cache.numel == config.batch_size * config.max_seq_len * 8
    assert attention.last_kv_cache.numel < attention.full_kv_cache_numel(
        config.batch_size, config.max_seq_len
    )


def test_mla_training_loss_decreases() -> None:
    config = _config()
    model = ToyDecoderModel(config)
    tokens = torch.arange(config.max_seq_len).unsqueeze(0).repeat(config.batch_size, 1)

    losses = train(model, [tokens] * 40, config, max_steps=40)

    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
