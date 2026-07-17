"""Per-layer NoPE and RoPE regression checks."""

from dataclasses import replace

import torch

from picotron.models.toy_model import ToyDecoderModel
from picotron.training.train_loop import train
from config_factory import make_test_config


def _config(*, nope_layers: tuple[int, ...] = ()):
    return make_test_config(
        num_hidden_layers=2,
        learning_rate=0.003,
        nope_layers=nope_layers,
    )


def test_nope_layer_skips_rotary_embedding() -> None:
    model = ToyDecoderModel(_config(nope_layers=(1,)))
    rope_attention = model.layers[0].attention
    nope_attention = model.layers[1].attention
    nope_attention.load_state_dict(rope_attention.state_dict(), strict=False)
    hidden_states = torch.randn(2, 8, 16)

    rope_output = rope_attention(hidden_states)
    nope_output = nope_attention(hidden_states)

    assert rope_attention.rotary_embedding is not None
    assert nope_attention.rotary_embedding is None
    assert not torch.allclose(rope_output, nope_output)


def test_mixed_rope_nope_model_trains_with_correct_shape() -> None:
    config = _config(nope_layers=(1,))
    model = ToyDecoderModel(config)
    tokens = torch.arange(config.tokens.sequence_length).unsqueeze(0).repeat(
        config.tokens.micro_batch_size, 1
    )

    logits = model(tokens)
    losses = train(model, [tokens] * 40, config, max_steps=40)

    assert logits.shape == (
        config.tokens.micro_batch_size,
        config.tokens.sequence_length,
        config.model.model_config.vocab_size,
    )
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_default_config_retains_all_rope_behavior() -> None:
    torch.manual_seed(7)
    default_model = ToyDecoderModel(_config())
    torch.manual_seed(7)
    base_config = _config()
    explicit_rope_config = replace(
        base_config,
        model=replace(
            base_config.model,
            model_config=replace(base_config.model.model_config, nope_layers=()),
        ),
    )
    explicit_rope_model = ToyDecoderModel(explicit_rope_config)
    input_ids = torch.randint(0, 32, (2, 8))

    torch.testing.assert_close(default_model(input_ids), explicit_rope_model(input_ids))
    assert all(layer.attention.rotary_embedding is not None for layer in default_model.layers)
