"""GQA and sliding-window attention checks."""

import torch

from config_factory import make_test_config
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.nn.attention import CausalSelfAttention


def test_gqa_output_shape_with_fewer_key_value_heads() -> None:
    attention = CausalSelfAttention(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    hidden_states = torch.randn(2, 5, 16)

    output = attention(hidden_states)

    assert attention.num_key_value_heads == 2
    assert output.shape == hidden_states.shape


def test_sliding_window_blocks_distant_past_tokens() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(
        hidden_size=8,
        num_attention_heads=2,
        sliding_window_size=2,
    )
    hidden_states = torch.randn(1, 5, 8)
    changed_past_states = hidden_states.clone()
    changed_past_states[:, 0, :] += 1000.0

    output = attention(hidden_states)
    changed_output = attention(changed_past_states)

    torch.testing.assert_close(output[:, 2:, :], changed_output[:, 2:, :])


def test_model_accepts_gqa_and_sliding_window_config() -> None:
    config = make_test_config(
        num_key_value_heads=2,
        attention_type="gqa",
        sliding_window_size=4,
    )
    model = PicotronDecoderModel(config)

    logits = model(torch.randint(0, config.model.model_config.vocab_size, (2, 8)))

    assert logits.shape == (2, 8, config.model.model_config.vocab_size)
