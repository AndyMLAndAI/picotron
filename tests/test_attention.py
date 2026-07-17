"""CPU tests for eager causal self-attention."""

import torch

from picotron.nn.attention import CausalSelfAttention


def test_causal_self_attention_output_shape() -> None:
    attention = CausalSelfAttention(hidden_size=8, num_attention_heads=2)
    hidden_states = torch.randn(2, 4, 8)

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape


def test_causal_self_attention_blocks_future_tokens() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(hidden_size=8, num_attention_heads=2)
    hidden_states = torch.randn(1, 4, 8)
    changed_future_states = hidden_states.clone()
    changed_future_states[:, 3, :] += 1000.0

    output = attention(hidden_states)
    changed_output = attention(changed_future_states)

    torch.testing.assert_close(output[:, :3, :], changed_output[:, :3, :])
