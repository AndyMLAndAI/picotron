"""CPU fallback checks for optional tiled Triton causal attention."""

import pytest
import torch

from picotron.nn.attention import CausalSelfAttention
from picotron.nn.triton_kernels.attention import (
    TritonAttentionUnavailable,
    triton_causal_attention,
)


def test_triton_attention_imports_and_rejects_cpu_tensors_safely() -> None:
    query = torch.randn(1, 2, 4, 4)
    key = torch.randn(1, 1, 4, 4)
    value = torch.randn(1, 1, 4, 4)

    with pytest.raises(TritonAttentionUnavailable):
        triton_causal_attention(query, key, value)


def test_opt_in_triton_attention_falls_back_to_identical_eager_gqa_output() -> None:
    torch.manual_seed(5)
    eager = CausalSelfAttention(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window_size=2,
        use_rope=False,
    )
    fallback = CausalSelfAttention(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window_size=2,
        use_rope=False,
        use_triton_attention=True,
    )
    fallback.load_state_dict(eager.state_dict())
    hidden_states = torch.randn(2, 6, 16)

    eager_output = eager(hidden_states)
    with pytest.warns(RuntimeWarning, match="Triton attention unavailable"):
        fallback_output = fallback(hidden_states)

    torch.testing.assert_close(fallback_output, eager_output, rtol=0, atol=0)
