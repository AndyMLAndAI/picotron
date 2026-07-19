"""Backend-routing checks for optional xFormers causal attention."""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch

import picotron.nn.attention as attention_module
from picotron.nn.attention import CausalSelfAttention
from picotron.utils.hardware import AttentionBackend, AttentionBackendReport


def test_attention_output_shape_without_xformers() -> None:
    """The standard CPU route remains shape-correct without optional xFormers."""

    attention = CausalSelfAttention(16, 4, num_key_value_heads=2)
    output = attention(torch.randn(2, 5, 16))
    assert output.shape == (2, 5, 16)


@pytest.mark.skipif(find_spec("xformers") is None, reason="xFormers is not installed.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="xFormers attention requires CUDA.")
def test_xformers_matches_eager_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    """A compatible xFormers CUDA kernel must agree with the eager result."""

    import xformers.ops as xops

    device = torch.device("cuda")
    torch.manual_seed(47)
    attention = CausalSelfAttention(
        16, 4, num_key_value_heads=2, use_rope=False
    ).to(device=device, dtype=torch.float16)
    hidden_states = torch.randn(2, 5, 16, device=device, dtype=torch.float16)
    report = AttentionBackendReport(
        selected=AttentionBackend.XFORMERS,
        flash_attention_available=False,
        xformers_available=True,
        sdpa_available=True,
    )
    monkeypatch.setattr(attention_module, "detect_attention_backend", lambda _: report)

    original_xformers_attention = xops.memory_efficient_attention
    xformers_called = False

    def checked_xformers_attention(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal xformers_called
        xformers_called = True
        try:
            return original_xformers_attention(*args, **kwargs)
        except Exception as error:
            pytest.skip(f"No compatible xFormers CUDA attention operator: {error}")

    monkeypatch.setattr(xops, "memory_efficient_attention", checked_xformers_attention)
    xformers_output = attention(hidden_states)
    eager_output = attention._apply_eager_attention(
        attention._split_heads(attention.query_projection(hidden_states), 4),
        attention._repeat_key_value_heads(
            attention._split_heads(attention.key_projection(hidden_states), 2)
        ),
        attention._repeat_key_value_heads(
            attention._split_heads(attention.value_projection(hidden_states), 2)
        ),
    )
    eager_output = attention.output_projection(
        eager_output.transpose(1, 2).contiguous().view(2, 5, 16)
    )

    assert xformers_called
    torch.testing.assert_close(xformers_output, eager_output, rtol=2e-2, atol=2e-2)
