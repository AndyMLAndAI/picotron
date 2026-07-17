"""CPU-safe fallbacks for the optional Triton RMSNorm, SwiGLU, and RoPE kernels."""

import pytest
import torch
from torch.nn import functional as F

from picotron.models.toy_model import RMSNorm
from picotron.nn.feedforward import SwiGLU
from picotron.nn.rope import RotaryEmbedding
from picotron.nn.triton_kernels import (
    triton_apply_rotary_embedding,
    triton_rms_norm,
    triton_swiglu,
)


def test_triton_kernel_modules_import_without_cuda() -> None:
    assert callable(triton_rms_norm)
    assert callable(triton_swiglu)
    assert callable(triton_apply_rotary_embedding)


def test_triton_enabled_cpu_fallbacks_match_native_pytorch() -> None:
    hidden_states = torch.randn(2, 3, 8)
    norm = RMSNorm(8, use_triton_rmsnorm=True)
    with pytest.warns(RuntimeWarning, match="Triton RMSNorm unavailable"):
        normalized = norm(hidden_states)
    expected_norm = hidden_states * torch.rsqrt(
        hidden_states.pow(2).mean(dim=-1, keepdim=True) + norm.eps
    )
    torch.testing.assert_close(normalized, expected_norm * norm.weight)

    swiglu = SwiGLU(8, 16, use_triton_swiglu=True)
    with pytest.warns(RuntimeWarning, match="Triton SwiGLU unavailable"):
        fused_output = swiglu(hidden_states)
    expected_swiglu = swiglu.down_projection(
        F.silu(swiglu.gate_projection(hidden_states)) * swiglu.up_projection(hidden_states)
    )
    torch.testing.assert_close(fused_output, expected_swiglu)

    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 4, 3, 8)
    triton_rope = RotaryEmbedding(8, use_triton_rope=True)
    native_rope = RotaryEmbedding(8)
    with pytest.warns(RuntimeWarning, match="Triton RoPE unavailable"):
        triton_query, triton_key = triton_rope(query, key)
    native_query, native_key = native_rope(query, key)
    torch.testing.assert_close(triton_query, native_query)
    torch.testing.assert_close(triton_key, native_key)
