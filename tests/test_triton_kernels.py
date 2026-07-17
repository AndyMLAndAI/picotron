"""CPU-safe import and fallback checks for optional Triton RMSNorm."""

import pytest
import torch

from picotron.models.toy_model import RMSNorm
from picotron.nn.triton_kernels import triton_rms_norm
from picotron.utils.hardware import detect_triton_support


def test_triton_module_imports_without_cuda() -> None:
    report = detect_triton_support()

    assert not report.enabled
    assert not report.available
    assert callable(triton_rms_norm)


def test_triton_enabled_rmsnorm_falls_back_to_pytorch_on_cpu() -> None:
    hidden_states = torch.randn(2, 3, 8)
    norm = RMSNorm(8, use_triton_rmsnorm=True)

    with pytest.warns(RuntimeWarning, match="Triton RMSNorm unavailable"):
        output = norm(hidden_states)

    expected = hidden_states * torch.rsqrt(
        hidden_states.pow(2).mean(dim=-1, keepdim=True) + norm.eps
    )
    torch.testing.assert_close(output, expected * norm.weight)
