"""Gradient checks for the custom-autograd Triton RMSNorm integration."""

import torch

from picotron.nn.triton_kernels.rmsnorm import rms_norm_with_pytorch_forward


def _native_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    inverse_rms = torch.rsqrt(hidden_states.square().mean(dim=-1, keepdim=True) + eps)
    return hidden_states * inverse_rms * weight


def test_custom_rmsnorm_backward_matches_native_rmsnorm() -> None:
    """Verify input and scale gradients without requiring Triton or CUDA."""

    torch.manual_seed(23)
    eps = 1e-6
    upstream_gradient = torch.randn(2, 3, 8, dtype=torch.float32)

    custom_input = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)
    custom_weight = torch.randn(8, dtype=torch.float32, requires_grad=True)
    custom_output = rms_norm_with_pytorch_forward(custom_input, custom_weight, eps)
    (custom_output * upstream_gradient).sum().backward()

    native_input = custom_input.detach().clone().requires_grad_(True)
    native_weight = custom_weight.detach().clone().requires_grad_(True)
    native_output = _native_rms_norm(native_input, native_weight, eps)
    (native_output * upstream_gradient).sum().backward()

    torch.testing.assert_close(custom_output, native_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_input.grad, native_input.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(custom_weight.grad, native_weight.grad, rtol=1e-5, atol=1e-6)
