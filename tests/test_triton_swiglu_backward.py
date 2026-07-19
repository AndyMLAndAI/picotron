"""Gradient correctness checks for the custom-autograd Triton SwiGLU path."""

import torch
from torch.nn import functional as F

from picotron.nn.triton_kernels.swiglu import swiglu_with_pytorch_forward


def test_custom_swiglu_backward_matches_native_pytorch() -> None:
    """Compare gate and up gradients with native SiLU-times-up autograd."""

    torch.manual_seed(31)
    upstream_gradient = torch.randn(2, 3, 7, dtype=torch.float64)
    custom_gate = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    custom_up = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    custom_output = swiglu_with_pytorch_forward(custom_gate, custom_up)
    (custom_output * upstream_gradient).sum().backward()

    native_gate = custom_gate.detach().clone().requires_grad_(True)
    native_up = custom_up.detach().clone().requires_grad_(True)
    native_output = F.silu(native_gate) * native_up
    (native_output * upstream_gradient).sum().backward()

    torch.testing.assert_close(custom_output, native_output, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(custom_gate.grad, native_gate.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(custom_up.grad, native_up.grad, rtol=1e-10, atol=1e-10)
