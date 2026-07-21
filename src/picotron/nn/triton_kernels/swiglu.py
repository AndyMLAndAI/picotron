"""Optional fused Triton SwiGLU activation with a correct autograd backward."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - expected on CPU-only environments.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


class TritonSwiGLUUnavailable(RuntimeError):
    """Raised when the optional Triton SwiGLU path cannot safely run."""


if _TRITON_AVAILABLE:

    @triton.jit
    def _swiglu_kernel(gate_ptr, up_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
        output = (gate * tl.sigmoid(gate)) * up
        # ``tl.store`` casts to the output pointer element type itself. Value
        # tensors do not expose ``dtype.element_ty`` on Kaggle's Triton build.
        tl.store(output_ptr + offsets, output, mask=mask)


def _run_triton_forward(gate: Tensor, up: Tensor) -> Tensor:
    """Launch the fused elementwise SiLU-times-up kernel."""

    assert triton is not None
    output = torch.empty_like(gate)
    block_size = 256
    _swiglu_kernel[(triton.cdiv(gate.numel(), block_size),)](
        gate, up, output, gate.numel(), BLOCK_SIZE=block_size
    )
    return output


class _SwiGLUAutogradFunction(torch.autograd.Function):
    """Fused Triton forward with a directly auditable PyTorch backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        gate: Tensor,
        up: Tensor,
        use_triton_forward: bool,
    ) -> Tensor:
        ctx.save_for_backward(gate, up)
        if use_triton_forward:
            return _run_triton_forward(gate, up)
        return F.silu(gate) * up

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor | None, Tensor | None, None]:
        gate, up = ctx.saved_tensors
        sigmoid_gate = torch.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        silu_derivative = sigmoid_gate * (1 + gate * (1 - sigmoid_gate))
        grad_gate = grad_output * up * silu_derivative
        grad_up = grad_output * silu_gate
        return grad_gate, grad_up, None


def _validate_triton_inputs(gate: Tensor, up: Tensor) -> None:
    """Validate Triton launch constraints, without rejecting autograd."""

    if not _TRITON_AVAILABLE:
        raise TritonSwiGLUUnavailable("Triton is not installed.")
    if not gate.is_cuda or not up.is_cuda:
        raise TritonSwiGLUUnavailable("Triton SwiGLU requires CUDA tensors.")
    if torch.cuda.get_device_capability(gate.device)[0] < 7:
        raise TritonSwiGLUUnavailable("Triton SwiGLU requires compute capability 7.0+.")
    if gate.shape != up.shape or not gate.is_contiguous() or not up.is_contiguous():
        raise TritonSwiGLUUnavailable("Triton SwiGLU requires matching contiguous inputs.")


def triton_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Run fused SwiGLU forward with a mathematically exact PyTorch backward."""

    _validate_triton_inputs(gate, up)
    return _SwiGLUAutogradFunction.apply(gate, up, True)


def swiglu_with_pytorch_forward(gate: Tensor, up: Tensor) -> Tensor:
    """Exercise the custom backward on CPU without requiring Triton or CUDA."""

    return _SwiGLUAutogradFunction.apply(gate, up, False)
