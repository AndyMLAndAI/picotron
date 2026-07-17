"""Optional fused Triton SwiGLU activation inference-forward kernel."""

from __future__ import annotations

import torch
from torch import Tensor

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
        # Modern Triton dtype API: element_ty, not the removed dtype_element.
        tl.store(output_ptr + offsets, output.to(gate.dtype.element_ty), mask=mask)


def triton_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Run fused SwiGLU activation or raise ``TritonSwiGLUUnavailable`` safely.

    The optional kernel is inference-forward only; autograd uses the native path.
    """

    if not _TRITON_AVAILABLE:
        raise TritonSwiGLUUnavailable("Triton is not installed.")
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        raise TritonSwiGLUUnavailable("Triton SwiGLU currently supports no-grad inference only.")
    if not gate.is_cuda or not up.is_cuda:
        raise TritonSwiGLUUnavailable("Triton SwiGLU requires CUDA tensors.")
    if torch.cuda.get_device_capability(gate.device)[0] < 7:
        raise TritonSwiGLUUnavailable("Triton SwiGLU requires compute capability 7.0+.")
    if gate.shape != up.shape or not gate.is_contiguous() or not up.is_contiguous():
        raise TritonSwiGLUUnavailable("Triton SwiGLU requires matching contiguous inputs.")

    output = torch.empty_like(gate)
    block_size = 256
    _swiglu_kernel[(triton.cdiv(gate.numel(), block_size),)](
        gate, up, output, gate.numel(), BLOCK_SIZE=block_size
    )
    return output
