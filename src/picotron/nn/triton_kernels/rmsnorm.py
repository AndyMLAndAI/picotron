"""Optional fused Triton RMSNorm inference-forward kernel."""

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


class TritonRMSNormUnavailable(RuntimeError):
    """Raised when the optional Triton RMSNorm path cannot safely run."""


if _TRITON_AVAILABLE:

    @triton.jit
    def _rmsnorm_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        row_stride,
        hidden_size,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        input_values = tl.load(input_ptr + row * row_stride + offsets, mask=mask, other=0.0)
        input_fp32 = input_values.to(tl.float32)
        variance = tl.sum(input_fp32 * input_fp32, axis=0) / hidden_size
        inverse_rms = tl.rsqrt(variance + eps)
        weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output_values = input_fp32 * inverse_rms * weights
        # Modern Triton dtype API: element_ty, not the removed dtype_element.
        tl.store(
            output_ptr + row * row_stride + offsets,
            output_values.to(input_values.dtype.element_ty),
            mask=mask,
        )


def triton_rms_norm(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Run fused RMSNorm or raise ``TritonRMSNormUnavailable`` safely."""

    if not _TRITON_AVAILABLE:
        raise TritonRMSNormUnavailable("Triton is not installed.")
    if torch.is_grad_enabled() and (hidden_states.requires_grad or weight.requires_grad):
        raise TritonRMSNormUnavailable(
            "Triton RMSNorm currently supports no-grad inference only."
        )
    if not hidden_states.is_cuda:
        raise TritonRMSNormUnavailable("Triton RMSNorm requires a CUDA tensor.")
    if torch.cuda.get_device_capability(hidden_states.device)[0] < 7:
        raise TritonRMSNormUnavailable("Triton RMSNorm requires compute capability 7.0+.")
    if hidden_states.ndim < 1 or hidden_states.size(-1) != weight.numel():
        raise TritonRMSNormUnavailable("RMSNorm hidden size must match the weight size.")
    if not hidden_states.is_contiguous() or not weight.is_contiguous():
        raise TritonRMSNormUnavailable("Triton RMSNorm requires contiguous inputs.")

    hidden_size = hidden_states.size(-1)
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 65536:
        raise TritonRMSNormUnavailable("Triton RMSNorm supports hidden sizes up to 65536.")
    output = torch.empty_like(hidden_states)
    rows = hidden_states.numel() // hidden_size
    _rmsnorm_kernel[(rows,)](
        hidden_states,
        weight,
        output,
        hidden_size,
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output
