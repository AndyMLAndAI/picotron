"""Optional fused Triton RMSNorm forward with a correct autograd backward."""

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


def _inverse_rms(hidden_states: Tensor, eps: float) -> Tensor:
    """Compute the per-row inverse RMS in fp32, matching the Triton kernel."""

    values = hidden_states.float()
    return torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)


def _pytorch_forward(hidden_states: Tensor, weight: Tensor, inverse_rms: Tensor) -> Tensor:
    """Reference forward with the Triton kernel's fp32 accumulation semantics."""

    output = hidden_states.float() * inverse_rms * weight.float()
    return output.to(dtype=hidden_states.dtype)


def _run_triton_forward(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Launch the fused forward kernel after caller-side capability validation."""

    assert triton is not None
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


class _RMSNormAutogradFunction(torch.autograd.Function):
    """Use a fused forward while preserving exact RMSNorm gradient semantics.

    The backward intentionally uses PyTorch tensor operations. It is easier to
    audit than a second custom kernel and still allows the forward's reduction,
    normalization, and scale to execute as one Triton launch during training.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        hidden_states: Tensor,
        weight: Tensor,
        eps: float,
        use_triton_forward: bool,
    ) -> Tensor:
        inverse_rms = _inverse_rms(hidden_states, eps)
        ctx.save_for_backward(hidden_states, weight, inverse_rms)
        if use_triton_forward:
            return _run_triton_forward(hidden_states, weight, eps)
        return _pytorch_forward(hidden_states, weight, inverse_rms)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor | None, Tensor | None, None, None]:
        hidden_states, weight, inverse_rms = ctx.saved_tensors
        hidden_states_fp32 = hidden_states.float()
        weight_fp32 = weight.float()
        grad_output_fp32 = grad_output.float()

        # y_i = x_i * r * w_i, where r = (mean(x^2) + eps)^(-1/2).
        # Therefore dx = r * (g*w - x*r^2*mean(g*w*x)) per row.
        weighted_grad = grad_output_fp32 * weight_fp32
        dot_mean = (weighted_grad * hidden_states_fp32).mean(dim=-1, keepdim=True)
        grad_hidden_states = inverse_rms * (
            weighted_grad - hidden_states_fp32 * inverse_rms.square() * dot_mean
        )
        grad_weight = (
            grad_output_fp32 * hidden_states_fp32 * inverse_rms
        ).reshape(-1, weight.numel()).sum(dim=0)

        return (
            grad_hidden_states.to(dtype=hidden_states.dtype),
            grad_weight.to(dtype=weight.dtype),
            None,
            None,
        )


def _validate_triton_inputs(hidden_states: Tensor, weight: Tensor) -> None:
    """Validate constraints of the Triton launch, not autograd itself."""

    if not _TRITON_AVAILABLE:
        raise TritonRMSNormUnavailable("Triton is not installed.")
    if not hidden_states.is_cuda:
        raise TritonRMSNormUnavailable("Triton RMSNorm requires a CUDA tensor.")
    if torch.cuda.get_device_capability(hidden_states.device)[0] < 7:
        raise TritonRMSNormUnavailable("Triton RMSNorm requires compute capability 7.0+.")
    if hidden_states.ndim < 1 or hidden_states.size(-1) != weight.numel():
        raise TritonRMSNormUnavailable("RMSNorm hidden size must match the weight size.")
    if not hidden_states.is_contiguous() or not weight.is_contiguous():
        raise TritonRMSNormUnavailable("Triton RMSNorm requires contiguous inputs.")


def triton_rms_norm(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Run Triton-forward RMSNorm with a PyTorch-autograd backward.

    Callers should catch :class:`TritonRMSNormUnavailable` and use their
    native RMSNorm implementation when CUDA/Triton requirements are unmet.
    """

    _validate_triton_inputs(hidden_states, weight)
    return _RMSNormAutogradFunction.apply(hidden_states, weight, eps, True)


def rms_norm_with_pytorch_forward(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Exercise the custom backward using a PyTorch forward for CPU verification.

    This is a testable reference entry point: production callers should use
    :func:`triton_rms_norm`, which selects the fused forward only after GPU
    capability validation.
    """

    return _RMSNormAutogradFunction.apply(hidden_states, weight, eps, False)
