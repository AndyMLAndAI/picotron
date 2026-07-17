"""Optional fused Triton RoPE rotation inference-forward kernel."""

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


class TritonRoPEUnavailable(RuntimeError):
    """Raised when the optional Triton RoPE path cannot safely run."""


if _TRITON_AVAILABLE:

    @triton.jit
    def _rope_kernel(
        states_ptr,
        cosine_ptr,
        sine_ptr,
        output_ptr,
        num_pairs,
        sequence_length: tl.constexpr,
        half_head_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pair_offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = pair_offsets < num_pairs
        pair_in_head = pair_offsets % half_head_dim
        sequence_index = (pair_offsets // half_head_dim) % sequence_length
        state_offsets = pair_offsets * 2
        even = tl.load(states_ptr + state_offsets, mask=mask, other=0.0)
        odd = tl.load(states_ptr + state_offsets + 1, mask=mask, other=0.0)
        table_offsets = sequence_index * half_head_dim + pair_in_head
        cosine = tl.load(cosine_ptr + table_offsets, mask=mask, other=0.0)
        sine = tl.load(sine_ptr + table_offsets, mask=mask, other=0.0)
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        # Modern Triton dtype API: element_ty, not the removed dtype_element.
        tl.store(output_ptr + state_offsets, rotated_even.to(even.dtype.element_ty), mask=mask)
        tl.store(output_ptr + state_offsets + 1, rotated_odd.to(even.dtype.element_ty), mask=mask)


def triton_apply_rotary_embedding(states: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Run fused RoPE rotation or raise ``TritonRoPEUnavailable`` safely.

    The optional kernel is inference-forward only; autograd uses the native path.
    """

    if not _TRITON_AVAILABLE:
        raise TritonRoPEUnavailable("Triton is not installed.")
    if torch.is_grad_enabled() and states.requires_grad:
        raise TritonRoPEUnavailable("Triton RoPE currently supports no-grad inference only.")
    if not states.is_cuda or not cosine.is_cuda or not sine.is_cuda:
        raise TritonRoPEUnavailable("Triton RoPE requires CUDA tensors.")
    if torch.cuda.get_device_capability(states.device)[0] < 7:
        raise TritonRoPEUnavailable("Triton RoPE requires compute capability 7.0+.")
    if (
        states.ndim != 4
        or states.size(-1) != cosine.size(-1) * 2
        or cosine.shape != sine.shape
        or cosine.size(0) != states.size(-2)
        or not states.is_contiguous()
        or not cosine.is_contiguous()
        or not sine.is_contiguous()
    ):
        raise TritonRoPEUnavailable("Triton RoPE requires compatible contiguous tensors.")

    output = torch.empty_like(states)
    half_head_dim = cosine.size(-1)
    num_pairs = states.numel() // 2
    block_size = 256
    _rope_kernel[(triton.cdiv(num_pairs, block_size),)](
        states,
        cosine,
        sine,
        output,
        num_pairs,
        sequence_length=states.size(-2),
        half_head_dim=half_head_dim,
        BLOCK_SIZE=block_size,
    )
    return output
