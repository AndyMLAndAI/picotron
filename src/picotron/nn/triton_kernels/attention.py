"""Optional tiled causal-attention forward kernel with an eager-safe fallback."""

from __future__ import annotations

import math

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


class TritonAttentionUnavailable(RuntimeError):
    """Raised when fused causal attention cannot safely execute."""


if _TRITON_AVAILABLE:

    @triton.jit
    def _tiled_causal_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        sequence_length: tl.constexpr,
        num_query_heads: tl.constexpr,
        num_key_value_heads: tl.constexpr,
        scale,
        WINDOW_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """FlashAttention-style online softmax for one batch/head/query tile."""

        query_block = tl.program_id(axis=0)
        batch_and_head = tl.program_id(axis=1)
        batch_index = batch_and_head // num_query_heads
        query_head_index = batch_and_head % num_query_heads
        key_value_head_index = query_head_index // (
            num_query_heads // num_key_value_heads
        )

        query_positions = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_positions = tl.arange(0, BLOCK_N)
        dimension_offsets = tl.arange(0, BLOCK_D)
        query_mask = (query_positions[:, None] < sequence_length) & (
            dimension_offsets[None, :] < HEAD_DIM
        )
        query_offsets = (
            ((batch_index * num_query_heads + query_head_index) * sequence_length)
            + query_positions[:, None]
        ) * HEAD_DIM + dimension_offsets[None, :]
        query = tl.load(query_ptr + query_offsets, mask=query_mask, other=0.0)

        running_maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_block_start in range(0, sequence_length, BLOCK_N):
            current_key_positions = key_block_start + key_positions
            key_mask = (current_key_positions[:, None] < sequence_length) & (
                dimension_offsets[None, :] < HEAD_DIM
            )
            key_value_offsets = (
                ((batch_index * num_key_value_heads + key_value_head_index) * sequence_length)
                + current_key_positions[:, None]
            ) * HEAD_DIM + dimension_offsets[None, :]
            key = tl.load(key_ptr + key_value_offsets, mask=key_mask, other=0.0)
            value = tl.load(value_ptr + key_value_offsets, mask=key_mask, other=0.0)

            scores = tl.dot(query, tl.trans(key)) * scale
            causal_mask = current_key_positions[None, :] <= query_positions[:, None]
            if WINDOW_SIZE > 0:
                causal_mask = causal_mask & (
                    current_key_positions[None, :]
                    >= query_positions[:, None] - WINDOW_SIZE + 1
                )
            valid_scores = causal_mask & (query_positions[:, None] < sequence_length) & (
                current_key_positions[None, :] < sequence_length
            )
            scores = tl.where(valid_scores, scores, -float("inf"))

            block_maximum = tl.max(scores, axis=1)
            next_maximum = tl.maximum(running_maximum, block_maximum)
            rescale = tl.exp(running_maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum[:, None])
            accumulator = accumulator * rescale[:, None] + tl.dot(probabilities, value)
            running_sum = running_sum * rescale + tl.sum(probabilities, axis=1)
            running_maximum = next_maximum

        output = accumulator / running_sum[:, None]
        output_offsets = (
            ((batch_index * num_query_heads + query_head_index) * sequence_length)
            + query_positions[:, None]
        ) * HEAD_DIM + dimension_offsets[None, :]
        tl.store(output_ptr + output_offsets, output, mask=query_mask)


def triton_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    sliding_window_size: int | None = None,
) -> Tensor:
    """Run tiled causal MHA/GQA attention or raise a safe-unavailable error.

    Inputs use ``(batch, heads, sequence, head_dim)``.  K/V may have fewer
    heads than Q; the kernel maps each Q head to its owning K/V head without
    allocating expanded K/V tensors.  This is inference-forward only: autograd
    deliberately falls back to the tested eager implementation.
    """

    if not _TRITON_AVAILABLE:
        raise TritonAttentionUnavailable("Triton is not installed.")
    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        raise TritonAttentionUnavailable(
            "Triton attention currently supports no-grad inference only."
        )
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise TritonAttentionUnavailable("Triton attention requires CUDA tensors.")
    if torch.cuda.get_device_capability(query.device)[0] < 7:
        raise TritonAttentionUnavailable(
            "Triton attention requires compute capability 7.0 or newer."
        )
    if query.device != key.device or query.device != value.device:
        raise TritonAttentionUnavailable("Q, K, and V must be on the same CUDA device.")
    if query.dtype != torch.float16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise TritonAttentionUnavailable("Triton attention currently supports matching fp16 Q/K/V.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise TritonAttentionUnavailable("Q, K, and V must have shape (batch, heads, sequence, head_dim).")
    batch_size, num_query_heads, sequence_length, head_dim = query.shape
    if (
        key.shape[0] != batch_size
        or value.shape[0] != batch_size
        or key.shape[2:] != (sequence_length, head_dim)
        or value.shape != key.shape
    ):
        raise TritonAttentionUnavailable("Q, K, and V shapes are incompatible.")
    num_key_value_heads = key.size(1)
    if num_key_value_heads <= 0 or num_query_heads % num_key_value_heads != 0:
        raise TritonAttentionUnavailable("K/V heads must divide the number of Q heads.")
    if head_dim > 128 or triton.next_power_of_2(head_dim) != head_dim:
        raise TritonAttentionUnavailable(
            "Triton attention supports power-of-two head dimensions up to 128."
        )
    if sliding_window_size is not None and sliding_window_size <= 0:
        raise TritonAttentionUnavailable("sliding_window_size must be positive when provided.")
    if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
        raise TritonAttentionUnavailable("Triton attention requires contiguous Q/K/V tensors.")

    output = torch.empty_like(query)
    block_m = 64
    block_n = 64
    _tiled_causal_attention_kernel[
        (triton.cdiv(sequence_length, block_m), batch_size * num_query_heads)
    ](
        query,
        key,
        value,
        output,
        sequence_length=sequence_length,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        scale=1.0 / math.sqrt(head_dim),
        WINDOW_SIZE=sliding_window_size if sliding_window_size is not None else -1,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
        num_warps=4,
        num_stages=1,
    )
    return output
