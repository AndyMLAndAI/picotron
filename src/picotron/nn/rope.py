"""Rotary position embeddings for decoder attention queries and keys."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn

from picotron.nn.triton_kernels.rope import triton_apply_rotary_embedding


def rotary_cos_sin(
    sequence_length: int,
    head_dim: int,
    theta: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Build RoPE cosine and sine tables for one sequence length."""

    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension.")
    if theta <= 0:
        raise ValueError("rope_theta must be positive.")
    inverse_frequencies = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequencies)
    return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def apply_rotary_embedding(states: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate ``(batch, heads, sequence, head_dim)`` states with RoPE tables."""

    if states.size(-1) != cosine.size(-1) * 2:
        raise ValueError("RoPE table dimension does not match the attention head dimension.")
    cosine = cosine.unsqueeze(0).unsqueeze(0)
    sine = sine.unsqueeze(0).unsqueeze(0)
    even_states = states[..., 0::2]
    odd_states = states[..., 1::2]
    rotated = torch.stack(
        (
            even_states * cosine - odd_states * sine,
            even_states * sine + odd_states * cosine,
        ),
        dim=-1,
    )
    return rotated.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    """Apply standard RoPE to attention query and key tensors."""

    def __init__(
        self, head_dim: int, theta: float = 10_000.0, *, use_triton_rope: bool = False
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        if theta <= 0:
            raise ValueError("rope_theta must be positive.")
        self.head_dim = head_dim
        self.theta = theta
        self.use_triton_rope = use_triton_rope
        self._triton_fallback_warned = False

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate query and key using their shared sequence positions."""

        if (
            query.size(0) != key.size(0)
            or query.size(-2) != key.size(-2)
            or query.size(-1) != key.size(-1)
        ):
            raise ValueError("RoPE query and key must share batch, sequence, and head dimensions.")
        cosine, sine = rotary_cos_sin(
            query.size(-2),
            self.head_dim,
            self.theta,
            device=query.device,
            dtype=query.dtype,
        )
        if self.use_triton_rope:
            try:
                return (
                    triton_apply_rotary_embedding(
                        query.contiguous(), cosine.contiguous(), sine.contiguous()
                    ),
                    triton_apply_rotary_embedding(
                        key.contiguous(), cosine.contiguous(), sine.contiguous()
                    ),
                )
            except Exception as error:
                if not self._triton_fallback_warned:
                    warnings.warn(
                        f"Triton RoPE unavailable; using PyTorch fallback: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._triton_fallback_warned = True
        return (
            apply_rotary_embedding(query, cosine, sine),
            apply_rotary_embedding(key, cosine, sine),
        )
