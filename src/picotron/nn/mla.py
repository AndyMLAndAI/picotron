"""Simplified Multi-head Latent Attention with a compressed K/V cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from picotron.nn.rope import RotaryEmbedding


@dataclass(frozen=True)
class MLALatentKVCache:
    """Compressed K/V states retained by the simplified MLA implementation."""

    latent_states: Tensor

    @property
    def numel(self) -> int:
        """Number of elements stored for the compressed K/V cache."""

        return self.latent_states.numel()

    @property
    def nbytes(self) -> int:
        """Byte size of the compressed K/V cache."""

        return self.latent_states.numel() * self.latent_states.element_size()


class MultiHeadLatentAttention(nn.Module):
    """Causal self-attention whose K/V states are stored in a shared latent space.

    This intentionally simplified MLA reconstructs full K/V tensors before the
    attention score calculation and applies conventional RoPE to reconstructed
    Q/K. DeepSeek's decoupled RoPE subspace is not implemented here.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_lora_rank: int,
        *,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        use_triton_rope: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_attention_heads <= 0:
            raise ValueError("hidden_size and num_attention_heads must be positive.")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if kv_lora_rank <= 0 or kv_lora_rank >= 2 * hidden_size:
            raise ValueError("kv_lora_rank must be positive and compress full K/V states.")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.kv_lora_rank = kv_lora_rank
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rotary_embedding = (
            RotaryEmbedding(self.head_dim, rope_theta, use_triton_rope=use_triton_rope)
            if use_rope
            else None
        )
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv_down_projection = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.key_up_projection = nn.Linear(kv_lora_rank, hidden_size, bias=False)
        self.value_up_projection = nn.Linear(kv_lora_rank, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_kv_cache: MLALatentKVCache | None = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply causal attention and retain only compressed K/V cache states."""

        self._validate_hidden_states(hidden_states)
        latent_states = self.kv_down_projection(hidden_states)
        self.last_kv_cache = MLALatentKVCache(latent_states.detach())
        query = self._split_heads(self.query_projection(hidden_states))
        key = self._split_heads(self.key_up_projection(latent_states))
        value = self._split_heads(self.value_up_projection(latent_states))
        if self.rotary_embedding is not None:
            query, key = self.rotary_embedding(query, key)

        attention_scores = (query @ key.transpose(-2, -1)) * self.scale
        attention_scores = attention_scores.masked_fill(
            ~self._causal_mask(hidden_states.size(1), hidden_states.device),
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_output = attention_weights.to(value.dtype) @ value
        batch_size, _, sequence_length, _ = attention_output.shape
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.hidden_size
        )
        return self.output_projection(attention_output)

    def full_kv_cache_numel(self, batch_size: int, sequence_length: int) -> int:
        """Return the element count of an uncompressed K/V cache for comparison."""

        return 2 * batch_size * sequence_length * self.hidden_size

    def _split_heads(self, projected_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = projected_states.shape
        return projected_states.view(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)

    @staticmethod
    def _causal_mask(sequence_length: int, device: torch.device) -> Tensor:
        positions = torch.arange(sequence_length, device=device)
        return positions.unsqueeze(0) <= positions.unsqueeze(1)

    def _validate_hidden_states(self, hidden_states: Tensor) -> None:
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                "hidden_states must have shape (batch, sequence_length, hidden_size)."
            )
