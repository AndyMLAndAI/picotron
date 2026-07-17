"""Eager multi-head causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from picotron.nn.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an eager causal attention path."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        *,
        num_key_value_heads: int | None = None,
        sliding_window_size: int | None = None,
        use_rope: bool = True,
        rope_theta: float = 10_000.0,
        use_triton_rope: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_key_value_heads <= 0 or num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_key_value_heads must be positive and divide num_attention_heads."
            )
        if sliding_window_size is not None and sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive when provided.")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window_size = sliding_window_size
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rotary_embedding = (
            RotaryEmbedding(self.head_dim, rope_theta, use_triton_rope=use_triton_rope)
            if use_rope
            else None
        )

        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        kv_hidden_size = num_key_value_heads * self.head_dim
        self.key_projection = nn.Linear(hidden_size, kv_hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, kv_hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply causal self-attention to ``(batch, sequence, hidden)`` states."""

        self._validate_hidden_states(hidden_states)
        query = self._split_heads(self.query_projection(hidden_states), self.num_attention_heads)
        key = self._split_heads(self.key_projection(hidden_states), self.num_key_value_heads)
        value = self._split_heads(self.value_projection(hidden_states), self.num_key_value_heads)
        if self.rotary_embedding is not None:
            query, key = self.rotary_embedding(query, key)
        key = self._repeat_key_value_heads(key)
        value = self._repeat_key_value_heads(value)

        attention_scores = (query @ key.transpose(-2, -1)) * self.scale
        attention_scores = attention_scores.masked_fill(
            ~self._causal_mask(
                hidden_states.size(1), hidden_states.device, self.sliding_window_size
            ),
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_output = attention_weights.to(value.dtype) @ value

        batch_size, _, sequence_length, _ = attention_output.shape
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.hidden_size
        )
        return self.output_projection(attention_output)

    def _split_heads(self, projected_states: Tensor, num_heads: int) -> Tensor:
        batch_size, sequence_length, _ = projected_states.shape
        return projected_states.view(
            batch_size,
            sequence_length,
            num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _repeat_key_value_heads(self, states: Tensor) -> Tensor:
        """Expand GQA key/value heads to the query-head count."""

        if self.num_key_value_heads == self.num_attention_heads:
            return states
        return states.repeat_interleave(
            self.num_attention_heads // self.num_key_value_heads, dim=1
        )

    @staticmethod
    def _causal_mask(
        sequence_length: int,
        device: torch.device,
        sliding_window_size: int | None,
    ) -> Tensor:
        positions = torch.arange(sequence_length, device=device)
        query_positions = positions.unsqueeze(1)
        key_positions = positions.unsqueeze(0)
        mask = key_positions <= query_positions
        if sliding_window_size is not None:
            mask &= key_positions >= query_positions - sliding_window_size + 1
        return mask

    def _validate_hidden_states(self, hidden_states: Tensor) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape (batch, sequence_length, hidden_size); "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size {self.hidden_size}; got {hidden_states.size(-1)}."
            )
