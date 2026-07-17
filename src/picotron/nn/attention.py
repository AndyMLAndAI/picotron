"""Eager multi-head causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an eager causal attention path."""

    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply causal self-attention to ``(batch, sequence, hidden)`` states."""

        self._validate_hidden_states(hidden_states)
        query = self._split_heads(self.query_projection(hidden_states))
        key = self._split_heads(self.key_projection(hidden_states))
        value = self._split_heads(self.value_projection(hidden_states))

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

    def _split_heads(self, projected_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = projected_states.shape
        return projected_states.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

    @staticmethod
    def _causal_mask(sequence_length: int, device: torch.device) -> Tensor:
        return torch.ones(
            (sequence_length, sequence_length), dtype=torch.bool, device=device
        ).tril()

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

