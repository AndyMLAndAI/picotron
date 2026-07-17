"""A small decoder-only transformer used to validate the project scaffold."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from picotron.config.config import PicotronConfig
from picotron.nn.attention import CausalSelfAttention
from picotron.nn.triton_kernels.rmsnorm import triton_rms_norm


class RMSNorm(nn.Module):
    """Root mean square normalization with a learned scale."""

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, *, use_triton_rmsnorm: bool = False
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.use_triton_rmsnorm = use_triton_rmsnorm
        self._triton_fallback_warned = False

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_triton_rmsnorm:
            try:
                return triton_rms_norm(hidden_states, self.weight, self.eps)
            except Exception as error:
                if not self._triton_fallback_warned:
                    warnings.warn(
                        f"Triton RMSNorm unavailable; using PyTorch fallback: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._triton_fallback_warned = True
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class SwiGLU(nn.Module):
    """Llama-style gated feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_projection = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = F.silu(self.gate_projection(hidden_states))
        return self.down_projection(gated * self.up_projection(hidden_states))


class DecoderBlock(nn.Module):
    """Pre-normalized causal decoder block with custom self-attention."""

    def __init__(self, config: PicotronConfig) -> None:
        super().__init__()
        use_triton_rmsnorm = bool(config.model_kwargs.get("use_triton_rmsnorm", False))
        self.attention_norm = RMSNorm(
            config.hidden_size, use_triton_rmsnorm=use_triton_rmsnorm
        )
        # This attribute remains the isolated attention implementation seam.
        self.attention = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, use_triton_rmsnorm=use_triton_rmsnorm)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self._apply_causal_attention(
            self.attention_norm(hidden_states)
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))

    def _apply_causal_attention(self, hidden_states: Tensor) -> Tensor:
        """Apply the swappable causal-attention implementation."""

        return self.attention(hidden_states)


class ToyDecoderModel(nn.Module):
    """Config-driven decoder-only transformer returning vocabulary logits.

    This reference model consumes only Picotron's core dimensions. Family-specific
    ``config.model_kwargs`` are intentionally left to their concrete model loader.
    """

    def __init__(self, config: PicotronConfig) -> None:
        super().__init__()
        self.config = config
        use_triton_rmsnorm = bool(config.model_kwargs.get("use_triton_rmsnorm", False))
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, use_triton_rmsnorm=use_triton_rmsnorm)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return logits of shape ``(batch, sequence_length, vocab_size)``."""

        self._validate_input_ids(input_ids)
        sequence_length = input_ids.size(1)
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings(positions).unsqueeze(0)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.output_projection(self.final_norm(hidden_states))

    def _validate_input_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch, sequence_length); "
                f"got {tuple(input_ids.shape)}."
            )
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {input_ids.size(1)} exceeds max_seq_len "
                f"{self.config.max_seq_len}."
            )
        if input_ids.is_floating_point() or input_ids.is_complex():
            raise TypeError("input_ids must use an integer tensor dtype.")
