"""Picotron's configurable native decoder-only transformer."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from picotron.config.config import PicotronConfig
from picotron.nn.attention import CausalSelfAttention
from picotron.nn.feedforward import SwiGLU
from picotron.nn.mla import MultiHeadLatentAttention
from picotron.nn.moe import MoEFeedForward
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


class TiedOutputProjection(nn.Module):
    """Vocabulary projection that shares the token-embedding parameter.

    The embedding reference intentionally is not registered as a child module:
    the parent model already owns it, and registering it again would create
    aliased entries in ``state_dict`` that ``safetensors.save_file`` rejects.
    """

    def __init__(self, token_embeddings: nn.Embedding) -> None:
        super().__init__()
        object.__setattr__(self, "_token_embeddings", token_embeddings)

    @property
    def weight(self) -> nn.Parameter:
        """Expose the shared vocabulary weight for inspection and tooling."""

        return self._token_embeddings.weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.linear(hidden_states, self.weight)


class DecoderBlock(nn.Module):
    """Pre-normalized causal decoder block with custom self-attention."""

    def __init__(self, config: PicotronConfig, *, use_rope: bool) -> None:
        super().__init__()
        model_config = config.model.model_config
        triton_kernels = config.model.triton_kernels
        use_triton_rmsnorm = triton_kernels.rmsnorm
        self.attention_norm = RMSNorm(
            model_config.hidden_size, use_triton_rmsnorm=use_triton_rmsnorm
        )
        # This attribute remains the isolated attention implementation seam.
        self.attention = _build_attention(
            config,
            use_rope=use_rope,
        )
        self.mlp_norm = RMSNorm(
            model_config.hidden_size, use_triton_rmsnorm=use_triton_rmsnorm
        )
        self.mlp = (
            MoEFeedForward(
                model_config.hidden_size,
                model_config.intermediate_size,
                model_config.moe_config,
            )
            if model_config.moe_config is not None
            else SwiGLU(
                model_config.hidden_size,
                model_config.intermediate_size,
                use_triton_swiglu=triton_kernels.swiglu,
            )
        )
        self.auxiliary_loss: Tensor | None = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return transformed states while retaining the legacy block interface."""

        hidden_states, auxiliary_loss = self.forward_with_auxiliary_loss(hidden_states)
        self.auxiliary_loss = auxiliary_loss
        return hidden_states

    def forward_with_auxiliary_loss(self, hidden_states: Tensor) -> tuple[Tensor, Tensor | None]:
        """Return block states and its differentiable MoE auxiliary loss, if any."""

        hidden_states = hidden_states + self._apply_causal_attention(
            self.attention_norm(hidden_states)
        )
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        if isinstance(self.mlp, MoEFeedForward):
            mlp_output, auxiliary_loss = mlp_output
            return hidden_states + mlp_output, auxiliary_loss
        return hidden_states + mlp_output, None

    def _apply_causal_attention(self, hidden_states: Tensor) -> Tensor:
        """Apply the swappable causal-attention implementation."""

        return self.attention(hidden_states)


class PicotronDecoderModel(nn.Module):
    """Config-driven decoder-only transformer returning vocabulary logits.

    RoPE is the default position scheme; set
    ``config.model.model_config.position_embedding_type`` to ``"learned"``
    to retain learned embeddings.
    """

    def __init__(self, config: PicotronConfig) -> None:
        super().__init__()
        self.config = config
        model_config = config.model.model_config
        triton_kernels = config.model.triton_kernels
        use_triton_rmsnorm = triton_kernels.rmsnorm
        self.position_embedding_type = _position_embedding_type(config)
        self.gradient_checkpointing = model_config.gradient_checkpointing
        self.token_embeddings = nn.Embedding(
            model_config.vocab_size, model_config.hidden_size
        )
        self.position_embeddings = (
            None
            if self.position_embedding_type == "rope"
            else nn.Embedding(
                config.tokens.sequence_length,
                model_config.hidden_size,
            )
        )
        self.layers = nn.ModuleList(
            DecoderBlock(
                config,
                use_rope=self.position_embedding_type == "rope"
                and layer_index not in model_config.nope_layers,
            )
            for layer_index in range(model_config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(
            model_config.hidden_size,
            use_triton_rmsnorm=use_triton_rmsnorm,
        )
        self.output_projection: nn.Module = (
            TiedOutputProjection(self.token_embeddings)
            if model_config.tie_word_embeddings
            else nn.Linear(
                model_config.hidden_size,
                model_config.vocab_size,
                bias=False,
            )
        )
        self.auxiliary_loss: Tensor | None = None

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return logits of shape ``(batch, sequence_length, vocab_size)``."""

        self._validate_input_ids(input_ids)
        sequence_length = input_ids.size(1)
        hidden_states = self.token_embeddings(input_ids)
        if self.position_embeddings is not None:
            positions = torch.arange(sequence_length, device=input_ids.device)
            hidden_states = hidden_states + self.position_embeddings(positions).unsqueeze(0)

        auxiliary_losses: list[Tensor] = []
        for layer in self.layers:
            hidden_states, auxiliary_loss = self._run_layer(layer, hidden_states)
            layer.auxiliary_loss = auxiliary_loss
            if auxiliary_loss is not None:
                auxiliary_losses.append(auxiliary_loss)

        self.auxiliary_loss = sum(auxiliary_losses) if auxiliary_losses else None

        return self.output_projection(self.final_norm(hidden_states))

    def _run_layer(
        self, layer: DecoderBlock, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Checkpoint a block only while training and preserve MoE aux gradients."""

        if not (self.gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return layer.forward_with_auxiliary_loss(hidden_states)
        if isinstance(layer.mlp, MoEFeedForward):
            return checkpoint(
                layer.forward_with_auxiliary_loss,
                hidden_states,
                use_reentrant=False,
            )
        checkpointed_states = checkpoint(
            layer.forward,
            hidden_states,
            use_reentrant=False,
        )
        return checkpointed_states, None

    def _validate_input_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch, sequence_length); "
                f"got {tuple(input_ids.shape)}."
            )
        sequence_length_limit = self.config.tokens.sequence_length
        if input_ids.size(1) > sequence_length_limit:
            raise ValueError(
                f"Input sequence length {input_ids.size(1)} exceeds configured "
                f"sequence_length {sequence_length_limit}."
            )
        if input_ids.is_floating_point() or input_ids.is_complex():
            raise TypeError("input_ids must use an integer tensor dtype.")


def _position_embedding_type(config: PicotronConfig) -> str:
    return config.model.model_config.position_embedding_type


def _rope_theta(config: PicotronConfig) -> float:
    return float(config.model.model_config.rope_theta)


def _build_attention(
    config: PicotronConfig, *, use_rope: bool
) -> CausalSelfAttention | MultiHeadLatentAttention:
    """Construct the config-selected eager attention implementation."""

    model_config = config.model.model_config
    if model_config.attention_type == "mla":
        assert model_config.kv_lora_rank is not None
        return MultiHeadLatentAttention(
            model_config.hidden_size,
            model_config.num_attention_heads,
            model_config.kv_lora_rank,
            use_rope=use_rope,
            rope_theta=_rope_theta(config),
            use_triton_rope=config.model.triton_kernels.rope,
        )
    return CausalSelfAttention(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        sliding_window_size=model_config.sliding_window_size,
        use_rope=use_rope,
        rope_theta=_rope_theta(config),
        use_triton_rope=config.model.triton_kernels.rope,
        use_triton_attention=config.model.triton_kernels.attention,
    )
