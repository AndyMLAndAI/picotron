"""Top-k Mixture-of-Experts feed-forward layer."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from picotron.config.config import MoEConfig
from picotron.nn.feedforward import SwiGLU


class MoEFeedForward(nn.Module):
    """Route each token to weighted top-k SwiGLU experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: MoEConfig,
        *,
        use_triton_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(
                hidden_size,
                intermediate_size,
                use_triton_swiglu=use_triton_swiglu,
            )
            for _ in range(config.num_experts)
        )
        self.last_routing_indices: Tensor | None = None

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Return expert-combined states and differentiable load-balancing loss."""

        input_shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, input_shape[-1])
        router_logits = self.router(tokens)
        topk_logits, topk_indices = torch.topk(router_logits, self.config.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        combined = torch.zeros_like(tokens)

        for expert_index, expert in enumerate(self.experts):
            token_indices, topk_slots = torch.where(topk_indices == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(tokens.index_select(0, token_indices))
            weighted_output = expert_output * topk_weights[token_indices, topk_slots].unsqueeze(-1)
            combined = combined.index_add(0, token_indices, weighted_output)

        self.last_routing_indices = topk_indices.detach()
        return combined.reshape(input_shape), self._auxiliary_loss(router_logits, topk_indices)

    def _auxiliary_loss(self, router_logits: Tensor, topk_indices: Tensor) -> Tensor:
        router_probabilities = F.softmax(router_logits, dim=-1)
        expert_importance = router_probabilities.mean(dim=0)
        top_one_load = F.one_hot(topk_indices[:, 0], self.config.num_experts).to(
            router_probabilities.dtype
        ).mean(dim=0)
        balancing_loss = self.config.num_experts * torch.sum(expert_importance * top_one_load)
        return balancing_loss * self.config.aux_loss_coefficient
