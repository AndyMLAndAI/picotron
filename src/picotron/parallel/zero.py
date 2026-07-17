"""Minimal ZeRO Stage 1 and 2 optimizer-state/gradient sharding."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import AdamW


class ZeroOptimizer:
    """Shard AdamW ownership across ranks while keeping model parameters replicated."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        learning_rate: float,
        stage: int,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if stage not in (1, 2):
            raise ValueError("ZeRO stage must be 1 or 2.")
        self.parameters = list(parameters)
        self.stage = stage
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._owners = [index % self.world_size for index in range(len(self.parameters))]
        self._owned_parameters = [
            parameter
            for parameter, owner in zip(self.parameters, self._owners, strict=True)
            if owner == self.rank
        ]
        self._optimizer = AdamW(
            self._owned_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.param_groups = self._optimizer.param_groups

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear gradients for replicated parameters on every rank."""

        for parameter in self.parameters:
            parameter.grad = None

    def backward(self, loss: Tensor, model: nn.Module) -> None:
        """Backpropagate and shard gradient ownership for ZeRO Stage 2."""

        context = model.no_sync() if self.stage == 2 and self.world_size > 1 else nullcontext()
        with context:
            loss.backward()
        if self.stage == 2 and self.world_size > 1:
            self._reduce_gradients_to_owners()

    def step(self) -> None:
        """Update local parameter shards, then broadcast the replicated weights."""

        self._optimizer.step()
        if self.world_size > 1:
            for parameter, owner in zip(self.parameters, self._owners, strict=True):
                dist.broadcast(parameter.data, src=owner)

    def clip_grad_norm_(self, max_norm: float) -> Tensor:
        """Clip gradients with globally correct norm semantics for both stages."""

        if max_norm <= 0:
            raise ValueError("max_norm must be positive.")
        gradients = [parameter.grad for parameter in self.parameters if parameter.grad is not None]
        if not gradients:
            return torch.tensor(0.0)

        norm_device = gradients[0].device
        local_squared_norm = torch.zeros((), device=norm_device, dtype=torch.float32)
        for gradient in gradients:
            local_squared_norm.add_(gradient.detach().float().pow(2).sum())
        if self.stage == 2 and self.world_size > 1:
            dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)

        total_norm = local_squared_norm.sqrt()
        clip_coefficient = max_norm / (total_norm + 1e-6)
        if clip_coefficient < 1:
            for gradient in gradients:
                gradient.mul_(clip_coefficient.to(dtype=gradient.dtype))
        return total_norm

    def state_dict(self) -> dict[str, Any]:
        """Return this rank's optimizer-state shard for rank-local persistence."""

        return {
            "stage": self.stage,
            "rank": self.rank,
            "owned_parameter_indices": [
                index for index, owner in enumerate(self._owners) if owner == self.rank
            ],
            "optimizer_state_dict": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a rank-local optimizer-state shard."""

        if state_dict.get("stage") != self.stage:
            raise ValueError("Checkpoint ZeRO stage does not match the active optimizer.")
        if state_dict.get("rank") != self.rank:
            raise ValueError("ZeRO optimizer shards must be restored on their owning rank.")
        self._optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def _reduce_gradients_to_owners(self) -> None:
        for parameter, owner in zip(self.parameters, self._owners, strict=True):
            has_local_gradient = torch.tensor(
                int(parameter.grad is not None), device=parameter.device
            )
            dist.all_reduce(has_local_gradient, op=dist.ReduceOp.MAX)
            if not has_local_gradient.item():
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.reduce(parameter.grad, dst=owner, op=dist.ReduceOp.SUM)
            if self.rank == owner:
                parameter.grad.div_(self.world_size)
            else:
                parameter.grad = None
