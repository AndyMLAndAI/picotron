"""Minimal ZeRO Stage 1 and 2 optimizer-state/gradient sharding."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import AdamW


class DistributedGradScaler:
    """Loss scaler whose overflow decision is shared by every ZeRO rank.

    ``torch.amp.GradScaler`` owns the scale tensor while ZeRO owns gradient
    unscaling and finite checks. This avoids relying on private GradScaler
    optimizer-state internals for a sharded optimizer.
    """

    def __init__(
        self,
        device_type: str,
        *,
        init_scale: float = 65_536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2_000,
    ) -> None:
        self._scaler = torch.amp.GradScaler(
            device_type,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._successful_steps = 0

    def scale(self, loss: Tensor) -> Tensor:
        """Scale one loss before its ZeRO-aware backward pass."""

        return self._scaler.scale(loss)

    def step(
        self,
        optimizer: "ZeroOptimizer",
        model: nn.Module,
        *,
        max_grad_norm: float | None = None,
    ) -> bool:
        """Unscale globally, then update all ranks together when finite.

        Returns ``True`` only when an optimizer update was performed.
        """

        del model  # ZeRO already receives the wrapped model during backward.
        found_inf = optimizer.unscale_and_check_(self._scaler.get_scale())
        if found_inf:
            self._successful_steps = 0
            self._set_scale(self._scaler.get_scale() * self._backoff_factor)
            return False
        if max_grad_norm is not None:
            optimizer.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        self._successful_steps += 1
        if self._successful_steps >= self._growth_interval:
            self._successful_steps = 0
            self._set_scale(self._scaler.get_scale() * self._growth_factor)
        else:
            self._set_scale(self._scaler.get_scale())
        return True

    def _set_scale(self, value: float) -> None:
        self._scaler.update(new_scale=max(value, 1.0))


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

    def unscale_and_check_(self, scale: float) -> bool:
        """Return a globally synchronized overflow result and unscale gradients.

        Stage 1 has replicated reduced gradients. Stage 2 retains only each
        owner's reduced gradient. In both cases, the all-reduced flag covers
        the complete logical gradient set before any rank decides to update.
        """

        if scale <= 0:
            raise ValueError("loss scale must be positive.")
        device = self.parameters[0].device if self.parameters else torch.device("cpu")
        found_inf = torch.zeros((), dtype=torch.int32, device=device)
        for parameter in self.parameters:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                found_inf.fill_(1)
                break
        if self.world_size > 1:
            dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)
        if found_inf.item():
            return True
        inverse_scale = 1.0 / scale
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(inverse_scale)
        return False

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
