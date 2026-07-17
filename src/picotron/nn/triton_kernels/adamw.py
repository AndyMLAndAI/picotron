"""Guarded extension point for a future fused Triton AdamW update kernel."""

from __future__ import annotations

import warnings

from torch.optim import Optimizer


class TritonAdamWUnavailable(RuntimeError):
    """Raised because fused AdamW is intentionally not enabled yet."""


def triton_adamw_step(optimizer: Optimizer) -> None:
    """Reserved fused AdamW implementation point.

    A fused AdamW update can silently alter optimizer math. It is deliberately
    not implemented until CUDA numerical parity is established against PyTorch.
    """

    del optimizer
    raise TritonAdamWUnavailable(
        "Fused Triton AdamW is not implemented; using PyTorch AdamW instead."
    )


class AdamWStepWithFallback:
    """Attempt the future fused update only when explicitly enabled, then fall back."""

    def __init__(self, *, use_triton: bool = False) -> None:
        self.use_triton = use_triton
        self._fallback_warned = False

    def step(self, optimizer: Optimizer) -> None:
        if self.use_triton:
            try:
                triton_adamw_step(optimizer)
                return
            except Exception as error:
                if not self._fallback_warned:
                    warnings.warn(
                        f"Triton AdamW unavailable; using PyTorch fallback: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._fallback_warned = True
        optimizer.step()
