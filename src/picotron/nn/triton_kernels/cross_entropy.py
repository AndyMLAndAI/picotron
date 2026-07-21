"""Optional fused Triton cross-entropy inference-forward kernel."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - expected on CPU-only environments.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


class TritonCrossEntropyUnavailable(RuntimeError):
    """Raised when the optional Triton cross-entropy path cannot safely run."""


if _TRITON_AVAILABLE:

    @triton.jit
    def _cross_entropy_kernel(
        logits_ptr,
        targets_ptr,
        losses_ptr,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=-float("inf"))
        logits_fp32 = logits.to(tl.float32)
        maximum = tl.max(logits_fp32, axis=0)
        log_sum_exp = maximum + tl.log(tl.sum(tl.exp(logits_fp32 - maximum), axis=0))
        target = tl.load(targets_ptr + row)
        target_logit = tl.load(logits_ptr + row * vocab_size + target).to(tl.float32)
        tl.store(losses_ptr + row, log_sum_exp - target_logit)


def triton_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy or raise ``TritonCrossEntropyUnavailable`` safely.

    This is a forward-only kernel. Gradient-enabled calls deliberately use the
    PyTorch fallback so that its tested autograd implementation is retained.
    """

    if not _TRITON_AVAILABLE:
        raise TritonCrossEntropyUnavailable("Triton is not installed.")
    if torch.is_grad_enabled() and logits.requires_grad:
        raise TritonCrossEntropyUnavailable(
            "Triton cross-entropy currently supports no-grad inference only."
        )
    if not logits.is_cuda or not targets.is_cuda:
        raise TritonCrossEntropyUnavailable("Triton cross-entropy requires CUDA tensors.")
    if torch.cuda.get_device_capability(logits.device)[0] < 7:
        raise TritonCrossEntropyUnavailable("Triton cross-entropy requires compute capability 7.0+.")
    if (
        logits.ndim != 2
        or targets.ndim != 1
        or logits.size(0) != targets.numel()
        or targets.dtype != torch.long
        or not logits.is_contiguous()
        or not targets.is_contiguous()
    ):
        raise TritonCrossEntropyUnavailable(
            "Triton cross-entropy requires contiguous logits (rows, vocab) and long targets."
        )

    vocab_size = logits.size(1)
    if bool(torch.any((targets < 0) | (targets >= vocab_size)).item()):
        raise TritonCrossEntropyUnavailable("Triton cross-entropy targets are out of range.")
    block_size = triton.next_power_of_2(vocab_size)
    if block_size > 65536:
        raise TritonCrossEntropyUnavailable("Triton cross-entropy supports vocab sizes up to 65536.")
    losses = torch.empty(logits.size(0), device=logits.device, dtype=logits.dtype)
    _cross_entropy_kernel[(logits.size(0),)](
        logits, targets, losses, vocab_size, BLOCK_SIZE=block_size, num_warps=4
    )
    return losses.mean()


class CrossEntropyWithFallback:
    """Run optional Triton inference CE and safely retain PyTorch autograd."""

    def __init__(self, *, use_triton: bool = False) -> None:
        self.use_triton = use_triton
        self._fallback_warned = False

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.use_triton:
            try:
                return triton_cross_entropy(logits, targets)
            except Exception as error:
                if not self._fallback_warned:
                    warnings.warn(
                        f"Triton cross-entropy unavailable; using PyTorch fallback: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._fallback_warned = True
        return F.cross_entropy(logits, targets)
