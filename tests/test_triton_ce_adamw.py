"""CPU fallback checks for optional Triton cross-entropy and AdamW support."""

import pytest
import torch
from torch.nn import functional as F
from torch.optim import AdamW

from picotron.nn.triton_kernels.adamw import AdamWStepWithFallback, triton_adamw_step
from picotron.nn.triton_kernels.cross_entropy import (
    CrossEntropyWithFallback,
    triton_cross_entropy,
)


def test_triton_cross_entropy_imports_and_cpu_fallback_matches_pytorch() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]])
    targets = torch.tensor([0, 2], dtype=torch.long)
    cross_entropy = CrossEntropyWithFallback(use_triton=True)

    with pytest.warns(RuntimeWarning, match="Triton cross-entropy unavailable"):
        actual = cross_entropy(logits, targets)

    torch.testing.assert_close(actual, F.cross_entropy(logits, targets))
    assert callable(triton_cross_entropy)


def test_adamw_fallback_matches_known_single_step_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = AdamW(
        [parameter], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    parameter.grad = torch.tensor([1.0])
    step = AdamWStepWithFallback(use_triton=True)

    with pytest.warns(RuntimeWarning, match="Triton AdamW unavailable"):
        step.step(optimizer)

    torch.testing.assert_close(parameter.detach(), torch.tensor([0.899]))
    with pytest.raises(RuntimeError, match="not implemented"):
        triton_adamw_step(optimizer)
