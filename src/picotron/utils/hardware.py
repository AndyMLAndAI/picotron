"""Hardware capability detection for safe runtime feature selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec

import torch


class AttentionBackend(StrEnum):
    """Attention backend selected for the detected runtime."""

    FLASH_ATTENTION = "flash_attn"
    XFORMERS = "xformers"
    SDPA = "sdpa"
    EAGER = "eager"


@dataclass(frozen=True, slots=True)
class AttentionBackendReport:
    """Detected backend availability and the selected safe fallback."""

    selected: AttentionBackend
    flash_attention_available: bool
    xformers_available: bool
    sdpa_available: bool


@dataclass(frozen=True, slots=True)
class TritonReport:
    """Optional Triton-kernel status; disabled unless explicitly requested."""

    enabled: bool
    installed: bool
    hardware_compatible: bool
    available: bool


def get_gpu_compute_capability(device: int | torch.device | None = None) -> tuple[int, int] | None:
    """Return CUDA compute capability, or ``None`` when CUDA is unavailable."""

    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError):
        return None


def select_training_dtype(device: int | torch.device | None = None) -> torch.dtype:
    """Select fp32 for CPU, bf16 for Ampere+, and fp16 for older CUDA GPUs."""

    capability = get_gpu_compute_capability(device)
    if capability is None:
        return torch.float32
    return torch.bfloat16 if capability[0] >= 8 else torch.float16


def detect_attention_backend(
    device: int | torch.device | None = None,
) -> AttentionBackendReport:
    """Report the highest-priority supported attention backend without using it."""

    capability = get_gpu_compute_capability(device)
    cuda_available = capability is not None
    flash_attention_available = (
        cuda_available
        and capability[0] >= 8
        and _module_is_installed("flash_attn")
    )
    xformers_available = cuda_available and _module_is_installed("xformers")
    sdpa_available = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    if flash_attention_available:
        selected = AttentionBackend.FLASH_ATTENTION
    elif xformers_available:
        selected = AttentionBackend.XFORMERS
    elif sdpa_available:
        selected = AttentionBackend.SDPA
    else:
        selected = AttentionBackend.EAGER

    return AttentionBackendReport(
        selected=selected,
        flash_attention_available=flash_attention_available,
        xformers_available=xformers_available,
        sdpa_available=sdpa_available,
    )


def detect_triton_support(
    *,
    enabled: bool = False,
    device: int | torch.device | None = None,
) -> TritonReport:
    """Report Triton status without enabling it implicitly."""

    capability = get_gpu_compute_capability(device)
    installed = _module_is_installed("triton")
    hardware_compatible = capability is not None and capability[0] >= 7
    return TritonReport(
        enabled=enabled,
        installed=installed,
        hardware_compatible=hardware_compatible,
        available=enabled and installed and hardware_compatible,
    )


def _module_is_installed(module_name: str) -> bool:
    """Check optional package presence without importing CUDA extensions."""

    return find_spec(module_name) is not None
