"""Optional Triton kernels with safe PyTorch fallbacks at call sites."""

from picotron.nn.triton_kernels.adamw import AdamWStepWithFallback, TritonAdamWUnavailable
from picotron.nn.triton_kernels.cross_entropy import (
    CrossEntropyWithFallback,
    TritonCrossEntropyUnavailable,
    triton_cross_entropy,
)
from picotron.nn.triton_kernels.rmsnorm import TritonRMSNormUnavailable, triton_rms_norm
from picotron.nn.triton_kernels.rope import TritonRoPEUnavailable, triton_apply_rotary_embedding
from picotron.nn.triton_kernels.swiglu import TritonSwiGLUUnavailable, triton_swiglu

__all__ = [
    "AdamWStepWithFallback",
    "CrossEntropyWithFallback",
    "TritonAdamWUnavailable",
    "TritonCrossEntropyUnavailable",
    "TritonRMSNormUnavailable",
    "TritonRoPEUnavailable",
    "TritonSwiGLUUnavailable",
    "triton_apply_rotary_embedding",
    "triton_cross_entropy",
    "triton_rms_norm",
    "triton_swiglu",
]
