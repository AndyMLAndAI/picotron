"""Optional Triton kernels with safe PyTorch fallbacks at call sites."""

from picotron.nn.triton_kernels.adamw import AdamWStepWithFallback, TritonAdamWUnavailable
from picotron.nn.triton_kernels.attention import (
    TritonAttentionUnavailable,
    triton_causal_attention,
)
from picotron.nn.triton_kernels.cross_entropy import (
    CrossEntropyWithFallback,
    TritonCrossEntropyUnavailable,
    triton_cross_entropy,
)
from picotron.nn.triton_kernels.rmsnorm import (
    TritonRMSNormUnavailable,
    rms_norm_with_pytorch_forward,
    triton_rms_norm,
)
from picotron.nn.triton_kernels.rope import TritonRoPEUnavailable, triton_apply_rotary_embedding
from picotron.nn.triton_kernels.swiglu import (
    TritonSwiGLUUnavailable,
    swiglu_with_pytorch_forward,
    triton_swiglu,
)

__all__ = [
    "AdamWStepWithFallback",
    "CrossEntropyWithFallback",
    "TritonAttentionUnavailable",
    "TritonAdamWUnavailable",
    "TritonCrossEntropyUnavailable",
    "TritonRMSNormUnavailable",
    "TritonRoPEUnavailable",
    "TritonSwiGLUUnavailable",
    "triton_causal_attention",
    "triton_apply_rotary_embedding",
    "triton_cross_entropy",
    "triton_rms_norm",
    "triton_swiglu",
    "swiglu_with_pytorch_forward",
    "rms_norm_with_pytorch_forward",
]
