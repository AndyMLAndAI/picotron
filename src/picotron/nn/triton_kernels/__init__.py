"""Optional Triton kernels with safe PyTorch fallbacks at call sites."""

from picotron.nn.triton_kernels.rmsnorm import TritonRMSNormUnavailable, triton_rms_norm

__all__ = ["TritonRMSNormUnavailable", "triton_rms_norm"]

