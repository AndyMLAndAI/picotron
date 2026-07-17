"""Dense feed-forward layers shared by decoder and MoE experts."""

from __future__ import annotations

import warnings

from torch import Tensor, nn
from torch.nn import functional as F

from picotron.nn.triton_kernels.swiglu import triton_swiglu


class SwiGLU(nn.Module):
    """Llama-style gated feed-forward network."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, *, use_triton_swiglu: bool = False
    ) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_projection = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.use_triton_swiglu = use_triton_swiglu
        self._triton_fallback_warned = False

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = self.gate_projection(hidden_states)
        up = self.up_projection(hidden_states)
        if self.use_triton_swiglu:
            try:
                activated = triton_swiglu(gate, up)
            except Exception as error:
                if not self._triton_fallback_warned:
                    warnings.warn(
                        f"Triton SwiGLU unavailable; using PyTorch fallback: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._triton_fallback_warned = True
                activated = F.silu(gate) * up
        else:
            activated = F.silu(gate) * up
        return self.down_projection(activated)
