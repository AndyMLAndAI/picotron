# Optional Triton kernels

Picotron's Triton paths are opt-in. Enable individual paths under
`model.triton_kernels`; every flag is `false` by default:

```yaml
model:
  triton_kernels:
    rmsnorm: true
    swiglu: true
    rope: true
    attention: true
    cross_entropy: true
    adamw: true
```

`detect_triton_support(enabled=True, device=0)` reports the environment-level
status. A T4 (compute capability 7.5) is hardware-compatible: the current
minimum is compute capability 7.0. `available` also requires the `triton`
package to be installed. This report does **not** prove a particular kernel
compiled or can be used for training.

## Current training status

RMSNorm uses the fused Triton forward during autograd and a mathematically
equivalent PyTorch backward. It is therefore the only shipped Triton option
that can accelerate training today, although its real T4 benefit must be
measured. SwiGLU, RoPE, causal-attention, and cross-entropy remain
forward/no-grad inference paths and deliberately fall back during autograd.
The AdamW Triton path is a guarded stub and always uses `torch.optim.AdamW`.

The dense-SwiGLU flag applies to dense decoder blocks. MoE expert SwiGLU
blocks currently use their PyTorch implementation.

Every kernel is runtime-guarded: an unavailable package, incompatible device,
compile failure, or execution failure falls back to PyTorch rather than
terminating a run. Validate any future fused-training implementation against
the eager path on the target GPU before relying on it.

## torch.compile

`torch.compile` is not currently wired into Picotron and has no config flag.
It should be introduced only with a dedicated GPU correctness/performance
validation pass, including DDP and the optional native-model feature
combinations.
