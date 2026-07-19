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

`torch.compile` is separately opt-in, not a Triton-kernel flag:

```yaml
model:
  compile_model: true
```

Picotron compiles the native model before DDP wraps it. If the compile call
fails, it emits a warning and continues with the eager model. Checkpointing
unwraps both DDP and the compiled wrapper, so `.safetensors` weights remain
portable to an eager model. Actual T4 speed and compatibility still require
a target-GPU test.

## Attention backends

`CausalSelfAttention` now consumes `detect_attention_backend()` at runtime.
When xFormers is selected, it calls `xformers.ops.memory_efficient_attention`
with causal masking; a failure falls through to PyTorch SDPA and then the
manual eager path. GQA repeats KV heads before the xFormers call, preserving
the native eager semantics and xFormers autograd support. FlashAttention is
still detected but is not directly wired; its selected report currently falls
through to SDPA.
