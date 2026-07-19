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

The RMSNorm, SwiGLU, RoPE, causal-attention, and cross-entropy kernels are
currently forward/no-grad inference paths. During autograd they deliberately
emit a one-time warning and use the PyTorch implementation instead. The AdamW
Triton path is presently a guarded stub and always uses `torch.optim.AdamW`.
Consequently, enabling these flags does not accelerate Picotron pretraining
today; it is safe, but it exercises fallback code rather than fused training.

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
