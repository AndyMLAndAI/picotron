# Picotron

Picotron is a from-scratch distributed LLM pretraining framework.

## Current capabilities

Picotron includes a native configurable decoder model, pretraining, checkpoint
resume, CPU/GPU DDP, ZeRO, data preprocessing, and thin SFT/DPO scripting
layers. GPU-specific combinations must still be verified on the target
hardware before being relied on for a production-sized run.

## Optional Triton kernels

Triton kernel flags live under `model.triton_kernels` and are disabled by
default. See [the Triton kernel status and configuration guide](docs/triton_kernels.md).
RMSNorm now has a fused Triton forward with a correct PyTorch backward;
the remaining Triton paths remain guarded inference-only experiments and
fall back during autograd.

## CPU checks

From this directory, run:

```powershell
python -m pytest tests/test_model.py tests/test_hardware.py
```
