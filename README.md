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
The currently shipped Triton paths are guarded inference-only experiments:
they intentionally fall back to PyTorch during autograd, so they do not yet
speed up training.

## CPU checks

From this directory, run:

```powershell
python -m pytest tests/test_model.py tests/test_hardware.py
```
