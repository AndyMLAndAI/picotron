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

## Mixing preprocessed datasets

Preprocess each source separately with `tools/preprocess_data.py`; the tool
shows a `tqdm` tokenization bar and produces one reusable uint16 cache per
invocation. Mix those caches at training time rather than concatenating them:

```yaml
data:
  datasets:
    - path: /data/fineweb.uint16
      weight: 0.7
    - path: /data/code.uint16
      weight: 0.3
```

Each pretraining batch is sampled from one source according to these relative
weights. The legacy `data.dataset_token_path` remains supported as a single
source with weight `1.0`; do not set both forms in the same config. Opening
the configured token caches shows a `tqdm` startup bar, while the existing
training display reports optimizer-step progress.

To let `picotron --config` preprocess missing caches before training, use an
HF source instead of `path`; the cache filename is derived deterministically
from the source, tokenizer, and token target:

```yaml
data:
  tokenizer_name: gpt2
  datasets:
    - hf_name: HuggingFaceFW/fineweb-edu
      hf_config: CC-MAIN-2024-10
      target_tokens: 500000000
      weight: 1.0
```

## Hugging Face access

For gated Hugging Face models or datasets, set `data.hf_token` in the run
configuration. It takes priority over the standard `HF_TOKEN` environment
variable; leave it `null` for public access or to use the environment value.

## Native checkpoint architecture sidecar

Every native `PicotronDecoderModel` safetensors checkpoint also writes
`config.json` beside its weights. It records the complete validated model
architecture (including GQA, RoPE/NoPE, MoE, MLA, and Triton settings).
`load_native_model(checkpoint_path)` reconstructs the model from that sidecar;
SFT and DPO can use the same path by passing `model=None` with
`base_checkpoint_path`.

## CPU checks

From this directory, run:

```powershell
python -m pytest tests/test_model.py tests/test_hardware.py
```
