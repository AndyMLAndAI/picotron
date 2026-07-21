<p align="center">
  <img src="src/banner.png" alt="Picotron" width="100%">
</p>

<h1 align="center">Picotron</h1>

<p align="center">
  A correctness-first, from-scratch PyTorch framework for decoder-only LLM pretraining.<br>
  Native pretraining · token-cache data pipeline · safetensors checkpoints · T4-aware acceleration
</p>

<p align="center">
  <a href="https://github.com/AndyMLAndAI/picotron/releases/tag/v0.9.8">v0.9.8 Beta</a>
  · <a href="https://andymlandai.github.io/picotron/">Interactive project site</a>
  · <a href="https://youtu.be/ciUhfGlGgaw">Build Week demo video</a>
</p>

> [!WARNING]
> Picotron is hackathon software, not a production training platform. CPU tests cover the core logic, but every CUDA-specific configuration—especially Triton, xFormers, fp16, DDP, and ZeRO—must be validated on the target GPU before a long run. The project deliberately prefers a guarded PyTorch fallback over a fast path that might silently be wrong.

## v0.9.8 Beta: what is real today

This is Picotron's first public release. The primary, demo-ready workflow is **native decoder pretraining**: strict configuration, Hugging Face token-cache preprocessing, fp16 T4-aware execution, guarded Triton RMSNorm/SwiGLU, safetensors checkpoints, and native inference.

Recent additions in this release include:

- A one-command `picotron --config config.yaml` workflow that can preprocess missing Hugging Face token caches before training.
- A self-documenting safetensors checkpoint format with a native `config.json` sidecar.
- Rich/tqdm progress output that stays readable in Kaggle notebooks, plus CSV and plain-text run logs.
- A hardware banner and an explicit Triton forward/backward probe, so a requested kernel cannot silently fall back during the demo run.
- An interactive GitHub Pages configuration lab and Kaggle-ready native-pretraining notebook.

### Primary path vs. experimental work

| Status | Included capability |
| --- | --- |
| **Primary release path** | Native Picotron pretraining, token caches, safetensors checkpoints/resume, configuration validation, logging, and short native inference. |
| **Hardware-validated demo path** | fp16 Turing/T4 execution and the Triton RMSNorm/SwiGLU probe. Exact throughput, memory use, and kernel numerics must still be checked for each target GPU and configuration. |
| **Experimental** | DDP/ZeRO scaling, xFormers/`torch.compile` tuning, MoE/MLA combinations, arbitrary-HF-model SFT, and DPO. These interfaces are present for exploration, but they are not the basis of the submission demo or a production compatibility guarantee. |

The word **Beta** is intentional: Picotron exposes its implementation and limitations rather than hiding them behind a claim of universal support.

## Why Picotron?

Picotron is a compact implementation of the parts of an LLM training stack that are most useful to inspect, modify, and verify:

- A native, configurable decoder model instead of a black-box training wrapper.
- Explicit Turing/T4-aware precision selection: CPU uses fp32; pre-Ampere CUDA uses fp16; bf16 is selected only on Ampere or later.
- Reproducible token caches, deterministic weighted multi-corpus mixing, and DDP rank sharding.
- Safetensors model weights, optimizer sidecars, and a native-model `config.json` architecture record.
- Small, script-first SFT and DPO packages that can use either Picotron's native model or compatible Hugging Face causal LMs.

The goal is not to claim every modern optimization. The goal is to make each implemented path visible, testable, and safe to fall back from.

## Contents

- [Installation](#installation)
- [v0.9.8 Beta: what is real today](#v098-beta-what-is-real-today)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Data pipelines](#data-pipelines)
- [Native decoder model](#native-decoder-model)
- [Training and distributed execution](#training-and-distributed-execution)
- [Checkpoints and resuming](#checkpoints-and-resuming)
- [Fine-tuning and preference optimization](#fine-tuning-and-preference-optimization)
- [Hardware and optional acceleration](#hardware-and-optional-acceleration)
- [Observability](#observability)
- [Built with Codex](#built-with-codex)
- [Verification](#verification)
- [Current limitations](#current-limitations)

## Installation

Picotron targets Python 3.10+ and uses PyTorch. Install a PyTorch build appropriate for your machine first, then install the project in editable mode:

```bash
git clone https://github.com/AndyMLAndAI/picotron.git
cd picotron
pip install -r requirements.txt
pip install -e .
```

This exposes the following commands:

```text
picotron       # native pretraining
picotron-sft   # factory-based SFT CLI
picotron-dpo   # factory-based DPO CLI
```

For a local CPU sanity run, keep the model and sequence dimensions tiny. CUDA training is not required for the unit tests.

## Quick start

### 1. Create a configuration

Save the following as `config.yaml`. This example uses synthetic data, so it starts immediately and is suitable for a CPU smoke check.

```yaml
checkpoints:
  checkpoint_interval: 100
  checkpoints_path: checkpoints/quickstart.safetensors
  save_final_state: true

model:
  dtype: auto
  compile_model: false
  triton_kernels:
    rmsnorm: false
    swiglu: false
  model_config:
    vocab_size: 256
    hidden_size: 128
    intermediate_size: 256
    num_hidden_layers: 2
    num_attention_heads: 4
    attention_type: mha
    rope_theta: 10000.0
    nope_layers: []
    tie_word_embeddings: false

optimizer:
  learning_rate_scheduler:
    learning_rate: 0.001
  optimizer_factory:
    name: adamw
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-08
  weight_decay: 0.0
  clip_grad: null

parallelism:
  dp: 1
  zero_stage: 0

tokens:
  sequence_length: 64
  micro_batch_size: 2
  train_steps: 100

data:
  vocab_size: 256
  num_workers: 0
  prefetch_factor: 2

logging:
  iteration_step_info_interval: 10
  file_logging: true
  file_logging_output_dir: logs

general:
  project: picotron
  run: quickstart
  seed: 1337
```

### 2. Train

```bash
picotron --config config.yaml --num-sequences 1024
```

Without a configured token cache, the CLI uses synthetic fixed-length tokens. This is intentional: it is a quick way to verify model, optimizer, checkpoint, and display wiring before a data run.

### 3. Resume

Set `checkpoints.resume_checkpoint_path` in YAML, or pass it directly:

```bash
picotron --config config.yaml --resume-from checkpoints/quickstart.safetensors
```

## Configuration

`PicotronConfig` is strict and nested. Unknown keys, missing required architecture fields, incompatible attention choices, and mismatched `data.vocab_size` fail at load time rather than silently selecting a default.

| Section | Purpose |
| --- | --- |
| `checkpoints` | Save cadence, output location, and resume path. |
| `model` | Compute dtype, optional compilation/Triton flags, and native architecture. |
| `optimizer` | AdamW and learning-rate schedule settings. |
| `parallelism` | Data-parallel world size and ZeRO stage. |
| `tokens` | Sequence length, micro-batch size, and training step count. |
| `data` | Token caches, HF preprocessing sources, tokenizer metadata, and loader settings. |
| `logging` | Console/file logging cadence and output location. |
| `general` | Project name, run name, and seed. |

The compact reference schema is at [`src/picotron/config/picotron_decoder.yaml`](src/picotron/config/picotron_decoder.yaml).

### Attention selection

`model.model_config.attention_type` is authoritative:

| Type | Required settings | Notes |
| --- | --- | --- |
| `mha` | `num_key_value_heads` omitted or equal to query-head count | Standard multi-head attention. |
| `gqa` | `num_key_value_heads < num_attention_heads` | Grouped-query attention. |
| `mla` | Positive `kv_lora_rank` | Native low-rank KV compression path; it cannot be combined with GQA or sliding-window attention in the current implementation. |

Example GQA model block:

```yaml
model:
  model_config:
    vocab_size: 50257
    hidden_size: 512
    intermediate_size: 2048
    num_hidden_layers: 8
    num_attention_heads: 8
    attention_type: gqa
    num_key_value_heads: 2
    rope_theta: 1000000.0
    nope_layers: [2, 3, 4, 5]
    sliding_window_size: 256
```

## Data pipelines

Picotron trains on flat uint16 token caches. `MemmapTokenDataset` opens those files as memory maps and slices fixed-length token sequences, avoiding per-step tokenization.

### Preprocess explicitly

Use the reusable preprocessing tool when you want full control over the cache creation step:

```bash
python tools/preprocess_data.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config CC-MAIN-2024-10 \
  --tokenizer-name gpt2 \
  --target-tokens 100000000 \
  --output-path data/fineweb_100m_gpt2.uint16
```

The tool streams the Hub dataset, tokenizes in ordered worker processes, writes uint16 tokens, supports resumable progress, and shows a `tqdm` progress bar. Raw streamed text shards are cached so a later run with a different token target can reuse them.

### Preprocess automatically from `picotron --config`

Alternatively, let the pretraining command create any missing token caches before it starts training:

```yaml
data:
  tokenizer_name: gpt2
  token_cache_dir: data/token_cache
  datasets:
    - hf_name: HuggingFaceFW/fineweb-edu
      hf_config: CC-MAIN-2024-10
      target_tokens: 500000000
      weight: 0.7
    - hf_name: some-org/another-corpus
      target_tokens: 200000000
      weight: 0.3
```

Then run:

```bash
picotron --config config.yaml
```

For each `hf_name` source, Picotron derives a stable cache filename from the dataset name/config, tokenizer, target-token count, and text field. A complete cache is reused; an absent or incomplete cache invokes the existing [`tools/preprocess_data.py`](tools/preprocess_data.py) script before model construction. In a DDP launch, rank 0 preprocesses and the remaining ranks wait for the resulting caches.

### Use existing caches

Path-based configs remain supported unchanged:

```yaml
data:
  datasets:
    - path: /data/fineweb.uint16
      weight: 0.7
    - path: /data/code.uint16
      weight: 0.3
```

At each batch, weighted interleaving samples one source according to the relative weights. This is not concatenation: a corpus is mixed throughout training rather than being exhausted before another corpus begins. A legacy singleton remains available:

```yaml
data:
  dataset_token_path: /data/fineweb.uint16
```

Do not set `dataset_token_path` and `datasets` together.

### Hugging Face authentication

For gated models or datasets, use either configuration or the standard environment variable:

```yaml
data:
  hf_token: hf_your_token_here
```

Resolution order is:

1. `data.hf_token`
2. `HF_TOKEN` environment variable
3. no token (public Hub access)

The CLI forwards the resolved token to the preprocessing subprocess through `HF_TOKEN`. Avoid committing real tokens in a training configuration.

### Loader behavior

The cache loaders use configurable workers (`data.num_workers`, default `4`), pinned memory, persistent worker processes when workers are nonzero, and `data.prefetch_factor` (default `2`). DDP uses a rank-specific partition for normal memmap datasets; weighted interleaving also has rank-aware sampling.

## Native decoder model

The native `PicotronDecoderModel` is a decoder-only Transformer with config-selected components:

- RMSNorm pre-normalization
- SwiGLU dense FFN, or optional top-k MoE FFN on every decoder block
- RoPE by default, optional learned positional embeddings, and a per-layer NoPE list
- MHA, GQA, or the current native MLA variant
- Optional causal sliding-window attention for MHA/GQA
- Optional tied input/output embeddings
- Optional activation gradient checkpointing

MoE is enabled by adding a nested config:

```yaml
model:
  model_config:
    moe_config:
      num_experts: 4
      top_k: 2
      aux_loss_coefficient: 0.01
```

The model's forward API is intentionally simple:

```python
logits = model(input_ids)  # (batch, sequence_length, vocab_size)
```

Loss computation belongs in the training adapter/loop, not the model.

## Training and distributed execution

### Single process

```bash
picotron --config config.yaml
```

### DDP

Use PyTorch's launcher for one process per GPU:

```bash
torchrun --standalone --nproc_per_node=2 picotron --config config.yaml
```

Set the expected data-parallel world size in the same config:

```yaml
parallelism:
  dp: 2
  zero_stage: 0
```

Picotron chooses NCCL on CUDA and Gloo for CPU-only distributed tests. The primary rank owns the rich/tqdm display and file output. Every rank trains on a distinct data partition.

### ZeRO

The implemented `zero_stage` choices are:

| Stage | Behavior |
| --- | --- |
| `0` | Standard DDP / normal AdamW. |
| `1` | Shards optimizer-state ownership across ranks. |
| `2` | Adds gradient sharding. |

```yaml
parallelism:
  dp: 2
  zero_stage: 1
```

ZeRO checkpointing writes replicated model weights once plus rank-specific optimizer sidecars. The CPU distributed tests check mathematical agreement with plain DDP; that does not measure actual CUDA memory savings or NCCL behavior.

### Mixed precision

`model.dtype: auto` is hardware-aware:

| Runtime | Selected dtype |
| --- | --- |
| CPU / no CUDA | fp32 |
| Pre-Ampere CUDA, including Turing T4 (sm_75) | fp16 |
| Ampere or newer | bf16 |

You may force `float32`, `float16`, or `bfloat16`; attempting bf16 on a pre-Ampere GPU is rejected by config validation. CUDA fp16 uses a gradient-scaling path. Test the exact precision/ZeRO combination on the intended GPU before scaling up.

## Checkpoints and resuming

Native checkpoints are self-describing. Saving a native model to:

```text
checkpoints/run.safetensors
```

creates these companion files:

```text
checkpoints/run.safetensors       # model weights
checkpoints/run.optimizer.pt      # optimizer state and step
checkpoints/config.json           # full validated native Picotron config
```

With distributed ZeRO, optimizer sidecars are rank-specific (`run.optimizer.rank0.pt`, and so on). Model state remains a regular safetensors file.

`config.json` records the entire validated Picotron configuration, including all native architecture fields. Load a native model without reconstructing its architecture by hand:

```python
from picotron.serialize import load_native_model

model = load_native_model("checkpoints/run.safetensors", device="cuda:0")
```

The normal checkpoint API restores weights and optimizer state into provided objects:

```python
from picotron.serialize import load_checkpoint, save_checkpoint

save_checkpoint(model, optimizer, step, "checkpoints/run.safetensors")
step = load_checkpoint(model, optimizer, "checkpoints/run.safetensors")
```

## Fine-tuning and preference optimization

`picotron_sft` and `picotron_dpo` are thin training layers; they do not duplicate the native model or checkpoint code.

### SFT

The direct API accepts a compatible causal LM and a dataset returning either:

- mappings with `input_ids` and `labels`, or
- `(input_ids, labels)` tuples.

```python
from picotron_sft import run_sft

losses = run_sft(
    model=my_causal_lm,
    dataset=my_dataset,
    learning_rate=2e-5,
    batch_size=2,
    num_steps=100,
    device="cuda:0",
)
```

For a native Picotron checkpoint, omit the model and let the architecture sidecar reconstruct it:

```python
losses = run_sft(
    model=None,
    dataset=my_native_sft_dataset,
    base_checkpoint_path="checkpoints/run.safetensors",
    learning_rate=2e-5,
    batch_size=2,
    num_steps=100,
    device="cuda:0",
)
```

`picotron_sft.load_model()` attempts an Unsloth-compatible loading path when available, then falls back to Hugging Face `AutoModelForCausalLM`. It accepts the same `token=` argument used by Hugging Face and respects `HF_TOKEN` when no explicit token is supplied. Quantized/Unsloth paths require target-GPU verification.

### DPO

`run_dpo()` takes `(prompt, chosen, rejected)` preference triples plus a tokenizer. It keeps a frozen reference copy and uses standard DPO log-probability comparisons:

```python
from picotron_dpo import run_dpo

losses = run_dpo(
    model=my_causal_lm,
    dataset=preference_triples,
    tokenizer=tokenizer,
    beta=0.1,
    learning_rate=1e-5,
    batch_size=1,
    num_steps=100,
    device="cuda:0",
)
```

Native checkpoint reconstruction works the same way as SFT by using `model=None` and `base_checkpoint_path`. DPO is a separate package from pretraining; validate a real model/dataset pairing before using it for a substantial run.

## Hardware and optional acceleration

### Attention backend selection

Picotron detects a safe backend in this order:

```text
flash-attn (Ampere+) → xFormers → PyTorch SDPA → eager implementation
```

xFormers is used when selected and its call is runtime-guarded; failures fall back through SDPA/eager. FlashAttention may be detected, but the native direct FlashAttention call is not wired yet, so its selection currently falls through to SDPA. Backend detection is not a performance guarantee.

### `torch.compile`

Compilation is opt-in:

```yaml
model:
  compile_model: true
```

Picotron compiles before DDP wrapping. If compilation fails, it logs a warning and keeps the eager model. Compiled-model checkpoint portability is CPU-tested; target-GPU speedups are not assumed.

### Triton kernels

All Triton flags are off by default:

```yaml
model:
  triton_kernels:
    rmsnorm: true
    swiglu: true
    rope: false
    attention: false
    cross_entropy: false
    adamw: false
```

| Kernel | Current training status |
| --- | --- |
| RMSNorm | Fused Triton forward with a tested PyTorch autograd backward. |
| SwiGLU activation | Fused Triton forward with a tested PyTorch autograd backward. |
| RoPE | Guarded fallback path during training. |
| Attention | Guarded fallback path during training. |
| Cross entropy | Guarded fallback path during training. |
| AdamW | Uses PyTorch AdamW; Triton is not a training implementation. |

Triton requires an installed package and a CUDA device with compute capability 7.0+. A T4 is hardware-compatible, but actual compile/execution/numerical validation remains a target-GPU responsibility. The startup banner reports both requested kernels and this maturity status.

More detail: [`docs/triton_kernels.md`](docs/triton_kernels.md).

## Observability

At startup, Picotron prints an ASCII banner and run summary: hardware, selected dtype, attention backend, DDP world size, requested Triton kernels, their maturity status, and parameter count.

- Interactive terminals use a Rich live progress bar and metrics table.
- Non-TTY output uses a tqdm fallback.
- File logging is enabled by default and writes `metrics.csv` and `run.log` under `logs/<run-name>/`.

The CSV includes core metrics (`step`, `loss`, learning rate, tokens/sec, elapsed time) and can accept method-specific trainer metrics. The log records startup configuration and warnings so runs are inspectable after a Kaggle session ends.

## Built with Codex

Picotron was built for OpenAI Build Week with **Codex and GPT-5.6 as an active engineering collaborator**, not as a one-shot code generator.

Codex helped break the rebuild into small, testable milestones; structure the Python packages; write and review unit tests; refactor configuration and checkpoint handling; build the project documentation and interactive site; and turn real Kaggle failures into targeted fixes. A concrete example is the Triton work: target-GPU logs exposed runtime dtype and Turing-lowering failures that CPU checks could not reveal. Codex helped trace those failures, preserve guarded fallbacks, add an explicit pretraining probe, and produce a reproducible Kaggle workflow.

The project author drove the product direction, chose the scope, operated the real Kaggle GPU runs, supplied the hardware results, and made the final decisions about what is represented as primary versus experimental. This human-in-the-loop process is central to Picotron's correctness-first approach: generated code is useful, but real execution and explicit verification decide what ships.

## Verification

Run the CPU suite from the repository root:

```bash
python -m pytest tests -q
```

Useful focused checks:

```bash
# Model, attention, configuration, data, checkpoint and trainer checks
python -m pytest tests/test_model.py tests/test_attention.py tests/test_config.py tests/test_data.py tests/test_checkpoint.py -q

# CPU distributed correctness proxies
python -m pytest tests/test_ddp.py tests/test_zero.py tests/test_zero_fp16_checkpoint.py -q

# Optional-kernel fallback and gradient-math checks
python -m pytest tests/test_triton_rmsnorm_backward.py tests/test_triton_swiglu_backward.py -q
```

The repository also includes Kaggle notebooks for target-GPU smoke and demo workflows in [`tests/`](tests/). They are not treated as verified merely because they exist: run them on the intended accelerator and inspect their explicit pass/fail cells.

## Current limitations

This list is intentional. A missing feature should be obvious rather than implied by a config flag.

- Native autoregressive generation has no KV cache. Short full-forward decoding is possible in notebooks, but it is not an efficient serving interface.
- Native model loading from checkpoint `config.json` applies to Picotron's own `PicotronDecoderModel`, not arbitrary Hugging Face checkpoint formats.
- Tensor parallelism, pipeline parallelism, FSDP, CUDA graphs, LoRA/QLoRA, and RL training are not part of the shipped scope.
- The native FlashAttention path is not wired. Triton RoPE, attention, cross-entropy, and AdamW are not fused training implementations.
- CPU distributed tests are correctness proxies; they do not validate NCCL behavior, T4 throughput, memory reduction, or fused-kernel numerical output on CUDA.
- Automatic preprocessing is designed for raw text fields (`text` by default). Use `text_field` per source when a dataset exposes a different text column.
- `config.json` is a run-directory sidecar. Store different native architectures in different checkpoint directories so each run retains its own architecture record.

## Repository layout

```text
src/picotron/          Core native model, configuration, data, DDP/ZeRO, logging, checkpoints
src/picotron_sft/      Script-first SFT adapter and optional model loader
src/picotron_dpo/      Script-first DPO adapter
tools/                 Reusable preprocessing utility
scripts/               Compatibility script entrypoints
tests/                 CPU unit tests and Kaggle notebooks
docs/                  Focused design/status notes
examples/              Minimal usage and model-family examples
```

## Contributing and verification policy

Picotron values observable correctness over silent optimization. When adding a feature:

1. Keep the eager/PyTorch reference path available.
2. Add a CPU-testable shape, numerical, state-restoration, or synchronization test where possible.
3. Guard optional CUDA paths and make their fallback visible.
4. Verify new GPU behavior on the actual target hardware before claiming performance or memory wins.

## License

See the repository license information before redistributing or using Picotron in another project.
