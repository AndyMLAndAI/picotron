# Hugging Face causal-LM compatibility audit

Picotron's SFT and DPO loops require a decoder-only model whose forward call
accepts tensor `input_ids` (and optionally `attention_mask`) and returns
logits shaped `(batch, sequence, vocab_size)`. They do not inspect a model's
internal attention, normalization, RoPE, GQA, or sliding-window design.

Each `examples/verify_*.py` first downloads only its official
`AutoConfig`. Passing `--train-steps 2` deliberately opts into downloading
the real model weights and running two SFT updates on a tokenized synthetic
batch; `--dpo-steps 1` additionally runs one DPO update. This is intentionally
not the default because the smallest selected checkpoints are 0.5B--9B
parameters and DPO needs a frozen reference copy.

| Family and checkpoint | Published HF interface | Picotron result | Caveat |
| --- | --- | --- | --- |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | `qwen2`, `AutoModelForCausalLM` | Generic contract compatible; opt-in smoke script provided. | Qwen2.5 is 0.5B, so real CPU training is slow. It has GQA (14 Q / 2 KV heads), but this is internal to Transformers and needs no Picotron special case. |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `qwen3`, `AutoModelForCausalLM` | Generic contract compatible; opt-in smoke script provided. | Its public config uses 16 Q / 8 KV heads and RoPE; no interface deviation found. |
| [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | `qwen3_5`, `AutoModelForMultimodalLM` | Text-only trainer contract compatible; **not compatible with the current generic `load_model()` path.** | The full checkpoint is multimodal and must be loaded with `AutoModelForMultimodalLM`. Its text-only forward accepts `input_ids`/`attention_mask` and returns logits, so the verifier can exercise SFT/DPO without vision examples. Multimodal SFT/DPO still needs processor-aware dataset support. |
| [StarCoder2-3B](https://huggingface.co/bigcode/starcoder2-3b) | `starcoder2`, `AutoModelForCausalLM` | Generic contract compatible; opt-in smoke script provided. | GQA and 4,096-token sliding window are internal. This model is not in Picotron's conservative Unsloth allowlist, so it uses the Transformers fallback. |
| [Llama 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | `llama`, `AutoModelForCausalLM` | Generic contract compatible; opt-in smoke script provided. | The Hugging Face license must be accepted and a token supplied if required. |
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | `mistral`, `AutoModelForCausalLM` | Generic contract compatible; opt-in smoke script provided. | Mistral's smallest official base checkpoint is 7B; it uses GQA and sliding-window attention, and needs GPU verification due size. |
| [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b) | `gemma2`, `AutoModelForCausalLM` | Generic contract compatible; opt-in smoke script provided. | Google license access is required. Gemma 2's alternating window/full attention and soft-capping are internal; its forward interface remains causal-LM standard. |

## What was verified locally

The scripts, their model-type expectations, and the generic SFT/DPO tensor
call contract were statically checked. Real Hub config downloads, model-weight
loads, and optimizer updates were **not executed locally**: this machine has
no usable project Python environment or installed `transformers`/`torch`
runtime, and downloading 0.5B--9B checkpoints is outside the local CPU test
budget.

Run these on Kaggle or another networked machine with the required model
access instead:

```bash
python examples/verify_qwen2_5.py --train-steps 2 --dpo-steps 1 --device cuda
python examples/verify_qwen3.py --train-steps 2 --dpo-steps 1 --device cuda
python examples/verify_starcoder2.py --train-steps 2 --device cuda
python examples/verify_llama_3_2.py --token "$HF_TOKEN" --train-steps 2 --device cuda
python examples/verify_mistral.py --train-steps 2 --device cuda
python examples/verify_gemma_2.py --token "$HF_TOKEN" --train-steps 2 --device cuda
```

`examples/verify_qwen3_5.py` intentionally bypasses the identified
generic-loader limitation by using `AutoModelForMultimodalLM` directly. It
tests only Qwen3.5's text-only forward path; multimodal examples are out of
scope for Picotron's text SFT/DPO data layer.
