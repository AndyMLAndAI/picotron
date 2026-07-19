"""Picotron SFT wiring sketch for a Llama 3-style causal LM."""

from __future__ import annotations

from typing import Any

from picotron.config.config import PicotronConfig
from picotron_sft import run_sft
from picotron_sft.sft_trainer import _extract_logits

# Dataset records: see docs/dataset_format.md. Model-specific: Llama 3 8B uses RMSNorm, SwiGLU, RoPE, and GQA. Picotron's
# toy model shares the RMSNorm/SwiGLU decoder style, but real Llama weights and
# tokenizer/chat-template loading belong in an external HF model factory.
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MODEL_SPECIFIC_KWARGS = {
    "architecture": "RMSNorm + SwiGLU + RoPE + GQA",
}


def make_picotron_config() -> PicotronConfig:
    """Return representative Llama 3 8B dimensions."""

    return PicotronConfig(
        checkpoints={"checkpoint_interval": 100},
        model={
            "model_config": {
                "vocab_size": 128256,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "attention_type": "gqa",
                "rope_theta": 500_000,
                "model_kwargs": MODEL_SPECIFIC_KWARGS,
            },
        },
        optimizer={"learning_rate_scheduler": {"learning_rate": 1e-5}},
        parallelism={"dp": 1},
        tokens={"sequence_length": 8192, "micro_batch_size": 1, "train_steps": 100},
        data={"vocab_size": 128256},
        logging={},
        general={"run": "llama_style_sft_example"},
    )


def logits_from_model_output(model_output: Any):
    """Adapt a standard HF causal-LM output without model-family branching."""

    return _extract_logits(model_output)


def run_example(model: Any, dataset: Any, checkpoint_path: str | None = None):
    """Direct script-first SFT call after external HF model/data setup."""

    config = make_picotron_config()
    return run_sft(
        model,
        dataset,
        base_checkpoint_path=checkpoint_path,
        learning_rate=config.optimizer.learning_rate_scheduler.learning_rate,
        batch_size=config.tokens.micro_batch_size,
        num_steps=config.tokens.train_steps,
        use_cache=False,
    )


if __name__ == "__main__":
    print(f"Configure {MODEL_ID} through an HF model factory; do not instantiate it here.")
