"""Picotron SFT wiring sketch for the dense Qwen3-8B model."""

from __future__ import annotations

from typing import Any

from picotron.config.config import PicotronConfig
from picotron_sft import run_sft
from picotron_sft.sft_trainer import _extract_logits

# Model-specific: these are Qwen3-8B dimensions. A real run must provide the
# HF weights/tokenizer and preserve Qwen chat-template tokenization. Picotron
# SFT generically handles causal labels, checkpoint restoration, and runtime.
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_SPECIFIC_KWARGS = {
    "model_type": "qwen3",
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rope_theta": 1_000_000,
}


def make_picotron_config() -> PicotronConfig:
    """Return base dimensions matching Qwen3-8B."""

    return PicotronConfig(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        max_seq_len=40960,
        learning_rate=1e-5,
        batch_size=1,
        num_epochs=1,
        checkpoint_interval=100,
        model_kwargs=MODEL_SPECIFIC_KWARGS,
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
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_steps=100,
        use_cache=False,
    )


if __name__ == "__main__":
    print(f"Configure {MODEL_ID} through an HF model factory; do not instantiate it here.")
