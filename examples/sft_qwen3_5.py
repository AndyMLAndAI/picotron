"""Picotron SFT wiring sketch for the hybrid Qwen3.5-27B text model."""

from __future__ import annotations

from typing import Any

from picotron.config.config import PicotronConfig
from picotron_sft import run_sft
from picotron_sft.sft_trainer import _extract_logits

# Model-specific: Qwen3.5 text defaults. The hybrid 3:1 linear-attention /
# full-attention stack must be instantiated by Transformers/Qwen code, not by
# Picotron. Real HF weights and the matching tokenizer are extension points.
MODEL_ID = "Qwen/Qwen3.5-27B"
MODEL_SPECIFIC_KWARGS = {
    "model_type": "qwen3_5",
    "num_key_value_heads": 4,
    "head_dim": 256,
    "layer_pattern": "3:1 GatedDeltaNet linear_attention/full_attention",
}


def make_picotron_config() -> PicotronConfig:
    """Return base dimensions matching Qwen3.5-27B's text configuration."""

    return PicotronConfig(
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        max_seq_len=32768,
        learning_rate=1e-5,
        batch_size=1,
        num_epochs=1,
        checkpoint_interval=100,
        model_kwargs=MODEL_SPECIFIC_KWARGS,
    )


def logits_from_model_output(model_output: Any):
    """The generic Picotron SFT output contract: Tensor or HF-style ``.logits``."""

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
