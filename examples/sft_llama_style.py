"""Picotron SFT wiring sketch for a Llama 3-style causal LM."""

from __future__ import annotations

from typing import Any

from picotron.config.config import PicotronConfig
from picotron_sft import run_sft
from picotron_sft.sft_trainer import _extract_logits

# Model-specific: Llama 3 8B uses RMSNorm, SwiGLU, RoPE, and GQA. Picotron's
# toy model shares the RMSNorm/SwiGLU decoder style, but real Llama weights and
# tokenizer/chat-template loading belong in an external HF model factory.
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MODEL_SPECIFIC_KWARGS = {
    "architecture": "RMSNorm + SwiGLU + RoPE + GQA",
    "num_key_value_heads": 8,
    "rope_theta": 500_000,
}


def make_picotron_config() -> PicotronConfig:
    """Return representative Llama 3 8B dimensions."""

    return PicotronConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_seq_len=8192,
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
