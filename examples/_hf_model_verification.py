"""Opt-in Hugging Face model-family compatibility smoke-test helpers.

These helpers deliberately keep real-weight loading opt-in: the smallest
listed checkpoints range from 0.5B to 9B parameters, so a config-only check
is safe by default while a training smoke run requires deliberate hardware and
Hub access.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelVerificationSpec:
    """A published checkpoint and its Picotron compatibility expectation."""

    family: str
    model_id: str
    expected_model_type: str
    generic_causal_lm: bool
    notes: str
    loader_kind: str = "picotron_load_model"


def main(spec: ModelVerificationSpec) -> None:
    """Inspect a model config and optionally run tiny SFT/DPO updates."""

    parser = argparse.ArgumentParser(description=f"Verify {spec.family} with Picotron.")
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help="Run this many SFT updates after loading real weights (default: config only).",
    )
    parser.add_argument(
        "--dpo-steps",
        type=int,
        default=0,
        help="Run this many DPO updates after the SFT smoke run (default: disabled).",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for opt-in training.")
    parser.add_argument("--token", default=None, help="HF token for gated checkpoints.")
    args = parser.parse_args()
    if args.train_steps < 0 or args.dpo_steps < 0:
        parser.error("--train-steps and --dpo-steps must be non-negative.")
    if args.dpo_steps and not args.train_steps:
        parser.error("--dpo-steps requires --train-steps so the policy model is loaded.")

    config = load_and_check_config(spec, token=args.token)
    print(f"{spec.family}: model_type={config.model_type}; expected={spec.expected_model_type}")
    print(spec.notes)
    if args.train_steps:
        run_training_smoke(
            spec,
            train_steps=args.train_steps,
            dpo_steps=args.dpo_steps,
            device=args.device,
            token=args.token,
        )


def load_and_check_config(spec: ModelVerificationSpec, *, token: str | None = None) -> Any:
    """Fetch only the official config and reject an unexpected architecture."""

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(spec.model_id, token=token)
    if config.model_type != spec.expected_model_type:
        raise RuntimeError(
            f"{spec.model_id} reported model_type={config.model_type!r}; "
            f"expected {spec.expected_model_type!r}."
        )
    if getattr(config, "is_encoder_decoder", False):
        raise RuntimeError(f"{spec.model_id} is not a decoder-only causal LM.")
    return config


def run_training_smoke(
    spec: ModelVerificationSpec,
    *,
    train_steps: int,
    dpo_steps: int,
    device: str,
    token: str | None,
) -> None:
    """Load real weights, then run tiny tensor-format SFT and optional DPO updates."""

    if spec.loader_kind not in {"picotron_load_model", "multimodal_lm"}:
        raise RuntimeError(f"Unsupported verification loader: {spec.loader_kind!r}.")

    import torch

    from picotron_dpo import run_dpo
    from picotron_sft import run_sft

    if spec.loader_kind == "picotron_load_model":
        from picotron_sft import load_model

        model, tokenizer = load_model(spec.model_id, max_seq_length=16, token=token)
    elif spec.loader_kind == "multimodal_lm":
        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        model = AutoModelForMultimodalLM.from_pretrained(spec.model_id, token=token)
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, token=token)
    model.config.use_cache = False
    model.to(device)
    encoded = tokenizer(
        "Picotron model-agnostic compatibility smoke test.",
        return_tensors="pt",
        truncation=True,
        max_length=16,
    )
    input_ids = encoded["input_ids"].to(dtype=torch.long)
    if input_ids.size(1) < 2:
        raise RuntimeError("The selected tokenizer produced fewer than two tokens.")
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}
    if "attention_mask" in encoded:
        batch["attention_mask"] = encoded["attention_mask"]

    sft_losses = run_sft(
        model,
        [batch] * train_steps,
        learning_rate=1e-6,
        num_steps=train_steps,
        device=device,
    )
    print(f"SFT losses: {sft_losses}")
    if dpo_steps:
        dpo_losses = run_dpo(
            model,
            [("Prompt: choose the accurate response.\nAnswer: ", "correct", "incorrect")],
            tokenizer=tokenizer,
            beta=0.1,
            learning_rate=1e-6,
            num_steps=dpo_steps,
            device=device,
        )
        print(f"DPO losses: {dpo_losses}")
