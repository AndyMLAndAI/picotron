"""Offline checks for the opt-in real-model compatibility scripts."""

from __future__ import annotations

import runpy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_spec(filename: str) -> object:
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        return runpy.run_path(EXAMPLES_DIR / filename)["SPEC"]
    finally:
        sys.path.pop(0)


def test_real_model_example_specs_are_complete_and_explicit() -> None:
    specs = {
        filename: _load_spec(filename)
        for filename in (
            "verify_qwen2_5.py",
            "verify_qwen3.py",
            "verify_qwen3_5.py",
            "verify_starcoder2.py",
            "verify_llama_3_2.py",
            "verify_mistral.py",
            "verify_gemma_2.py",
        )
    }

    assert all(getattr(spec, "model_id") for spec in specs.values())
    assert all(getattr(spec, "expected_model_type") for spec in specs.values())
    assert not specs["verify_qwen3_5.py"].generic_causal_lm
    assert specs["verify_qwen3_5.py"].loader_kind == "multimodal_lm"
    assert all(
        spec.generic_causal_lm
        for filename, spec in specs.items()
        if filename != "verify_qwen3_5.py"
    )


def test_unknown_verification_loader_fails_clearly_before_importing_runtime() -> None:
    spec = _load_spec("verify_qwen3_5.py")
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        from _hf_model_verification import run_training_smoke

        with pytest.raises(RuntimeError, match="Unsupported verification loader"):
            run_training_smoke(
                replace(spec, loader_kind="unsupported"),
                train_steps=1,
                dpo_steps=0,
                device="cpu",
                token=None,
            )
    finally:
        sys.path.pop(0)
