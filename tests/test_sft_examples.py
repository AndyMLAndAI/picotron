"""Interface-only checks for model-family SFT examples; no weights downloaded."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import pytest
import torch


@dataclass
class _HFStyleOutput:
    logits: torch.Tensor


EXAMPLE_NAMES = ("sft_qwen3_5", "sft_qwen3", "sft_llama_style")


def _load_example(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_sft_example_config_and_hf_output_contract(name: str) -> None:
    example = _load_example(name)
    config = example.make_picotron_config()
    output = _HFStyleOutput(torch.zeros(1, 2, config.vocab_size))

    logits = example.logits_from_model_output(output)

    assert logits.shape == (1, 2, config.vocab_size)
    assert config.hidden_size % config.num_attention_heads == 0
    assert example.MODEL_SPECIFIC_KWARGS

