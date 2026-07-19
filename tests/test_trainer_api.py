"""Equivalence checks for the TRL-style SFT and DPO adapters."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from picotron.config.config import load_config
from picotron.models.toy_model import ToyDecoderModel
from picotron_dpo import PicotronDPOConfig, PicotronDPOTrainer, run_dpo
from picotron_sft import PicotronSFTConfig, PicotronSFTTrainer, run_sft


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=True)}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        token_ids = {"P": 1, "A": 2, "B": 3, "C": 4}
        return [token_ids[character] for character in text]


class _TextDataset(Dataset[dict[str, str]]):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> dict[str, str]:
        del index
        return {"text": "AB"}


class _TokenizedDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        return {
            "input_ids": torch.tensor([2, 3, 0, 0]),
            "labels": torch.tensor([2, 3, -100, -100]),
        }


def _toy_config():
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/toy_model.yaml"
    loaded = load_config(config_path)
    return replace(loaded, tokens=replace(loaded.tokens, sequence_length=4, micro_batch_size=2))


def test_sft_trainer_delegates_to_run_sft_with_identical_losses() -> None:
    config = _toy_config()
    tokenizer = _Tokenizer()
    torch.manual_seed(13)
    direct_model = ToyDecoderModel(config)
    wrapper_model = copy.deepcopy(direct_model)

    direct_losses = run_sft(
        direct_model,
        _TokenizedDataset(),
        optimizer=AdamW(direct_model.parameters(), lr=0.01),
        learning_rate=0.01,
        batch_size=2,
        num_steps=4,
    )
    wrapped_losses = PicotronSFTTrainer(
        wrapper_model,
        tokenizer,
        _TextDataset(),
        args=PicotronSFTConfig(
            per_device_train_batch_size=2,
            max_steps=4,
            max_seq_length=4,
            learning_rate=0.01,
            seed=13,
        ),
    ).train()

    assert wrapped_losses == pytest.approx(direct_losses, abs=1e-7)


def test_dpo_trainer_delegates_to_run_dpo_with_identical_losses() -> None:
    config = _toy_config()
    tokenizer = _Tokenizer()
    preferences = [("P", "A", "B")] * 12
    torch.manual_seed(17)
    direct_model = ToyDecoderModel(config)
    wrapper_model = copy.deepcopy(direct_model)

    direct_losses = run_dpo(
        direct_model,
        preferences,
        tokenizer=tokenizer,
        optimizer=AdamW(direct_model.parameters(), lr=0.01),
        learning_rate=0.01,
        batch_size=1,
        max_length=4,
        num_steps=4,
    )
    wrapped_losses = PicotronDPOTrainer(
        wrapper_model,
        tokenizer,
        preferences,
        args=PicotronDPOConfig(
            per_device_train_batch_size=1,
            max_steps=4,
            max_length=4,
            learning_rate=0.01,
            seed=17,
        ),
    ).train()

    assert wrapped_losses == pytest.approx(direct_losses, abs=1e-7)
