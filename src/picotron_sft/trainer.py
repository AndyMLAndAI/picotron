"""TRL-shaped thin wrapper around Picotron's script-first SFT API."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import Dataset

from picotron.config.config import PicotronConfig
from picotron_sft.sft_trainer import run_sft


LOGGER = logging.getLogger("picotron_sft")


@dataclass(frozen=True, slots=True)
class PicotronSFTConfig:
    """Supported TRL-style arguments for :class:`PicotronSFTTrainer`."""

    dataset_text_field: str = "text"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    max_steps: int | None = None
    learning_rate: float = 1e-5
    logging_steps: int = 10
    optim: str = "adamw_torch"
    weight_decay: float = 0.0
    lr_scheduler_type: str = "constant"
    seed: int = 3407
    max_seq_length: int = 1024
    device: torch.device | str = torch.device("cpu")
    display_config: PicotronConfig | None = None

    def __post_init__(self) -> None:
        if not self.dataset_text_field:
            raise ValueError("dataset_text_field must be non-empty.")
        for name in ("per_device_train_batch_size", "gradient_accumulation_steps", "logging_steps", "max_seq_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.optim not in {"adamw_torch", "adamw"}:
            raise ValueError("Only AdamW ('adamw_torch' or 'adamw') is implemented.")
        if self.gradient_accumulation_steps != 1:
            raise NotImplementedError("Gradient accumulation is not implemented by Picotron's SFT loop yet.")
        if self.warmup_steps != 0 or self.lr_scheduler_type != "constant":
            raise NotImplementedError("SFT currently supports only a constant learning rate with no warmup.")


class PicotronSFTTrainer:
    """TRL-familiar adapter that delegates optimization to :func:`run_sft`."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: Dataset[Any],
        eval_dataset: Dataset[Any] | None = None,
        args: PicotronSFTConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args or PicotronSFTConfig()

    def train(self) -> list[float]:
        """Tokenize the configured text field and call the existing SFT API."""

        if self.eval_dataset is not None:
            LOGGER.info("eval_dataset is accepted for TRL compatibility but evaluation is not implemented yet.")
        torch.manual_seed(self.args.seed)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        dataset = _SFTTextDataset(
            self.train_dataset,
            self.tokenizer,
            text_field=self.args.dataset_text_field,
            max_length=self.args.max_seq_length,
            include_attention_mask=_accepts_attention_mask(self.model),
        )
        return run_sft(
            self.model,
            dataset,
            learning_rate=self.args.learning_rate,
            batch_size=self.args.per_device_train_batch_size,
            num_steps=self.args.max_steps,
            optimizer=optimizer,
            device=self.args.device,
            display_config=_display_config(self.args.display_config, self.args.logging_steps),
        )


class _SFTTextDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        dataset: Dataset[Any],
        tokenizer: Any,
        *,
        text_field: str,
        max_length: int,
        include_attention_mask: bool,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length
        self.include_attention_mask = include_attention_mask
        self.pad_token_id = _pad_token_id(tokenizer)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        example = self.dataset[index]
        if not isinstance(example, Mapping) or not isinstance(example.get(self.text_field), str):
            raise TypeError(f"SFT examples must map '{self.text_field}' to a string.")
        token_ids = _tokenize(self.tokenizer, example[self.text_field], self.max_length)
        padding = self.max_length - len(token_ids)
        input_ids = token_ids + [self.pad_token_id] * padding
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(token_ids + [-100] * padding, dtype=torch.long),
        }
        if self.include_attention_mask:
            result["attention_mask"] = torch.tensor([1] * len(token_ids) + [0] * padding, dtype=torch.long)
        return result


def _tokenize(tokenizer: Any, text: str, max_length: int) -> list[int]:
    if callable(tokenizer):
        encoded = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        token_ids = encoded["input_ids"]
    else:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    if not token_ids:
        raise ValueError("SFT text must tokenize to at least one token.")
    return list(token_ids[:max_length])


def _pad_token_id(tokenizer: Any) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _accepts_attention_mask(model: nn.Module) -> bool:
    import inspect

    signature = inspect.signature(model.forward)
    return "attention_mask" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _display_config(config: PicotronConfig | None, logging_steps: int) -> PicotronConfig | None:
    if config is None:
        return None
    return replace(
        config,
        logging=replace(config.logging, iteration_step_info_interval=logging_steps),
    )
