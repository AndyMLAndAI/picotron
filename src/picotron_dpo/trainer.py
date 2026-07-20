"""TRL-shaped thin wrapper around Picotron's script-first DPO API."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from picotron.config.config import PicotronConfig
from picotron_dpo.dpo_trainer import run_dpo


LOGGER = logging.getLogger("picotron_dpo")


@dataclass(frozen=True, slots=True)
class PicotronDPOConfig:
    """Supported TRL-style arguments for :class:`PicotronDPOTrainer`."""

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
    max_length: int = 1024
    beta: float = 0.1
    device: torch.device | str = torch.device("cpu")
    display_config: PicotronConfig | None = None

    def __post_init__(self) -> None:
        for name in ("per_device_train_batch_size", "gradient_accumulation_steps", "logging_steps", "max_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided.")
        if self.learning_rate <= 0 or self.beta <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate and beta must be positive; weight_decay must be non-negative.")
        if self.optim not in {"adamw_torch", "adamw"}:
            raise ValueError("Only AdamW ('adamw_torch' or 'adamw') is implemented.")
        if self.gradient_accumulation_steps != 1:
            raise NotImplementedError("Gradient accumulation is not implemented by Picotron's DPO loop yet.")
        if self.warmup_steps != 0 or self.lr_scheduler_type != "constant":
            raise NotImplementedError("DPO currently supports only a constant learning rate with no warmup.")


class PicotronDPOTrainer:
    """TRL-familiar adapter that delegates optimization to :func:`run_dpo`."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        args: PicotronDPOConfig | None = None,
        ref_model: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args or PicotronDPOConfig()
        self.ref_model = ref_model

    def train(self) -> list[float]:
        """Call the existing DPO functional API with equivalent arguments."""

        if self.eval_dataset is not None:
            LOGGER.info("eval_dataset is accepted for TRL compatibility but evaluation is not implemented yet.")
        torch.manual_seed(self.args.seed)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return run_dpo(
            self.model,
            self.train_dataset,
            tokenizer=self.tokenizer,
            ref_model=self.ref_model,
            beta=self.args.beta,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            batch_size=self.args.per_device_train_batch_size,
            max_length=self.args.max_length,
            num_steps=self.args.max_steps,
            optimizer=optimizer,
            device=self.args.device,
            display_config=_display_config(self.args.display_config, self.args.logging_steps),
        )


def _display_config(config: PicotronConfig | None, logging_steps: int) -> PicotronConfig | None:
    if config is None:
        return None
    return replace(
        config,
        logging=replace(config.logging, iteration_step_info_interval=logging_steps),
    )
