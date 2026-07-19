"""Label-aware full fine-tuning adapter using Picotron runtime components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset

from picotron.config.config import PicotronConfig
from picotron.logging.display import TrainingDisplay
from picotron.logging.file_logger import FileLogger
from picotron.parallel.ddp import initialize_distributed, wrap_model
from picotron.serialize.checkpoint import load_checkpoint
class SFTTrainer:
    """Fine-tune a causal LM on standard ``input_ids``/``labels`` batches.

    Picotron's pretraining loop owns its fixed next-token tensor-batch API.
    This thin adapter reuses its runtime services while supplying the labels
    required by general SFT datasets.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable[Mapping[str, Tensor] | tuple[Tensor, Tensor]],
        *,
        base_checkpoint_path: str | None = None,
        learning_rate: float = 1e-5,
        num_steps: int | None = None,
        display_config: Any | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        optimizer: Optimizer | None = None,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if num_steps is not None and num_steps <= 0:
            raise ValueError("num_steps must be positive when provided.")
        self.model = model
        self.data_loader = data_loader
        self.base_checkpoint_path = base_checkpoint_path
        self.num_steps = num_steps
        self.display_config = display_config
        self.model_kwargs = dict(model_kwargs or {})
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = optimizer or AdamW(self.model.parameters(), lr=learning_rate)
        self.resumed_step = 0
        self._checkpoint_loaded = False

    def load_pretrained(self) -> int:
        """Load the configured Picotron checkpoint into this trainer's objects."""

        if self.base_checkpoint_path is not None and not self._checkpoint_loaded:
            self.resumed_step = load_checkpoint(
                self.model, self.optimizer, self.base_checkpoint_path
            )
            self._checkpoint_loaded = True
        return self.resumed_step

    def train(self, *, num_steps: int | None = None) -> list[float]:
        """Run SFT steps and return the per-step causal language-model losses."""

        step_limit = num_steps if num_steps is not None else self.num_steps
        if step_limit is not None and step_limit <= 0:
            raise ValueError("max_steps must be positive when provided.")
        self.load_pretrained()
        distributed_info = initialize_distributed()
        self.model = wrap_model(self.model, distributed_info, device=self.device)
        self.model.train()
        losses: list[float] = []

        display = _make_display(self.display_config, step_limit)
        file_config = self.display_config if isinstance(self.display_config, PicotronConfig) else None
        with display, FileLogger(file_config, method="sft") as file_logger:
            for batch in self.data_loader:
                if step_limit is not None and len(losses) >= step_limit:
                    return losses
                model_inputs, labels = _unpack_batch(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = _extract_logits(self.model(**{**self.model_kwargs, **model_inputs}))
                loss = _causal_loss(logits, labels)
                loss.backward()
                self.optimizer.step()

                loss_value = loss.detach().cpu().item()
                losses.append(loss_value)
                step = self.resumed_step + len(losses)
                display.update(
                    step=step,
                    loss=loss_value,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    tokens_seen=step * model_inputs["input_ids"].numel(),
                )
                file_logger.log_step(
                    step=step,
                    loss=loss_value,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    tokens_seen=step * model_inputs["input_ids"].numel(),
                )
        return losses


def run_sft(
    model: nn.Module,
    dataset: Dataset[Any] | Iterable[Mapping[str, Tensor] | tuple[Tensor, Tensor]],
    *,
    base_checkpoint_path: str | None = None,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    num_steps: int | None = None,
    device: torch.device | str = torch.device("cpu"),
    optimizer: Optimizer | None = None,
    display_config: Any | None = None,
    **model_kwargs: Any,
) -> list[float]:
    """Fine-tune a causal LM directly from ordinary Python objects.

    ``dataset`` may be a PyTorch Dataset (batched with ``batch_size``) or an
    iterable that already yields ``(input_ids, labels)`` or mapping batches.
    Extra keyword arguments are forwarded to the model on every SFT step.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    data_loader = _as_data_loader(dataset, batch_size)
    trainer = SFTTrainer(
        model,
        data_loader,
        base_checkpoint_path=base_checkpoint_path,
        learning_rate=learning_rate,
        num_steps=num_steps,
        display_config=display_config,
        model_kwargs=model_kwargs,
        optimizer=optimizer,
        device=device,
    )
    return trainer.train()


def _as_data_loader(
    dataset: Dataset[Any] | Iterable[Mapping[str, Tensor] | tuple[Tensor, Tensor]],
    batch_size: int,
) -> Iterable[Mapping[str, Tensor] | tuple[Tensor, Tensor]]:
    if isinstance(dataset, DataLoader):
        return dataset
    if isinstance(dataset, Dataset):
        return DataLoader(dataset, batch_size=batch_size)
    return dataset


def _make_display(
    display_config: Any | None, total_steps: int | None
) -> TrainingDisplay | AbstractContextManager[None]:
    return TrainingDisplay(display_config, total_steps=total_steps) if display_config else _NullDisplay()


class _NullDisplay(AbstractContextManager[None]):
    """Keep direct SFT scripts independent of Picotron's YAML display config."""

    def update(self, **_: Any) -> None:
        pass

    def __enter__(self) -> "_NullDisplay":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


def _unpack_batch(
    batch: Mapping[str, Tensor] | tuple[Tensor, Tensor], device: torch.device
) -> tuple[dict[str, Tensor], Tensor]:
    if isinstance(batch, Mapping):
        try:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
        except KeyError as error:
            raise ValueError("SFT mapping batches require 'input_ids' and 'labels'.") from error
        model_inputs = {
            name: value.to(device=device)
            for name, value in batch.items()
            if name != "labels" and isinstance(value, Tensor)
        }
        if len(model_inputs) != len(batch) - 1:
            raise TypeError("All SFT model input fields must be tensors.")
    elif isinstance(batch, tuple) and len(batch) == 2:
        input_ids, labels = batch
        model_inputs = {"input_ids": input_ids}
    else:
        raise TypeError("SFT batches must be mappings or (input_ids, labels) tuples.")
    if not isinstance(input_ids, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("SFT input_ids and labels must be tensors.")
    model_inputs["input_ids"] = input_ids.to(device=device, dtype=torch.long)
    return model_inputs, labels.to(device=device, dtype=torch.long)


def _extract_logits(model_output: Any) -> Tensor:
    if isinstance(model_output, Tensor):
        return model_output
    if hasattr(model_output, "logits") and isinstance(model_output.logits, Tensor):
        return model_output.logits
    if isinstance(model_output, Mapping) and isinstance(model_output.get("logits"), Tensor):
        return model_output["logits"]
    raise TypeError("Model output must be logits or expose a Tensor 'logits' attribute.")


def _causal_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 3:
        raise ValueError("Causal LM logits must have shape (batch, sequence, vocab_size).")
    if labels.shape != logits.shape[:2]:
        raise ValueError("SFT labels must match the batch and sequence dimensions of logits.")
    if logits.size(1) < 2:
        raise ValueError("SFT batches require sequences of at least two tokens.")
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
