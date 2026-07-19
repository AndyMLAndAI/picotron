"""A script-first implementation of Direct Preference Optimization."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset

from picotron.logging.display import TrainingDisplay
from picotron.serialize.checkpoint import load_checkpoint
from picotron_dpo.data import (
    PreferenceDataset,
    PreferenceTriple,
    collate_preference_batch,
    infer_pad_token_id,
)


class DPOTrainer:
    """Train a policy model against a frozen reference model using DPO."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable[Mapping[str, Tensor]],
        *,
        ref_model: nn.Module | None = None,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        num_steps: int | None = None,
        optimizer: Optimizer | None = None,
        device: torch.device | str = torch.device("cpu"),
        display_config: Any | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        base_checkpoint_path: str | Path | None = None,
    ) -> None:
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if num_steps is not None and num_steps <= 0:
            raise ValueError("num_steps must be positive when provided.")
        if ref_model is model:
            raise ValueError("ref_model must be a separate model from the trainable policy.")

        self.model = model
        self.data_loader = data_loader
        self.beta = beta
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.display_config = display_config
        self.model_kwargs = dict(model_kwargs or {})
        self.model.to(self.device)
        self.optimizer = optimizer or AdamW(self.model.parameters(), lr=learning_rate)
        if base_checkpoint_path is not None:
            load_checkpoint(self.model, self.optimizer, base_checkpoint_path)

        self.ref_model = copy.deepcopy(self.model) if ref_model is None else ref_model
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    def train(self, *, num_steps: int | None = None) -> list[float]:
        """Run DPO updates and return one scalar loss per optimizer step."""

        step_limit = self.num_steps if num_steps is None else num_steps
        if step_limit is not None and step_limit <= 0:
            raise ValueError("num_steps must be positive when provided.")
        self.model.train()
        self.ref_model.eval()
        losses: list[float] = []

        display = _make_display(self.display_config, step_limit)
        with display:
            for batch in self.data_loader:
                if step_limit is not None and len(losses) >= step_limit:
                    break
                prepared = _prepare_batch(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self._dpo_loss_and_metrics(prepared)
                loss.backward()
                self.optimizer.step()

                loss_value = loss.detach().cpu().item()
                losses.append(loss_value)
                display.update(
                    step=len(losses),
                    loss=loss_value,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    tokens_seen=len(losses)
                    * (
                        prepared["chosen_input_ids"].numel()
                        + prepared["rejected_input_ids"].numel()
                    ),
                    metrics=metrics,
                )
        return losses

    def _dpo_loss(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Return only the DPO loss for callers that do not need display metrics."""

        return self._dpo_loss_and_metrics(batch)[0]

    def _dpo_loss_and_metrics(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Compute the standard DPO objective plus policy log-prob diagnostics."""

        policy_chosen = _sequence_log_probability(
            _extract_logits(self.model(batch["chosen_input_ids"], **self.model_kwargs)),
            batch["chosen_labels"],
        )
        policy_rejected = _sequence_log_probability(
            _extract_logits(self.model(batch["rejected_input_ids"], **self.model_kwargs)),
            batch["rejected_labels"],
        )
        with torch.no_grad():
            reference_chosen = _sequence_log_probability(
                _extract_logits(
                    self.ref_model(batch["chosen_input_ids"], **self.model_kwargs)
                ),
                batch["chosen_labels"],
            )
            reference_rejected = _sequence_log_probability(
                _extract_logits(
                    self.ref_model(batch["rejected_input_ids"], **self.model_kwargs)
                ),
                batch["rejected_labels"],
            )

        policy_log_ratio = policy_chosen - policy_rejected
        reference_log_ratio = reference_chosen - reference_rejected
        loss = -F.logsigmoid(self.beta * (policy_log_ratio - reference_log_ratio)).mean()
        return loss, {
            "chosen_logprob": policy_chosen.detach().mean().item(),
            "rejected_logprob": policy_rejected.detach().mean().item(),
            "margin": policy_log_ratio.detach().mean().item(),
        }


def run_dpo(
    model: nn.Module,
    dataset: (
        Sequence[PreferenceTriple | Mapping[str, str]]
        | Dataset[PreferenceTriple | Mapping[str, str] | dict[str, Tensor]]
        | Iterable[Mapping[str, Tensor]]
    ),
    *,
    tokenizer: Any | None = None,
    ref_model: nn.Module | None = None,
    beta: float = 0.1,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    max_length: int = 1024,
    num_steps: int | None = None,
    optimizer: Optimizer | None = None,
    device: torch.device | str = torch.device("cpu"),
    display_config: Any | None = None,
    base_checkpoint_path: str | Path | None = None,
    **model_kwargs: Any,
) -> list[float]:
    """Run DPO directly from text triples or already-tokenized preference batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    data_loader = _as_preference_dataloader(dataset, tokenizer, batch_size, max_length)
    trainer = DPOTrainer(
        model,
        data_loader,
        ref_model=ref_model,
        beta=beta,
        learning_rate=learning_rate,
        num_steps=num_steps,
        optimizer=optimizer,
        device=device,
        display_config=display_config,
        model_kwargs=model_kwargs,
        base_checkpoint_path=base_checkpoint_path,
    )
    return trainer.train()


def _as_preference_dataloader(
    dataset: (
        Sequence[PreferenceTriple | Mapping[str, str]]
        | Dataset[PreferenceTriple | Mapping[str, str] | dict[str, Tensor]]
        | Iterable[Mapping[str, Tensor]]
    ),
    tokenizer: Any | None,
    batch_size: int,
    max_length: int,
) -> Iterable[Mapping[str, Tensor]]:
    if isinstance(dataset, DataLoader):
        return dataset
    if isinstance(dataset, PreferenceDataset):
        active_tokenizer = tokenizer if tokenizer is not None else dataset.tokenizer
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda examples: collate_preference_batch(
                examples, pad_token_id=infer_pad_token_id(active_tokenizer)
            ),
        )
    if isinstance(dataset, Dataset) or _is_map_style_dataset(dataset):
        if tokenizer is not None:
            text_dataset = PreferenceDataset(dataset, tokenizer, max_length=max_length)
            return DataLoader(
                text_dataset,
                batch_size=batch_size,
                collate_fn=lambda examples: collate_preference_batch(
                    examples, pad_token_id=infer_pad_token_id(tokenizer)
                ),
            )
        return DataLoader(dataset, batch_size=batch_size)
    if isinstance(dataset, Sequence):
        if tokenizer is None:
            raise ValueError("tokenizer is required when dataset yields text preference triples.")
        text_dataset = PreferenceDataset(dataset, tokenizer, max_length=max_length)
        return DataLoader(
            text_dataset,
            batch_size=batch_size,
            collate_fn=lambda examples: collate_preference_batch(
                examples, pad_token_id=infer_pad_token_id(tokenizer)
            ),
        )
    return dataset


def _is_map_style_dataset(dataset: object) -> bool:
    """Recognize Hugging Face-style datasets without importing datasets."""

    return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")


def _prepare_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    required = {
        "chosen_input_ids",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_labels",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"DPO batch is missing fields: {sorted(missing)}.")
    prepared = {name: batch[name].to(device=device, dtype=torch.long) for name in required}
    if prepared["chosen_input_ids"].shape != prepared["chosen_labels"].shape:
        raise ValueError("chosen input_ids and labels must have the same shape.")
    if prepared["rejected_input_ids"].shape != prepared["rejected_labels"].shape:
        raise ValueError("rejected input_ids and labels must have the same shape.")
    return prepared


def _sequence_log_probability(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Logits and labels must have shapes (batch, sequence, vocab) and (batch, sequence).")
    if logits.size(1) < 2:
        raise ValueError("DPO candidate sequences require at least two tokens.")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    if not torch.all(valid.any(dim=1)):
        raise ValueError("Every DPO candidate must contain at least one unmasked response token.")
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_log_probs = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * valid).sum(dim=-1)


def _extract_logits(model_output: Any) -> Tensor:
    if isinstance(model_output, Tensor):
        return model_output
    if hasattr(model_output, "logits") and isinstance(model_output.logits, Tensor):
        return model_output.logits
    if isinstance(model_output, Mapping) and isinstance(model_output.get("logits"), Tensor):
        return model_output["logits"]
    raise TypeError("Model output must be logits or expose a Tensor 'logits' attribute.")


def _make_display(
    display_config: Any | None, total_steps: int | None
) -> TrainingDisplay | AbstractContextManager[None]:
    return (
        TrainingDisplay(display_config, total_steps=total_steps, loss_label="dpo_loss")
        if display_config
        else _NullDisplay()
    )


class _NullDisplay(AbstractContextManager[None]):
    def update(self, **_: Any) -> None:
        pass

    def __enter__(self) -> "_NullDisplay":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None
