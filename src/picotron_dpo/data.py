"""Tokenization and batching for prompt/chosen/rejected preference triples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


PreferenceTriple = tuple[str, str, str]


class PreferenceDataset(Dataset[dict[str, Tensor]]):
    """Tokenize preference triples into causal-LM inputs and response masks.

    Prompts are deliberately preserved verbatim; callers own their chat-template
    formatting.  Loss labels mask the prompt and include response tokens only.
    """

    def __init__(
        self,
        triples: Sequence[PreferenceTriple | Mapping[str, str]] | Dataset[PreferenceTriple | Mapping[str, str]],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if len(triples) == 0:
            raise ValueError("Preference dataset must contain at least one triple.")
        self.triples = triples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        prompt, chosen, rejected = _normalize_triple(self.triples[index])
        chosen_ids, chosen_labels = _encode_completion(
            self.tokenizer, prompt, chosen, self.max_length
        )
        rejected_ids, rejected_labels = _encode_completion(
            self.tokenizer, prompt, rejected, self.max_length
        )
        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }


def collate_preference_batch(
    examples: Sequence[Mapping[str, Tensor]], *, pad_token_id: int
) -> dict[str, Tensor]:
    """Right-pad candidate sequences while keeping pad labels ignored."""

    if not examples:
        raise ValueError("Cannot collate an empty preference batch.")
    return {
        "chosen_input_ids": _pad_tensors(examples, "chosen_input_ids", pad_token_id),
        "chosen_labels": _pad_tensors(examples, "chosen_labels", -100),
        "rejected_input_ids": _pad_tensors(examples, "rejected_input_ids", pad_token_id),
        "rejected_labels": _pad_tensors(examples, "rejected_labels", -100),
    }


def infer_pad_token_id(tokenizer: Any) -> int:
    """Return a safe pad id for common Hugging Face-style tokenizers."""

    for attribute in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _normalize_triple(value: PreferenceTriple | Mapping[str, str]) -> PreferenceTriple:
    if isinstance(value, Mapping):
        try:
            prompt, chosen, rejected = value["prompt"], value["chosen"], value["rejected"]
        except KeyError as error:
            raise ValueError("Preference mappings require prompt, chosen, and rejected fields.") from error
    elif isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 3:
        prompt, chosen, rejected = value
    else:
        raise TypeError("Preference examples must be (prompt, chosen, rejected) triples or mappings.")
    if not all(isinstance(part, str) for part in (prompt, chosen, rejected)):
        raise TypeError("Prompt, chosen, and rejected values must be strings.")
    return prompt, chosen, rejected


def _encode_completion(
    tokenizer: Any, prompt: str, completion: str, max_length: int
) -> tuple[list[int], list[int]]:
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("Prompt must tokenize to at least one token for causal DPO.")
    if not completion_ids:
        raise ValueError("Chosen and rejected completions must tokenize to at least one token.")

    completion_ids = completion_ids[:max_length]
    prompt_budget = max(0, max_length - len(completion_ids))
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) < 2:
        raise ValueError("Preference examples must contain at least two combined tokens.")
    return input_ids, labels


def _pad_tensors(examples: Sequence[Mapping[str, Tensor]], key: str, value: int) -> Tensor:
    tensors = [example[key] for example in examples]
    if not all(isinstance(tensor, Tensor) and tensor.ndim == 1 for tensor in tensors):
        raise TypeError(f"'{key}' values must be one-dimensional tensors.")
    max_length = max(tensor.size(0) for tensor in tensors)
    padded = torch.full((len(tensors), max_length), value, dtype=torch.long)
    for row, tensor in enumerate(tensors):
        padded[row, : tensor.size(0)] = tensor
    return padded
