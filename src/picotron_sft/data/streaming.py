"""Lazy Hugging Face streaming batches with batched tokenization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from queue import Queue
from threading import Thread
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import IterableDataset


class StreamingSFTDataset(IterableDataset[dict[str, Tensor]]):
    """Turn raw streaming text examples into padded, causal-LM training batches.

    A producer thread buffers raw examples; tokenization occurs only when each
    completed batch is consumed, avoiding any corpus-wide preprocessing step.
    """

    def __init__(
        self,
        source: Iterable[Mapping[str, Any]],
        tokenizer: Any,
        *,
        batch_size: int,
        text_field: str = "text",
        max_length: int | None = None,
        prefetch_batches: int = 1,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive when provided.")
        if prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be positive.")
        self.source = source
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.text_field = text_field
        self.max_length = max_length
        self.prefetch_batches = prefetch_batches

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        tokenizer: Any,
        *,
        split: str = "train",
        batch_size: int,
        text_field: str = "text",
        max_length: int | None = None,
        **dataset_kwargs: Any,
    ) -> "StreamingSFTDataset":
        """Create a lazy dataset backed by ``datasets.load_dataset(..., streaming=True)``."""

        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face streaming requires the optional 'datasets' package."
            ) from error
        source = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            **dataset_kwargs,
        )
        return cls(
            source,
            tokenizer,
            batch_size=batch_size,
            text_field=text_field,
            max_length=max_length,
        )

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        queue: Queue[list[Mapping[str, Any]] | BaseException | None] = Queue(
            maxsize=self.prefetch_batches
        )
        producer = Thread(target=self._prefetch_raw_batches, args=(queue,), daemon=True)
        producer.start()

        while True:
            item = queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield self._tokenize_batch(item)

    def _prefetch_raw_batches(
        self, queue: Queue[list[Mapping[str, Any]] | BaseException | None]
    ) -> None:
        try:
            batch: list[Mapping[str, Any]] = []
            for example in self.source:
                batch.append(example)
                if len(batch) == self.batch_size:
                    queue.put(batch)
                    batch = []
            if batch:
                queue.put(batch)
            queue.put(None)
        except BaseException as error:
            queue.put(error)

    def _tokenize_batch(self, raw_batch: list[Mapping[str, Any]]) -> dict[str, Tensor]:
        texts = [self._extract_text(example) for example in raw_batch]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "input_ids" not in encoded:
            raise ValueError("Tokenizer output must contain 'input_ids'.")
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(
            encoded.get("attention_mask", input_ids.ne(0)), dtype=torch.long
        )
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Tokenizer must return matching 2D input_ids and attention_mask.")
        labels = input_ids.clone()
        labels.masked_fill_(attention_mask.eq(0), -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _extract_text(self, example: Mapping[str, Any]) -> str:
        text = example.get(self.text_field)
        if not isinstance(text, str):
            raise ValueError(f"Streaming example field '{self.text_field}' must be a string.")
        return text

