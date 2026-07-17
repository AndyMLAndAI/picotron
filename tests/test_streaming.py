"""Offline checks for batched tokenization of a fake streaming source."""

from typing import Any

import torch

from picotron_sft.data.streaming import StreamingSFTDataset


class _BatchTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        self.calls.append(texts)
        token_lists = [[ord(character) % 31 + 1 for character in text] for text in texts]
        longest = max(len(tokens) for tokens in token_lists)
        input_ids = torch.zeros((len(texts), longest), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, tokens in enumerate(token_lists):
            input_ids[row, : len(tokens)] = torch.tensor(tokens)
            attention_mask[row, : len(tokens)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_streaming_batches_tokenize_lazily_and_pad_per_batch() -> None:
    source = iter(
        [
            {"text": "a"},
            {"text": "long"},
            {"text": "xy"},
            {"text": "three"},
            {"text": "z"},
        ]
    )
    tokenizer = _BatchTokenizer()
    dataset = StreamingSFTDataset(source, tokenizer, batch_size=2)

    batches = list(dataset)

    assert [len(call) for call in tokenizer.calls] == [2, 2, 1]
    assert [batch["input_ids"].shape for batch in batches] == [(2, 4), (2, 5), (1, 1)]
    assert batches[0]["attention_mask"].tolist() == [[1, 0, 0, 0], [1, 1, 1, 1]]
    assert batches[0]["labels"].tolist()[0] == [ord("a") % 31 + 1, -100, -100, -100]
