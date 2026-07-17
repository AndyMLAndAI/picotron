"""CPU tests for mocked streaming tokenization and uint16 output."""

import json
from pathlib import Path

import numpy as np

from tools import preprocess_data


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) % 100 for character in text]


def test_preprocess_writes_uint16_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "tokens.uint16"
    monkeypatch.setattr(preprocess_data, "_load_tokenizer", lambda _: _FakeTokenizer())
    monkeypatch.setattr(
        preprocess_data,
        "_load_streaming_dataset",
        lambda *_: [{"text": "abc"}, {"text": "defgh"}],
    )

    skipped = preprocess_data.preprocess_dataset(
        dataset_name="mock",
        dataset_config=None,
        tokenizer_name="mock-tokenizer",
        target_tokens=6,
        output_path=output_path,
    )

    assert skipped is False
    assert output_path.stat().st_size == 6 * np.dtype(np.uint16).itemsize
    tokens = np.memmap(output_path, mode="r", dtype=np.uint16, shape=(6,))
    np.testing.assert_array_equal(tokens, np.array([97, 98, 99, 0, 1, 2], dtype=np.uint16))

    def should_not_load(*_):
        raise AssertionError("complete preprocessing should be skipped")

    monkeypatch.setattr(preprocess_data, "_load_streaming_dataset", should_not_load)
    assert preprocess_data.preprocess_dataset(
        dataset_name="mock",
        dataset_config=None,
        tokenizer_name="mock-tokenizer",
        target_tokens=6,
        output_path=output_path,
    ) is True


def test_preprocess_resumes_from_progress_sidecar(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "resume.uint16"
    progress_path = output_path.with_suffix(".uint16.progress.json")
    partial = np.memmap(output_path, mode="w+", dtype=np.uint16, shape=(6,))
    partial[:2] = [97, 98]
    partial.flush()
    progress_path.write_text(json.dumps({"written": 2, "target_tokens": 6}), encoding="utf-8")
    monkeypatch.setattr(preprocess_data, "_load_tokenizer", lambda _: _FakeTokenizer())
    monkeypatch.setattr(
        preprocess_data,
        "_load_streaming_dataset",
        lambda *_: [{"text": "abc"}, {"text": "defgh"}],
    )

    preprocess_data.preprocess_dataset(
        dataset_name="mock",
        dataset_config=None,
        tokenizer_name="mock-tokenizer",
        target_tokens=6,
        output_path=output_path,
    )

    tokens = np.memmap(output_path, mode="r", dtype=np.uint16, shape=(6,))
    np.testing.assert_array_equal(tokens, np.array([97, 98, 99, 0, 1, 2], dtype=np.uint16))
    assert not progress_path.exists()
