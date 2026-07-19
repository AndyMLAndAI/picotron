"""CPU tests for deterministic, cached, and compressed preprocessing."""

import gzip
import json
from pathlib import Path

import numpy as np

from tools import preprocess_data


class _FakeTokenizer:
    """Pickle-safe tokenizer used by spawned multiprocessing workers."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) % 100 for character in text]


def _install_fake_sources(monkeypatch, examples: list[dict[str, str]]) -> None:
    monkeypatch.setattr(preprocess_data, "_load_tokenizer", lambda _: _FakeTokenizer())
    monkeypatch.setattr(preprocess_data, "_load_streaming_dataset", lambda *_: iter(examples))


def test_preprocess_writes_uint16_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "tokens.uint16"
    _install_fake_sources(monkeypatch, [{"text": "abc"}, {"text": "defgh"}])

    skipped = preprocess_data.preprocess_dataset(
        dataset_name="mock",
        dataset_config=None,
        tokenizer_name="mock-tokenizer",
        target_tokens=6,
        output_path=output_path,
        tokenize_workers=1,
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
        tokenize_workers=1,
    ) is True


def test_preprocess_resumes_from_progress_sidecar(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "resume.uint16"
    progress_path = output_path.with_suffix(".uint16.progress.json")
    partial = np.memmap(output_path, mode="w+", dtype=np.uint16, shape=(6,))
    partial[:2] = [97, 98]
    partial.flush()
    progress_path.write_text(json.dumps({"written": 2, "target_tokens": 6}), encoding="utf-8")
    _install_fake_sources(monkeypatch, [{"text": "abc"}, {"text": "defgh"}])

    preprocess_data.preprocess_dataset(
        dataset_name="mock",
        dataset_config=None,
        tokenizer_name="mock-tokenizer",
        target_tokens=6,
        output_path=output_path,
        tokenize_workers=1,
    )

    tokens = np.memmap(output_path, mode="r", dtype=np.uint16, shape=(6,))
    np.testing.assert_array_equal(tokens, np.array([97, 98, 99, 0, 1, 2], dtype=np.uint16))
    assert not progress_path.exists()


def test_multiprocess_tokenization_matches_single_process(tmp_path: Path, monkeypatch) -> None:
    examples = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}] * 32
    _install_fake_sources(monkeypatch, examples)
    single_output = tmp_path / "single.uint16"
    parallel_output = tmp_path / "parallel.uint16"
    target_tokens = 256

    preprocess_data.preprocess_dataset(
        dataset_name="mock",
        dataset_config=None,
        tokenizer_name="mock-tokenizer",
        target_tokens=target_tokens,
        output_path=single_output,
        tokenize_workers=1,
        text_cache_dir=tmp_path / "single-cache",
    )
    preprocess_data.preprocess_dataset(
        dataset_name="mock",
        dataset_config=None,
        tokenizer_name="mock-tokenizer",
        target_tokens=target_tokens,
        output_path=parallel_output,
        tokenize_workers=2,
        text_cache_dir=tmp_path / "parallel-cache",
    )

    assert single_output.read_bytes() == parallel_output.read_bytes()


def test_gzip_output_round_trip_is_byte_identical(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "tokens.uint16.gz"
    _install_fake_sources(monkeypatch, [{"text": "a" * 2048}])

    preprocess_data.preprocess_dataset(
        dataset_name="mock",
        dataset_config=None,
        tokenizer_name="mock-tokenizer",
        target_tokens=1024,
        output_path=output_path,
        tokenize_workers=1,
        compression="gzip",
    )

    expected = np.full(1024, ord("a") % 100, dtype=np.uint16).tobytes()
    with gzip.open(output_path, "rb") as compressed_file:
        assert compressed_file.read() == expected
    assert output_path.stat().st_size < len(expected)


def test_text_cache_avoids_refetch_for_second_output(tmp_path: Path, monkeypatch) -> None:
    examples = [{"text": "abc"}, {"text": "defgh"}]
    fetches = 0

    def load_stream(*_):
        nonlocal fetches
        fetches += 1
        return iter(examples)

    monkeypatch.setattr(preprocess_data, "_load_tokenizer", lambda _: _FakeTokenizer())
    monkeypatch.setattr(preprocess_data, "_load_streaming_dataset", load_stream)
    cache_dir = tmp_path / "raw-text-cache"
    common = {
        "dataset_name": "mock",
        "dataset_config": "config",
        "tokenizer_name": "mock-tokenizer",
        "target_tokens": 6,
        "tokenize_workers": 1,
        "text_cache_dir": cache_dir,
    }

    preprocess_data.preprocess_dataset(output_path=tmp_path / "first.uint16", **common)
    assert fetches == 1

    def should_not_refetch(*_):
        raise AssertionError("cached text should satisfy the second preprocessing run")

    monkeypatch.setattr(preprocess_data, "_load_streaming_dataset", should_not_refetch)
    preprocess_data.preprocess_dataset(output_path=tmp_path / "second.uint16", **common)
    assert (tmp_path / "first.uint16").read_bytes() == (tmp_path / "second.uint16").read_bytes()
