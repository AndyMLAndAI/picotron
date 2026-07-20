"""Stream, cache, and tokenize a Hugging Face dataset into uint16 token data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from tqdm.auto import tqdm


_WORKER_TOKENIZER: Any | None = None
_UINT16_MAX = np.iinfo(np.uint16).max


def preprocess_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None,
    tokenizer_name: str,
    target_tokens: int,
    output_path: str | Path,
    text_field: str = "text",
    tokenize_workers: int | None = None,
    compression: str | None = None,
    text_cache_dir: str | Path | None = None,
) -> bool:
    """Create a deterministic uint16 token cache from a streamed dataset.

    This tool deliberately preprocesses one source per invocation; configure
    multiple resulting caches under ``data.datasets`` to mix them at training
    time. Tokenization is batched across an ordered process pool. Raw dataset text is
    cached as JSONL, allowing a repeat preprocessing run to avoid re-fetching
    already-consumed examples.  ``compression='gzip'`` writes a gzip-compressed
    token cache; uncompressed uint16 remains the default.

    Returns ``True`` when an already-complete output was skipped.
    """

    _validate_arguments(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        target_tokens=target_tokens,
        text_field=text_field,
        compression=compression,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    compression_name = compression or "none"
    progress_path = output.with_suffix(output.suffix + ".progress.json")
    raw_output = _raw_output_path(output, compression_name)

    if _is_complete_output(output, progress_path, target_tokens, compression_name):
        print(f"Skipping complete token file: {output}")
        return True

    written, resume_tokens = _load_progress(progress_path, target_tokens)
    if written and not raw_output.exists():
        raise ValueError("A resumable progress file exists but the raw token file is missing.")
    expected_bytes = target_tokens * np.dtype(np.uint16).itemsize
    if written and raw_output.stat().st_size != expected_bytes:
        raise ValueError("A resumable progress file exists but the token file size is invalid.")

    mode = "r+" if written else "w+"
    token_memmap = np.memmap(raw_output, mode=mode, dtype=np.uint16, shape=(target_tokens,))
    cache_path = _text_cache_path(
        output.parent if text_cache_dir is None else Path(text_cache_dir),
        dataset_name,
        dataset_config,
        text_field,
    )
    worker_count = _resolve_worker_count(tokenize_workers)
    tokenizer = _load_tokenizer(tokenizer_name)
    texts = _iter_cached_and_streamed_texts(
        cache_path,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        text_field=text_field,
    )

    progress_bar = tqdm(
        total=target_tokens,
        initial=written,
        desc=f"tokenizing {dataset_name}",
        unit="token",
        unit_scale=True,
    )
    try:
        for token_ids in _tokenize_in_order(texts, tokenizer, worker_count):
            if resume_tokens:
                if len(token_ids) <= resume_tokens:
                    resume_tokens -= len(token_ids)
                    continue
                token_ids = token_ids[resume_tokens:]
                resume_tokens = 0
            if not token_ids:
                continue
            _validate_token_ids(token_ids)
            count = min(len(token_ids), target_tokens - written)
            token_memmap[written : written + count] = np.asarray(
                token_ids[:count], dtype=np.uint16
            )
            written += count
            progress_bar.update(count)
            _save_progress(progress_path, written, target_tokens)
            if written >= target_tokens:
                break
    finally:
        progress_bar.close()
        token_memmap.flush()
        del token_memmap

    if written < target_tokens:
        raise ValueError(f"Dataset ended after {written} tokens; {target_tokens} were requested.")

    if compression_name == "gzip":
        _compress_gzip(raw_output, output)
        raw_output.unlink()
    progress_path.unlink(missing_ok=True)
    print(f"Wrote {written} uint16 tokens to {output}")
    return False


def _validate_arguments(
    *,
    dataset_name: str,
    tokenizer_name: str,
    target_tokens: int,
    text_field: str,
    compression: str | None,
) -> None:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")
    if not dataset_name or not tokenizer_name or not text_field:
        raise ValueError("dataset_name, tokenizer_name, and text_field are required.")
    if compression not in (None, "gzip"):
        raise ValueError("compression must be None or 'gzip'.")


def _is_complete_output(
    output: Path,
    progress_path: Path,
    target_tokens: int,
    compression: str,
) -> bool:
    if not output.exists() or progress_path.exists():
        return False
    if compression == "gzip":
        return True
    return output.stat().st_size == target_tokens * np.dtype(np.uint16).itemsize


def _raw_output_path(output: Path, compression: str) -> Path:
    return output if compression == "none" else output.with_name(output.name + ".incomplete.uint16")


def _resolve_worker_count(requested_workers: int | None) -> int:
    if requested_workers is not None:
        if requested_workers <= 0:
            raise ValueError("tokenize_workers must be positive when provided.")
        return requested_workers
    return min(os.cpu_count() or 1, 8)


def _tokenize_in_order(
    texts: Iterable[str], tokenizer: Any, worker_count: int
) -> Iterator[list[int]]:
    """Yield independently tokenized documents in their original input order."""

    if worker_count == 1:
        for text in texts:
            yield _encode_text(tokenizer, text)
        return

    context = mp.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=_initialize_worker_tokenizer,
        initargs=(tokenizer,),
    ) as pool:
        # imap, unlike imap_unordered, preserves the source-document order.
        for token_ids in pool.imap(_tokenize_worker, texts, chunksize=64):
            yield token_ids


def _initialize_worker_tokenizer(tokenizer: Any) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _tokenize_worker(text: str) -> list[int]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker was not initialized.")
    return _encode_text(_WORKER_TOKENIZER, text)


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _validate_token_ids(token_ids: Sequence[int]) -> None:
    if token_ids and (max(token_ids) > _UINT16_MAX or min(token_ids) < 0):
        raise ValueError("Tokenizer produced token ids outside uint16 range.")


def _text_cache_path(
    cache_dir: Path, dataset_name: str, dataset_config: str | None, text_field: str
) -> Path:
    identity = json.dumps(
        {"dataset_name": dataset_name, "dataset_config": dataset_config, "text_field": text_field},
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"raw_text_{digest}.jsonl"


def _iter_cached_and_streamed_texts(
    cache_path: Path,
    *,
    dataset_name: str,
    dataset_config: str | None,
    text_field: str,
) -> Iterator[str]:
    cached_count = 0
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as cache_file:
            for line in cache_file:
                payload = json.loads(line)
                text = payload.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"Invalid cached text shard: {cache_path}")
                cached_count += 1
                yield text

    stream = _load_streaming_dataset(dataset_name, dataset_config)
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for index, example in enumerate(stream):
            if index < cached_count:
                continue
            text = example.get(text_field) if isinstance(example, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"Dataset examples must contain string field '{text_field}'.")
            cache_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            cache_file.flush()
            yield text


def _compress_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as raw_file, gzip.open(destination, "wb", compresslevel=6) as compressed_file:
        shutil.copyfileobj(raw_file, compressed_file, length=1024 * 1024)


def _load_progress(path: Path, target_tokens: int) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("target_tokens") != target_tokens:
        raise ValueError("Progress sidecar target_tokens does not match this invocation.")
    written = payload.get("written")
    if not isinstance(written, int) or not 0 <= written <= target_tokens:
        raise ValueError("Progress sidecar contains an invalid written token count.")
    return written, written


def _save_progress(path: Path, written: int, target_tokens: int) -> None:
    path.write_text(
        json.dumps({"written": written, "target_tokens": target_tokens}),
        encoding="utf-8",
    )


def _load_tokenizer(tokenizer_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _load_streaming_dataset(dataset_name: str, dataset_config: str | None) -> Iterable[Any]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": "train", "streaming": True}
    if dataset_config is not None:
        kwargs["name"] = dataset_config
    return load_dataset(dataset_name, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the preprocessing CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--target-tokens", required=True, type=int)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        default=None,
        help="Tokenization worker processes (default: up to 8 available CPU cores).",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip",),
        default=None,
        help="Optional on-disk compression; raw uint16 is the default.",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=None,
        help="Directory for reusable raw streamed-text JSONL shards.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    preprocess_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer_name,
        target_tokens=args.target_tokens,
        output_path=args.output_path,
        text_field=args.text_field,
        tokenize_workers=args.tokenize_workers,
        compression=args.compression,
        text_cache_dir=args.text_cache_dir,
    )


if __name__ == "__main__":
    main()
