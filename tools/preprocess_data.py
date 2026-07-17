"""Stream and tokenize a Hugging Face dataset into a uint16 token memmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def preprocess_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None,
    tokenizer_name: str,
    target_tokens: int,
    output_path: str | Path,
    text_field: str = "text",
) -> bool:
    """Tokenize a streaming dataset into ``output_path``.

    Returns ``True`` when an already-complete file was skipped. A progress
    sidecar permits rerunning after interruption without rewriting prior tokens.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")
    if not dataset_name or not tokenizer_name or not text_field:
        raise ValueError("dataset_name, tokenizer_name, and text_field are required.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(output.suffix + ".progress.json")
    expected_bytes = target_tokens * np.dtype(np.uint16).itemsize

    if output.exists() and output.stat().st_size == expected_bytes and not progress_path.exists():
        print(f"Skipping complete token file: {output}")
        return True

    written, resume_tokens = _load_progress(progress_path, target_tokens)
    if written and output.stat().st_size != expected_bytes:
        raise ValueError("A resumable progress file exists but the token file size is invalid.")
    mode = "r+" if written else "w+"
    token_memmap = np.memmap(output, mode=mode, dtype=np.uint16, shape=(target_tokens,))
    tokenizer = _load_tokenizer(tokenizer_name)
    stream = _load_streaming_dataset(dataset_name, dataset_config)
    skipped = resume_tokens

    try:
        for example in stream:
            text = example.get(text_field) if isinstance(example, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"Dataset examples must contain string field '{text_field}'.")
            token_ids = list(tokenizer.encode(text, add_special_tokens=False))
            if skipped:
                if len(token_ids) <= skipped:
                    skipped -= len(token_ids)
                    continue
                token_ids = token_ids[skipped:]
                skipped = 0
            if not token_ids:
                continue
            if max(token_ids) > np.iinfo(np.uint16).max or min(token_ids) < 0:
                raise ValueError("Tokenizer produced token ids outside uint16 range.")
            count = min(len(token_ids), target_tokens - written)
            token_memmap[written:written + count] = np.asarray(
                token_ids[:count], dtype=np.uint16
            )
            written += count
            _save_progress(progress_path, written, target_tokens)
            if written >= target_tokens:
                break
    finally:
        token_memmap.flush()

    if written < target_tokens:
        raise ValueError(
            f"Dataset ended after {written} tokens; {target_tokens} were requested."
        )
    progress_path.unlink(missing_ok=True)
    print(f"Wrote {written} uint16 tokens to {output}")
    return False


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
    )


if __name__ == "__main__":
    main()
