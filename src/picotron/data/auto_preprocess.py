"""CLI orchestration for converting configured Hugging Face sources into token caches."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from picotron.config.config import DatasetSourceConfig, PicotronConfig


def materialize_hf_dataset_sources(
    config: PicotronConfig, *, project_root: str | Path
) -> PicotronConfig:
    """Run the existing preprocessing tool for missing configured HF token caches."""

    sources = config.data.dataset_sources
    if not any(source.needs_preprocessing for source in sources):
        return config
    tokenizer_name = config.data.tokenizer_name
    assert tokenizer_name is not None  # Validated by DataConfig.
    tool_path = Path(project_root) / "tools" / "preprocess_data.py"
    if not tool_path.exists():
        raise FileNotFoundError(f"Picotron preprocessing tool not found: {tool_path}")

    resolved_sources: list[DatasetSourceConfig] = []
    for source in sources:
        if not source.needs_preprocessing:
            resolved_sources.append(source)
            continue
        output_path = Path(
            source.cache_path(
                tokenizer_name=tokenizer_name,
                cache_dir=config.data.token_cache_dir,
            )
        )
        if _is_complete_raw_cache(output_path, source.target_tokens):
            print(f"Using cached token file: {output_path}")
        else:
            print(f"Preprocessing {source.hf_name} before training: {output_path}")
            command = [
                sys.executable,
                str(tool_path),
                "--dataset-name",
                source.hf_name or "",
                "--tokenizer-name",
                tokenizer_name,
                "--target-tokens",
                str(source.target_tokens),
                "--output-path",
                str(output_path),
                "--text-field",
                source.text_field,
                "--text-cache-dir",
                str(Path(config.data.token_cache_dir) / "raw_text"),
            ]
            if source.hf_config is not None:
                command.extend(("--dataset-config", source.hf_config))
            environment = os.environ.copy()
            hf_token = config.data.resolve_hf_token()
            if hf_token is not None:
                environment["HF_TOKEN"] = hf_token
            subprocess.run(command, check=True, env=environment)
        resolved_sources.append(DatasetSourceConfig(path=str(output_path), weight=source.weight))

    return replace(config, data=replace(config.data, datasets=tuple(resolved_sources)))


def _is_complete_raw_cache(path: Path, target_tokens: int | None) -> bool:
    """Return whether an uncompressed uint16 cache has the expected byte size."""

    return bool(
        target_tokens is not None
        and path.exists()
        and path.stat().st_size == target_tokens * 2
        and not path.with_suffix(path.suffix + ".progress.json").exists()
    )
