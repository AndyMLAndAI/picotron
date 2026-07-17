"""SFT configuration that references, rather than copies, Picotron config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from picotron.config.config import PicotronConfig, load_config


@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Fine-tuning inputs layered on top of a loaded Picotron config."""

    base_config: PicotronConfig
    base_checkpoint_path: Path
    dataset_path: Path
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided.")


def load_sft_config(path: str | Path) -> SFTConfig:
    """Load SFT-specific settings and the referenced Picotron base config."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except OSError as error:
        raise ValueError(f"Could not read SFT configuration '{config_path}': {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid SFT YAML in '{config_path}': {error}") from error
    if not isinstance(raw_config, Mapping):
        raise ValueError("SFT configuration must contain a top-level mapping.")

    required = {"base_config_path", "base_checkpoint_path", "dataset_path"}
    optional = {"max_steps"}
    supplied = set(raw_config)
    missing = required - supplied
    unexpected = supplied - required - optional
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing required fields: {', '.join(sorted(missing))}")
        if unexpected:
            parts.append(f"unexpected fields: {', '.join(sorted(unexpected))}")
        raise ValueError("Invalid SFT configuration: " + "; ".join(parts) + ".")

    base_config_path = _resolve_path(config_path, raw_config["base_config_path"])
    return SFTConfig(
        base_config=load_config(base_config_path),
        base_checkpoint_path=_resolve_path(config_path, raw_config["base_checkpoint_path"]),
        dataset_path=_resolve_path(config_path, raw_config["dataset_path"]),
        max_steps=raw_config.get("max_steps"),
    )


def _resolve_path(config_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("SFT path fields must be non-empty strings.")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else config_path.parent / candidate

