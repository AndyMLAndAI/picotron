"""YAML convenience configuration for factory-based DPO scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from picotron.config.config import PicotronConfig, load_config


@dataclass(frozen=True, slots=True)
class DPOConfig:
    """DPO inputs layered over a Picotron base configuration."""

    base_config: PicotronConfig
    dataset_path: Path
    base_checkpoint_path: Path | None = None
    beta: float = 0.1
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided.")


def load_dpo_config(path: str | Path) -> DPOConfig:
    """Load DPO settings while delegating base-model validation to Picotron."""

    config_path = Path(path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Could not read DPO configuration '{config_path}': {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid DPO YAML in '{config_path}': {error}") from error
    if not isinstance(raw_config, Mapping):
        raise ValueError("DPO configuration must contain a top-level mapping.")

    required = {"base_config_path", "dataset_path"}
    optional = {"base_checkpoint_path", "beta", "max_steps"}
    missing = required - set(raw_config)
    unexpected = set(raw_config) - required - optional
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required fields: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(sorted(unexpected))}")
        raise ValueError("Invalid DPO configuration: " + "; ".join(details) + ".")

    return DPOConfig(
        base_config=load_config(_resolve_path(config_path, raw_config["base_config_path"])),
        dataset_path=_resolve_path(config_path, raw_config["dataset_path"]),
        base_checkpoint_path=(
            _resolve_path(config_path, raw_config["base_checkpoint_path"])
            if "base_checkpoint_path" in raw_config
            else None
        ),
        beta=raw_config.get("beta", 0.1),
        max_steps=raw_config.get("max_steps"),
    )


def _resolve_path(config_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("DPO path fields must be non-empty strings.")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else config_path.parent / candidate
