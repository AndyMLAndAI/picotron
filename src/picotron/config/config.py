"""Schema and YAML loader for Picotron configuration."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigValidationError(ValueError):
    """Raised when a Picotron configuration is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class PicotronConfig:
    """Required model and future training settings for a Picotron run."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_seq_len: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    checkpoint_interval: int
    zero_stage: int = 0
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_seq_len",
            "batch_size",
            "num_epochs",
            "checkpoint_interval",
        )
        for field_name in positive_integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigValidationError(
                    f"'{field_name}' must be a positive integer; got {value!r}."
                )

        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            raise ConfigValidationError(
                f"'learning_rate' must be a positive number; got {self.learning_rate!r}."
            )

        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigValidationError(
                "'hidden_size' must be divisible by 'num_attention_heads'."
            )
        if not isinstance(self.model_kwargs, dict):
            raise ConfigValidationError("'model_kwargs' must be a mapping.")
        if self.zero_stage not in (0, 1, 2):
            raise ConfigValidationError("'zero_stage' must be one of 0, 1, or 2.")


def load_config(path: str | Path) -> PicotronConfig:
    """Load and strictly validate a Picotron YAML configuration file."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except OSError as error:
        raise ConfigValidationError(
            f"Could not read configuration file '{config_path}': {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigValidationError(
            f"Invalid YAML in configuration file '{config_path}': {error}"
        ) from error

    if not isinstance(raw_config, Mapping):
        raise ConfigValidationError(
            "Configuration must contain a top-level YAML mapping of settings."
        )

    return _build_config(raw_config)


def _build_config(raw_config: Mapping[str, Any]) -> PicotronConfig:
    config_fields = fields(PicotronConfig)
    expected_fields = {config_field.name for config_field in config_fields}
    required_fields = {
        config_field.name
        for config_field in config_fields
        if config_field.default is MISSING and config_field.default_factory is MISSING
    }
    supplied_fields = set(raw_config)
    missing_fields = sorted(required_fields - supplied_fields)
    unexpected_fields = sorted(supplied_fields - expected_fields)

    errors: list[str] = []
    if missing_fields:
        errors.append(f"missing required fields: {', '.join(missing_fields)}")
    if unexpected_fields:
        errors.append(f"unexpected fields: {', '.join(unexpected_fields)}")
    if errors:
        raise ConfigValidationError("Invalid configuration: " + "; ".join(errors) + ".")

    try:
        return PicotronConfig(**dict(raw_config))
    except TypeError as error:
        raise ConfigValidationError(f"Invalid configuration types: {error}") from error
