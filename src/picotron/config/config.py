"""Schema and YAML loader for Picotron configuration."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigValidationError(ValueError):
    """Raised when a Picotron configuration is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class MoEConfig:
    """Optional top-k MoE routing configuration."""

    num_experts: int
    top_k: int = 2
    aux_loss_coefficient: float = 0.01

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_experts, bool)
            or not isinstance(self.num_experts, int)
            or self.num_experts <= 0
        ):
            raise ConfigValidationError("'num_experts' must be a positive integer.")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or not 1 <= self.top_k <= self.num_experts
        ):
            raise ConfigValidationError("'top_k' must be between 1 and num_experts.")
        if (
            isinstance(self.aux_loss_coefficient, bool)
            or not isinstance(self.aux_loss_coefficient, (int, float))
            or self.aux_loss_coefficient < 0
        ):
            raise ConfigValidationError("'aux_loss_coefficient' must be non-negative.")


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
    num_key_value_heads: int | None = None
    sliding_window_size: int | None = None
    moe_config: MoEConfig | None = None
    nope_layers: tuple[int, ...] = ()
    attention_type: str = "mha"
    kv_lora_rank: int | None = None
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
        if self.num_key_value_heads is not None:
            if (
                isinstance(self.num_key_value_heads, bool)
                or not isinstance(self.num_key_value_heads, int)
                or self.num_key_value_heads <= 0
            ):
                raise ConfigValidationError("'num_key_value_heads' must be a positive integer.")
            if self.num_attention_heads % self.num_key_value_heads != 0:
                raise ConfigValidationError(
                    "'num_attention_heads' must be divisible by 'num_key_value_heads'."
                )
        if self.sliding_window_size is not None and (
            isinstance(self.sliding_window_size, bool)
            or not isinstance(self.sliding_window_size, int)
            or self.sliding_window_size <= 0
        ):
            raise ConfigValidationError("'sliding_window_size' must be a positive integer.")
        if self.attention_type not in ("mha", "gqa", "mla"):
            raise ConfigValidationError("'attention_type' must be one of 'mha', 'gqa', or 'mla'.")
        if self.kv_lora_rank is not None and (
            isinstance(self.kv_lora_rank, bool)
            or not isinstance(self.kv_lora_rank, int)
            or self.kv_lora_rank <= 0
        ):
            raise ConfigValidationError("'kv_lora_rank' must be a positive integer when provided.")
        if self.attention_type == "mla":
            if self.kv_lora_rank is None:
                raise ConfigValidationError("'kv_lora_rank' is required when attention_type is 'mla'.")
            if self.kv_lora_rank >= 2 * self.hidden_size:
                raise ConfigValidationError(
                    "'kv_lora_rank' must be smaller than twice hidden_size for MLA compression."
                )
            if self.num_key_value_heads not in (None, self.num_attention_heads):
                raise ConfigValidationError("MLA cannot be combined with grouped-query attention.")
            if self.sliding_window_size is not None:
                raise ConfigValidationError("MLA cannot be combined with sliding-window attention.")
        elif self.attention_type == "gqa" and self.num_key_value_heads is None:
            raise ConfigValidationError("'num_key_value_heads' is required when attention_type is 'gqa'.")
        if isinstance(self.moe_config, Mapping):
            object.__setattr__(self, "moe_config", MoEConfig(**dict(self.moe_config)))
        if self.moe_config is not None and not isinstance(self.moe_config, MoEConfig):
            raise ConfigValidationError("'moe_config' must be an MoE configuration mapping.")
        if not isinstance(self.nope_layers, (list, tuple)):
            raise ConfigValidationError("'nope_layers' must be a sequence of layer indices.")
        normalized_nope_layers = tuple(self.nope_layers)
        if any(
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.num_hidden_layers
            for layer_index in normalized_nope_layers
        ):
            raise ConfigValidationError(
                "'nope_layers' entries must be valid zero-based decoder layer indices."
            )
        if len(set(normalized_nope_layers)) != len(normalized_nope_layers):
            raise ConfigValidationError("'nope_layers' must not contain duplicate layer indices.")
        object.__setattr__(self, "nope_layers", normalized_nope_layers)


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
