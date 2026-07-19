"""Strict, nested configuration schema for Picotron runs."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields as dataclass_fields
from math import cos, pi
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from picotron.utils.hardware import get_gpu_compute_capability, select_training_dtype


class ConfigValidationError(ValueError):
    """Raised when a Picotron configuration is incomplete or invalid."""


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigValidationError(f"'{name}' must be a positive integer; got {value!r}.")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigValidationError(f"'{name}' must be a non-negative integer; got {value!r}.")


def _require_positive_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigValidationError(f"'{name}' must be a positive number; got {value!r}.")


def _require_nonnegative_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigValidationError(f"'{name}' must be a non-negative number; got {value!r}.")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigValidationError(f"'{name}' must be a boolean; got {value!r}.")


def _require_optional_path(name: str, value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ConfigValidationError(f"'{name}' must be a non-empty string when provided.")


@dataclass(frozen=True, slots=True)
class MoEConfig:
    """Optional top-k MoE routing configuration."""

    num_experts: int
    top_k: int = 2
    aux_loss_coefficient: float = 0.01

    def __post_init__(self) -> None:
        _require_positive_int("num_experts", self.num_experts)
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or not 1 <= self.top_k <= self.num_experts
        ):
            raise ConfigValidationError("'top_k' must be between 1 and num_experts.")
        _require_nonnegative_number("aux_loss_coefficient", self.aux_loss_coefficient)


@dataclass(frozen=True, slots=True)
class TritonKernelsConfig:
    """Explicit opt-ins for optional Triton kernels."""

    rmsnorm: bool = False
    swiglu: bool = False
    rope: bool = False
    attention: bool = False
    cross_entropy: bool = False
    adamw: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "rmsnorm",
            "swiglu",
            "rope",
            "attention",
            "cross_entropy",
            "adamw",
        ):
            _require_bool(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture settings consumed by the bundled decoder model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    attention_type: str = "mha"
    nope_layers: tuple[int, ...] = ()
    rope_theta: float = 10_000.0
    sliding_window_size: int | None = None
    moe_config: MoEConfig | None = None
    kv_lora_rank: int | None = None
    tie_word_embeddings: bool = False
    position_embedding_type: str = "rope"
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigValidationError(
                "'hidden_size' must be divisible by 'num_attention_heads'."
            )
        if self.num_key_value_heads is not None:
            _require_positive_int("num_key_value_heads", self.num_key_value_heads)
            if self.num_attention_heads % self.num_key_value_heads != 0:
                raise ConfigValidationError(
                    "'num_attention_heads' must be divisible by 'num_key_value_heads'."
                )
        if self.attention_type not in ("mha", "gqa", "mla"):
            raise ConfigValidationError(
                "'attention_type' must be one of 'mha', 'gqa', or 'mla'."
            )
        if self.attention_type == "mha" and self.num_key_value_heads not in (
            None,
            self.num_attention_heads,
        ):
            raise ConfigValidationError(
                "'attention_type: mha' requires num_key_value_heads to be unset "
                "or equal to num_attention_heads."
            )
        if self.attention_type == "gqa":
            if self.num_key_value_heads is None:
                raise ConfigValidationError(
                    "'num_key_value_heads' is required when attention_type is 'gqa'."
                )
            if self.num_key_value_heads >= self.num_attention_heads:
                raise ConfigValidationError(
                    "'attention_type: gqa' requires num_key_value_heads to be "
                    "smaller than num_attention_heads."
                )
        if self.sliding_window_size is not None:
            _require_positive_int("sliding_window_size", self.sliding_window_size)
        _require_positive_number("rope_theta", self.rope_theta)
        if self.position_embedding_type not in ("rope", "learned"):
            raise ConfigValidationError(
                "'position_embedding_type' must be either 'rope' or 'learned'."
            )
        _require_bool("tie_word_embeddings", self.tie_word_embeddings)

        if self.kv_lora_rank is not None:
            _require_positive_int("kv_lora_rank", self.kv_lora_rank)
        if self.attention_type == "mla":
            if self.kv_lora_rank is None:
                raise ConfigValidationError(
                    "'kv_lora_rank' is required when attention_type is 'mla'."
                )
            if self.kv_lora_rank >= 2 * self.hidden_size:
                raise ConfigValidationError(
                    "'kv_lora_rank' must be smaller than twice hidden_size for MLA compression."
                )
            if self.num_key_value_heads not in (None, self.num_attention_heads):
                raise ConfigValidationError("MLA cannot be combined with grouped-query attention.")
            if self.sliding_window_size is not None:
                raise ConfigValidationError("MLA cannot be combined with sliding-window attention.")

        if isinstance(self.moe_config, Mapping):
            object.__setattr__(self, "moe_config", _build_dataclass(self.moe_config, MoEConfig, "moe_config"))
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

        if not isinstance(self.model_kwargs, Mapping):
            raise ConfigValidationError("'model_kwargs' must be a mapping.")
        reserved_keys = {
            "num_key_value_heads",
            "attention_type",
            "nope_layers",
            "rope_theta",
            "sliding_window_size",
            "moe_config",
            "kv_lora_rank",
            "tie_word_embeddings",
            "position_embedding_type",
            "use_triton_rmsnorm",
            "use_triton_swiglu",
            "use_triton_rope",
            "use_triton_attention",
            "use_triton_cross_entropy",
            "use_triton_adamw",
        }
        conflicting_keys = sorted(reserved_keys.intersection(self.model_kwargs))
        if conflicting_keys:
            raise ConfigValidationError(
                "'model_kwargs' must not override typed settings: "
                + ", ".join(conflicting_keys)
                + "."
            )
        object.__setattr__(self, "model_kwargs", dict(self.model_kwargs))


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Precision policy and model architecture configuration."""

    model_config: ModelConfig
    dtype: str = "auto"
    triton_kernels: TritonKernelsConfig = field(default_factory=TritonKernelsConfig)

    def __post_init__(self) -> None:
        if isinstance(self.model_config, Mapping):
            object.__setattr__(
                self,
                "model_config",
                _build_dataclass(self.model_config, ModelConfig, "model.model_config"),
            )
        if not isinstance(self.model_config, ModelConfig):
            raise ConfigValidationError("'model_config' must be a model configuration mapping.")
        if isinstance(self.triton_kernels, Mapping):
            object.__setattr__(
                self,
                "triton_kernels",
                _build_dataclass(
                    self.triton_kernels, TritonKernelsConfig, "model.triton_kernels"
                ),
            )
        if not isinstance(self.triton_kernels, TritonKernelsConfig):
            raise ConfigValidationError("'triton_kernels' must be a Triton configuration mapping.")
        if self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ConfigValidationError(
                "'dtype' must be one of 'auto', 'float32', 'float16', or 'bfloat16'."
            )

    def resolve_dtype(self, device: int | torch.device | None = None) -> torch.dtype:
        """Resolve the configured dtype while enforcing the Turing bf16 rule."""

        if self.dtype == "auto":
            return select_training_dtype(device)
        capability = get_gpu_compute_capability(device)
        if self.dtype == "bfloat16" and capability is not None and capability[0] < 8:
            raise ConfigValidationError(
                "bfloat16 is unsupported on pre-Ampere GPUs; use float16 on Turing."
            )
        return getattr(torch, self.dtype)


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint cadence and safetensors checkpoint-path policy."""

    checkpoint_interval: int
    checkpoints_path: str | None = None
    resume_checkpoint_path: str | None = None
    load_optimizer: bool = True
    load_lr_scheduler: bool = False
    save_final_state: bool = True

    def __post_init__(self) -> None:
        _require_positive_int("checkpoint_interval", self.checkpoint_interval)
        _require_optional_path("checkpoints_path", self.checkpoints_path)
        _require_optional_path("resume_checkpoint_path", self.resume_checkpoint_path)
        _require_bool("load_optimizer", self.load_optimizer)
        _require_bool("load_lr_scheduler", self.load_lr_scheduler)
        _require_bool("save_final_state", self.save_final_state)
        if self.load_lr_scheduler:
            raise ConfigValidationError(
                "'load_lr_scheduler' cannot be true because scheduler state is not checkpointed yet."
            )


@dataclass(frozen=True, slots=True)
class LearningRateSchedulerConfig:
    """Learning-rate schedule settings used by the training loop."""

    learning_rate: float
    lr_decay_steps: int | None = None
    lr_decay_style: str = "constant"
    lr_warmup_steps: int = 0
    lr_warmup_style: str = "linear"
    min_decay_lr: float | None = None

    def __post_init__(self) -> None:
        _require_positive_number("learning_rate", self.learning_rate)
        if self.lr_decay_steps is not None:
            _require_positive_int("lr_decay_steps", self.lr_decay_steps)
        _require_nonnegative_int("lr_warmup_steps", self.lr_warmup_steps)
        if self.lr_decay_style not in ("constant", "linear", "cosine"):
            raise ConfigValidationError(
                "'lr_decay_style' must be one of 'constant', 'linear', or 'cosine'."
            )
        if self.lr_warmup_style not in ("linear", "constant"):
            raise ConfigValidationError(
                "'lr_warmup_style' must be either 'linear' or 'constant'."
            )
        if self.lr_decay_style != "constant" and self.lr_decay_steps is None:
            raise ConfigValidationError(
                "'lr_decay_steps' is required for a non-constant learning-rate decay style."
            )
        if self.min_decay_lr is not None:
            _require_nonnegative_number("min_decay_lr", self.min_decay_lr)
            if self.min_decay_lr > self.learning_rate:
                raise ConfigValidationError(
                    "'min_decay_lr' must not exceed 'learning_rate'."
                )

    def learning_rate_at(self, step: int) -> float:
        """Return the scheduled learning rate for zero-based optimizer ``step``."""

        _require_nonnegative_int("step", step)
        if self.lr_warmup_steps and step < self.lr_warmup_steps:
            if self.lr_warmup_style == "linear":
                return float(self.learning_rate * (step + 1) / self.lr_warmup_steps)
            return float(self.learning_rate)
        if self.lr_decay_style == "constant" or self.lr_decay_steps is None:
            return float(self.learning_rate)
        decay_step = max(0, step - self.lr_warmup_steps)
        progress = min(decay_step / self.lr_decay_steps, 1.0)
        minimum = 0.0 if self.min_decay_lr is None else float(self.min_decay_lr)
        if self.lr_decay_style == "linear":
            return float(self.learning_rate + (minimum - self.learning_rate) * progress)
        cosine = 0.5 * (1.0 + cos(pi * progress))
        return float(minimum + (self.learning_rate - minimum) * cosine)


@dataclass(frozen=True, slots=True)
class OptimizerFactoryConfig:
    """Supported AdamW construction settings."""

    name: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ConfigValidationError("Only the implemented 'adamw' optimizer is supported.")
        for field_name in ("adam_beta1", "adam_beta2"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ConfigValidationError(f"'{field_name}' must be in the range [0, 1).")
        _require_positive_number("adam_eps", self.adam_eps)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Optimizer and scheduling settings for a Picotron run."""

    learning_rate_scheduler: LearningRateSchedulerConfig
    optimizer_factory: OptimizerFactoryConfig = field(default_factory=OptimizerFactoryConfig)
    weight_decay: float = 0.0
    clip_grad: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.learning_rate_scheduler, Mapping):
            object.__setattr__(
                self,
                "learning_rate_scheduler",
                _build_dataclass(
                    self.learning_rate_scheduler,
                    LearningRateSchedulerConfig,
                    "optimizer.learning_rate_scheduler",
                ),
            )
        if not isinstance(self.learning_rate_scheduler, LearningRateSchedulerConfig):
            raise ConfigValidationError(
                "'learning_rate_scheduler' must be a scheduler configuration mapping."
            )
        if isinstance(self.optimizer_factory, Mapping):
            object.__setattr__(
                self,
                "optimizer_factory",
                _build_dataclass(
                    self.optimizer_factory,
                    OptimizerFactoryConfig,
                    "optimizer.optimizer_factory",
                ),
            )
        if not isinstance(self.optimizer_factory, OptimizerFactoryConfig):
            raise ConfigValidationError(
                "'optimizer_factory' must be an optimizer configuration mapping."
            )
        _require_nonnegative_number("weight_decay", self.weight_decay)
        if self.clip_grad is not None:
            _require_positive_number("clip_grad", self.clip_grad)


@dataclass(frozen=True, slots=True)
class ParallelismConfig:
    """Data-parallel and ZeRO settings implemented by Picotron."""

    dp: int = 1
    zero_stage: int = 0

    def __post_init__(self) -> None:
        _require_positive_int("dp", self.dp)
        if self.zero_stage not in (0, 1, 2):
            raise ConfigValidationError("'zero_stage' must be one of 0, 1, or 2.")


@dataclass(frozen=True, slots=True)
class TokensConfig:
    """Sequence, micro-batch, and total-step settings."""

    sequence_length: int
    micro_batch_size: int
    train_steps: int

    def __post_init__(self) -> None:
        for field_name in ("sequence_length", "micro_batch_size", "train_steps"):
            _require_positive_int(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Optional preprocessed-data and tokenizer metadata."""

    dataset_token_path: str | None = None
    tokenizer_name: str | None = None
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        _require_optional_path("dataset_token_path", self.dataset_token_path)
        _require_optional_path("tokenizer_name", self.tokenizer_name)
        if self.vocab_size is not None:
            _require_positive_int("data.vocab_size", self.vocab_size)


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Settings consumed by the training display and standard logging."""

    log_level: str = "INFO"
    iteration_step_info_interval: int = 1
    file_logging: bool = True
    file_logging_output_dir: str = "logs"

    def __post_init__(self) -> None:
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigValidationError(
                "'log_level' must be one of DEBUG, INFO, WARNING, ERROR, or CRITICAL."
            )
        _require_positive_int(
            "iteration_step_info_interval", self.iteration_step_info_interval
        )
        _require_bool("file_logging", self.file_logging)
        _require_optional_path("file_logging_output_dir", self.file_logging_output_dir)


@dataclass(frozen=True, slots=True)
class GeneralConfig:
    """Run-identifying metadata and deterministic seed."""

    project: str = "picotron"
    run: str = "default"
    seed: int = 1337

    def __post_init__(self) -> None:
        for field_name in ("project", "run"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError(f"'{field_name}' must be a non-empty string.")
        _require_nonnegative_int("seed", self.seed)


@dataclass(frozen=True, slots=True)
class PicotronConfig:
    """Complete strict configuration for one Picotron training run."""

    checkpoints: CheckpointConfig
    model: ModelSettings
    optimizer: OptimizerConfig
    parallelism: ParallelismConfig
    tokens: TokensConfig
    data: DataConfig
    logging: LoggingConfig
    general: GeneralConfig

    def __post_init__(self) -> None:
        nested_sections: tuple[tuple[str, type[Any]], ...] = (
            ("checkpoints", CheckpointConfig),
            ("model", ModelSettings),
            ("optimizer", OptimizerConfig),
            ("parallelism", ParallelismConfig),
            ("tokens", TokensConfig),
            ("data", DataConfig),
            ("logging", LoggingConfig),
            ("general", GeneralConfig),
        )
        for field_name, field_type in nested_sections:
            value = getattr(self, field_name)
            if isinstance(value, Mapping):
                object.__setattr__(
                    self,
                    field_name,
                    _build_dataclass(value, field_type, field_name),
                )
            elif not isinstance(value, field_type):
                raise ConfigValidationError(f"'{field_name}' must be a configuration mapping.")
        if self.data.vocab_size is not None and (
            self.data.vocab_size != self.model.model_config.vocab_size
        ):
            raise ConfigValidationError(
                "'data.vocab_size' must match 'model.model_config.vocab_size'."
            )


_NESTED_FIELD_TYPES: dict[type[Any], dict[str, type[Any]]] = {
    PicotronConfig: {
        "checkpoints": CheckpointConfig,
        "model": ModelSettings,
        "optimizer": OptimizerConfig,
        "parallelism": ParallelismConfig,
        "tokens": TokensConfig,
        "data": DataConfig,
        "logging": LoggingConfig,
        "general": GeneralConfig,
    },
    ModelSettings: {
        "model_config": ModelConfig,
        "triton_kernels": TritonKernelsConfig,
    },
    ModelConfig: {"moe_config": MoEConfig},
    OptimizerConfig: {
        "learning_rate_scheduler": LearningRateSchedulerConfig,
        "optimizer_factory": OptimizerFactoryConfig,
    },
}


def load_config(path: str | Path) -> PicotronConfig:
    """Load a strictly validated nested Picotron YAML configuration file."""

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
    return _build_dataclass(raw_config, PicotronConfig, "configuration")


def _build_dataclass(
    raw_config: object,
    config_type: type[Any],
    path: str,
) -> Any:
    """Recursively build a strict nested dataclass from a YAML mapping."""

    if not isinstance(raw_config, Mapping):
        raise ConfigValidationError(f"'{path}' must be a YAML mapping.")
    config_fields = dataclass_fields(config_type)
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
        errors.append(
            "missing required fields: "
            + ", ".join(f"{path}.{field_name}" for field_name in missing_fields)
        )
    if unexpected_fields:
        errors.append(
            "unexpected fields: "
            + ", ".join(f"{path}.{field_name}" for field_name in unexpected_fields)
        )
    if errors:
        raise ConfigValidationError("Invalid configuration: " + "; ".join(errors) + ".")

    values = dict(raw_config)
    for field_name, nested_type in _NESTED_FIELD_TYPES.get(config_type, {}).items():
        if field_name in values and values[field_name] is not None:
            values[field_name] = _build_dataclass(
                values[field_name], nested_type, f"{path}.{field_name}"
            )
    try:
        return config_type(**values)
    except ConfigValidationError:
        raise
    except TypeError as error:
        raise ConfigValidationError(f"Invalid configuration at '{path}': {error}") from error
