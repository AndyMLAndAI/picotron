"""Strict required-field and optional model-kwargs configuration tests."""

from pathlib import Path

import pytest

from picotron.config.config import ConfigValidationError, PicotronConfig, load_config


REQUIRED_CONFIG = """\
vocab_size: 64
hidden_size: 32
intermediate_size: 64
num_hidden_layers: 2
num_attention_heads: 4
max_seq_len: 16
learning_rate: 0.001
batch_size: 2
num_epochs: 1
checkpoint_interval: 100
"""


def test_minimal_required_config_uses_empty_model_kwargs(tmp_path: Path) -> None:
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(REQUIRED_CONFIG, encoding="utf-8")

    config = load_config(config_path)

    assert isinstance(config, PicotronConfig)
    assert config.model_kwargs == {}


def test_qwen_style_model_kwargs_are_accepted_without_schema_changes(tmp_path: Path) -> None:
    config_path = tmp_path / "qwen3_5.yaml"
    config_path.write_text(
        REQUIRED_CONFIG
        + """\
model_kwargs:
  num_key_value_heads: 4
  head_dim: 256
  layer_pattern: 3:1 GatedDeltaNet linear_attention/full_attention
  linear_conv_kernel_dim: 4
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model_kwargs["num_key_value_heads"] == 4
    assert config.model_kwargs["layer_pattern"].startswith("3:1")


def test_missing_required_field_still_fails_loudly(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_vocab.yaml"
    config_path.write_text(REQUIRED_CONFIG.replace("vocab_size: 64\n", ""), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="missing required fields: vocab_size"):
        load_config(config_path)


def test_all_advanced_config_fields_load_together(tmp_path: Path) -> None:
    """Prevent advanced model fields from falling out of the strict YAML schema."""

    config_path = tmp_path / "advanced.yaml"
    config_path.write_text(
        REQUIRED_CONFIG
        + """\
zero_stage: 0
num_key_value_heads: 2
sliding_window_size: 8
moe_config:
  num_experts: 4
  top_k: 2
  aux_loss_coefficient: 0.01
nope_layers: [1]
attention_type: gqa
kv_lora_rank: 8
model_kwargs:
  position_embedding_type: rope
  rope_theta: 500000.0
  use_triton_rmsnorm: false
  use_triton_swiglu: false
  use_triton_rope: false
  use_triton_cross_entropy: false
  use_triton_adamw: false
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.attention_type == "gqa"
    assert config.nope_layers == (1,)
    assert config.num_key_value_heads == 2
    assert config.sliding_window_size == 8
    assert config.kv_lora_rank == 8
    assert config.moe_config is not None
    assert config.moe_config.num_experts == 4
    assert config.model_kwargs["rope_theta"] == 500000.0


def test_attention_type_and_nope_layers_yaml_regression(tmp_path: Path) -> None:
    """Cover the exact advanced-field combination that failed on Kaggle."""

    config_path = tmp_path / "kaggle_regression.yaml"
    config_path.write_text(
        REQUIRED_CONFIG + "attention_type: mha\nnope_layers: [0]\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.attention_type == "mha"
    assert config.nope_layers == (0,)
