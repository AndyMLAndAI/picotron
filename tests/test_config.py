"""Strict nested Picotron configuration schema regression tests."""

from pathlib import Path

import pytest

from picotron.config.config import ConfigValidationError, ModelConfig, PicotronConfig, load_config


REQUIRED_CONFIG = """\
checkpoints:
  checkpoint_interval: 100
model:
  model_config:
    vocab_size: 64
    hidden_size: 32
    intermediate_size: 64
    num_hidden_layers: 2
    num_attention_heads: 4
optimizer:
  learning_rate_scheduler:
    learning_rate: 0.001
parallelism: {}
tokens:
  sequence_length: 16
  micro_batch_size: 2
  train_steps: 1
data: {}
logging: {}
general: {}
"""


def test_minimal_nested_config_uses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(REQUIRED_CONFIG, encoding="utf-8")

    config = load_config(config_path)

    assert isinstance(config, PicotronConfig)
    assert config.model.model_config.model_kwargs == {}
    assert config.model.dtype == "auto"
    assert config.parallelism.dp == 1
    assert config.parallelism.zero_stage == 0


def test_qwen_style_model_kwargs_are_accepted_without_schema_changes(tmp_path: Path) -> None:
    config_path = tmp_path / "qwen3_5.yaml"
    config_path.write_text(
        REQUIRED_CONFIG.replace(
            "    num_attention_heads: 4\n",
            "    num_attention_heads: 4\n"
            "    num_key_value_heads: 4\n"
            "    model_kwargs:\n"
            "      head_dim: 256\n"
            "      layer_pattern: 3:1 GatedDeltaNet linear_attention/full_attention\n"
            "      linear_conv_kernel_dim: 4\n",
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.model_config.num_key_value_heads == 4
    assert config.model.model_config.model_kwargs["layer_pattern"].startswith("3:1")


def test_missing_required_field_still_fails_loudly(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_vocab.yaml"
    config_path.write_text(
        REQUIRED_CONFIG.replace("    vocab_size: 64\n", ""), encoding="utf-8"
    )

    with pytest.raises(
        ConfigValidationError,
        match="missing required fields: configuration.model.model_config.vocab_size",
    ):
        load_config(config_path)


def test_all_advanced_config_fields_load_together(tmp_path: Path) -> None:
    """Prevent implemented nested fields from falling out of strict YAML validation."""

    config_path = tmp_path / "advanced.yaml"
    config_path.write_text(
        """\
checkpoints:
  checkpoint_interval: 25
  checkpoints_path: /tmp/checkpoints/run.pt
  resume_checkpoint_path: /tmp/checkpoints/resume.pt
  load_optimizer: false
  load_lr_scheduler: false
  save_final_state: true
model:
  dtype: float32
  triton_kernels:
    rmsnorm: true
    swiglu: true
    rope: true
    attention: true
    cross_entropy: true
    adamw: true
  model_config:
    vocab_size: 64
    hidden_size: 32
    intermediate_size: 64
    num_hidden_layers: 2
    num_attention_heads: 4
    num_key_value_heads: 2
    attention_type: gqa
    nope_layers: [1]
    rope_theta: 500000.0
    sliding_window_size: 8
    moe_config:
      num_experts: 4
      top_k: 2
      aux_loss_coefficient: 0.01
    kv_lora_rank: 8
    tie_word_embeddings: true
    position_embedding_type: learned
    model_kwargs:
      head_dim: 8
      layer_pattern: 3:1 GatedDeltaNet linear_attention/full_attention
optimizer:
  learning_rate_scheduler:
    learning_rate: 0.001
    lr_decay_steps: 100
    lr_decay_style: cosine
    lr_warmup_steps: 5
    lr_warmup_style: linear
    min_decay_lr: 0.0001
  optimizer_factory:
    name: adamw
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
  weight_decay: 0.1
  clip_grad: 1.0
parallelism:
  dp: 1
  zero_stage: 2
tokens:
  sequence_length: 16
  micro_batch_size: 2
  train_steps: 100
data:
  dataset_token_path: /tmp/tokens.uint16
  tokenizer_name: gpt2
  vocab_size: 64
logging:
  log_level: INFO
  iteration_step_info_interval: 5
  file_logging: true
  file_logging_output_dir: /tmp/logs
general:
  project: picotron
  run: advanced-config
  seed: 17
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    model_config = config.model.model_config

    assert config.checkpoints.checkpoint_interval == 25
    assert config.checkpoints.checkpoints_path == "/tmp/checkpoints/run.pt"
    assert config.checkpoints.resume_checkpoint_path == "/tmp/checkpoints/resume.pt"
    assert not config.checkpoints.load_optimizer
    assert not config.checkpoints.load_lr_scheduler
    assert config.checkpoints.save_final_state
    assert config.model.dtype == "float32"
    assert all(
        (
            config.model.triton_kernels.rmsnorm,
            config.model.triton_kernels.swiglu,
            config.model.triton_kernels.rope,
            config.model.triton_kernels.attention,
            config.model.triton_kernels.cross_entropy,
            config.model.triton_kernels.adamw,
        )
    )
    assert model_config.attention_type == "gqa"
    assert model_config.nope_layers == (1,)
    assert model_config.num_key_value_heads == 2
    assert model_config.sliding_window_size == 8
    assert model_config.kv_lora_rank == 8
    assert model_config.tie_word_embeddings
    assert model_config.position_embedding_type == "learned"
    assert model_config.moe_config is not None
    assert model_config.moe_config.num_experts == 4
    assert model_config.rope_theta == 500000.0
    assert model_config.model_kwargs["layer_pattern"].startswith("3:1")
    assert config.optimizer.learning_rate_scheduler.lr_warmup_steps == 5
    assert config.optimizer.optimizer_factory.adam_beta2 == 0.95
    assert config.optimizer.learning_rate_scheduler.lr_decay_style == "cosine"
    assert config.optimizer.weight_decay == 0.1
    assert config.optimizer.clip_grad == 1.0
    assert config.parallelism.dp == 1
    assert config.parallelism.zero_stage == 2
    assert config.tokens.sequence_length == 16
    assert config.tokens.micro_batch_size == 2
    assert config.tokens.train_steps == 100
    assert config.data.dataset_token_path == "/tmp/tokens.uint16"
    assert config.data.tokenizer_name == "gpt2"
    assert config.data.vocab_size == 64
    assert config.logging.log_level == "INFO"
    assert config.logging.iteration_step_info_interval == 5
    assert config.logging.file_logging
    assert config.logging.file_logging_output_dir == "/tmp/logs"
    assert config.general.run == "advanced-config"
    assert config.general.seed == 17


def test_attention_type_and_nope_layers_yaml_regression(tmp_path: Path) -> None:
    """Cover the exact advanced-field combination that failed on Kaggle."""

    config_path = tmp_path / "kaggle_regression.yaml"
    config_path.write_text(
        REQUIRED_CONFIG.replace(
            "    num_attention_heads: 4\n",
            "    num_attention_heads: 4\n    attention_type: mha\n    nope_layers: [0]\n",
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model.model_config.attention_type == "mha"
    assert config.model.model_config.nope_layers == (0,)


def test_mha_rejects_grouped_key_value_heads() -> None:
    with pytest.raises(
        ConfigValidationError,
        match="attention_type: mha.*num_key_value_heads.*num_attention_heads",
    ):
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_type="mha",
        )


def test_gqa_requires_strictly_fewer_key_value_heads() -> None:
    with pytest.raises(
        ConfigValidationError,
        match="attention_type: gqa.*smaller than num_attention_heads",
    ):
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            attention_type="gqa",
        )


def test_gqa_requires_key_value_heads_to_be_specified() -> None:
    with pytest.raises(
        ConfigValidationError,
        match="num_key_value_heads.*required when attention_type is 'gqa'",
    ):
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            attention_type="gqa",
        )


def test_data_vocab_size_must_match_model_vocab_size(tmp_path: Path) -> None:
    config_path = tmp_path / "mismatched_vocab.yaml"
    config_path.write_text(
        REQUIRED_CONFIG.replace("data: {}\n", "data:\n  vocab_size: 63\n"),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigValidationError,
        match="data.vocab_size.*model.model_config.vocab_size",
    ):
        load_config(config_path)


def test_flat_format_is_rejected_instead_of_silently_migrated(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy_flat.yaml"
    config_path.write_text(
        "vocab_size: 64\nhidden_size: 32\nmax_seq_len: 16\n", encoding="utf-8"
    )

    with pytest.raises(ConfigValidationError, match="unexpected fields"):
        load_config(config_path)
