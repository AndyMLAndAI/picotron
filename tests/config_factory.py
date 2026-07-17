"""Small nested-config factory shared by CPU tests."""

from __future__ import annotations

from typing import Any, Mapping

from picotron.config.config import (
    CheckpointConfig,
    DataConfig,
    GeneralConfig,
    LearningRateSchedulerConfig,
    LoggingConfig,
    ModelConfig,
    ModelSettings,
    MoEConfig,
    OptimizerConfig,
    OptimizerFactoryConfig,
    ParallelismConfig,
    PicotronConfig,
    TokensConfig,
    TritonKernelsConfig,
)


def make_test_config(
    *,
    vocab_size: int = 32,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 4,
    sequence_length: int = 8,
    micro_batch_size: int = 2,
    train_steps: int = 1,
    learning_rate: float = 0.001,
    checkpoint_interval: int = 100,
    num_key_value_heads: int | None = None,
    attention_type: str = "mha",
    nope_layers: tuple[int, ...] = (),
    rope_theta: float = 10_000.0,
    sliding_window_size: int | None = None,
    moe_config: MoEConfig | None = None,
    kv_lora_rank: int | None = None,
    tie_word_embeddings: bool = False,
    position_embedding_type: str = "rope",
    model_kwargs: Mapping[str, Any] | None = None,
    triton_kernels: TritonKernelsConfig | None = None,
    data: DataConfig | None = None,
    parallelism: ParallelismConfig | None = None,
) -> PicotronConfig:
    """Build a compact, valid nested Picotron configuration for tests."""

    return PicotronConfig(
        checkpoints=CheckpointConfig(checkpoint_interval=checkpoint_interval),
        model=ModelSettings(
            model_config=ModelConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                attention_type=attention_type,
                nope_layers=nope_layers,
                rope_theta=rope_theta,
                sliding_window_size=sliding_window_size,
                moe_config=moe_config,
                kv_lora_rank=kv_lora_rank,
                tie_word_embeddings=tie_word_embeddings,
                position_embedding_type=position_embedding_type,
                model_kwargs=dict(model_kwargs or {}),
            ),
            triton_kernels=triton_kernels or TritonKernelsConfig(),
        ),
        optimizer=OptimizerConfig(
            learning_rate_scheduler=LearningRateSchedulerConfig(
                learning_rate=learning_rate
            ),
            optimizer_factory=OptimizerFactoryConfig(),
        ),
        parallelism=parallelism or ParallelismConfig(),
        tokens=TokensConfig(
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            train_steps=train_steps,
        ),
        data=data or DataConfig(),
        logging=LoggingConfig(),
        general=GeneralConfig(),
    )
