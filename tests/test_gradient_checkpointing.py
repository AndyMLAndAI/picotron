"""Correctness checks for block-level activation checkpointing."""

from __future__ import annotations

import copy
from dataclasses import replace

import torch
from torch.nn import functional as F

from picotron.config.config import MoEConfig
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.training.train_loop import train
from tests.config_factory import make_test_config


def _loss(model: PicotronDecoderModel, tokens: torch.Tensor) -> torch.Tensor:
    logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].transpose(1, 2),
        tokens[:, 1:],
    )
    return loss if model.auxiliary_loss is None else loss + model.auxiliary_loss


def _matching_models(config: object) -> tuple[PicotronDecoderModel, PicotronDecoderModel]:
    torch.manual_seed(71)
    plain_model = PicotronDecoderModel(config)
    checkpointed_config = replace(
        config,
        model=replace(
            config.model,
            model_config=replace(
                config.model.model_config,
                gradient_checkpointing=True,
            ),
        ),
    )
    checkpointed_model = PicotronDecoderModel(checkpointed_config)
    checkpointed_model.load_state_dict(copy.deepcopy(plain_model.state_dict()))
    return plain_model, checkpointed_model


def test_checkpointing_matches_gqa_sliding_window_nope_training_math() -> None:
    config = make_test_config(
        num_hidden_layers=2,
        num_key_value_heads=2,
        attention_type="gqa",
        sliding_window_size=4,
        nope_layers=(1,),
        learning_rate=0.002,
    )
    config = replace(config, logging=replace(config.logging, file_logging=False))
    plain_model, checkpointed_model = _matching_models(config)
    tokens = torch.arange(16, dtype=torch.long).reshape(2, 8) % 32

    plain_losses = train(plain_model, [tokens] * 4, config, max_steps=4)
    checkpointed_config = replace(
        config,
        model=replace(
            config.model,
            model_config=replace(config.model.model_config, gradient_checkpointing=True),
        ),
    )
    checkpointed_losses = train(
        checkpointed_model, [tokens] * 4, checkpointed_config, max_steps=4
    )

    torch.testing.assert_close(
        torch.tensor(plain_losses), torch.tensor(checkpointed_losses), rtol=1e-5, atol=1e-6
    )
    for name, value in plain_model.state_dict().items():
        torch.testing.assert_close(
            value, checkpointed_model.state_dict()[name], rtol=1e-5, atol=1e-6
        )


def test_checkpointing_preserves_moe_auxiliary_loss_gradients() -> None:
    config = make_test_config(
        num_hidden_layers=2,
        num_key_value_heads=2,
        attention_type="gqa",
        moe_config=MoEConfig(num_experts=2, top_k=1, aux_loss_coefficient=0.1),
    )
    plain_model, checkpointed_model = _matching_models(config)
    plain_model.train()
    checkpointed_model.train()
    tokens = torch.arange(16, dtype=torch.long).reshape(2, 8) % 32

    plain_loss = _loss(plain_model, tokens)
    checkpointed_loss = _loss(checkpointed_model, tokens)
    assert checkpointed_model.auxiliary_loss is not None
    assert checkpointed_model.auxiliary_loss.requires_grad
    plain_loss.backward()
    checkpointed_loss.backward()

    torch.testing.assert_close(plain_loss, checkpointed_loss, rtol=1e-5, atol=1e-6)
    for (plain_name, plain_parameter), (checkpointed_name, checkpointed_parameter) in zip(
        plain_model.named_parameters(), checkpointed_model.named_parameters(), strict=True
    ):
        assert plain_name == checkpointed_name
        assert plain_parameter.grad is not None
        assert checkpointed_parameter.grad is not None
        torch.testing.assert_close(
            plain_parameter.grad, checkpointed_parameter.grad, rtol=1e-5, atol=1e-6
        )


def test_checkpointing_supports_mla_backward() -> None:
    config = make_test_config(attention_type="mla", kv_lora_rank=8)
    checkpointed_config = replace(
        config,
        model=replace(
            config.model,
            model_config=replace(config.model.model_config, gradient_checkpointing=True),
        ),
    )
    model = PicotronDecoderModel(checkpointed_config)
    model.train()
    tokens = torch.arange(16, dtype=torch.long).reshape(2, 8) % 32

    _loss(model, tokens).backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
