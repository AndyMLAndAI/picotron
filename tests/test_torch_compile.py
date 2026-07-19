"""CPU checks for the optional guarded ``torch.compile`` integration."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW

from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.serialize.checkpoint import load_checkpoint, save_checkpoint
from picotron.training.train_loop import _maybe_compile_model, train
from tests.config_factory import make_test_config


def _compile_config():
    config = make_test_config(train_steps=3, learning_rate=0.002, compile_model=True)
    return replace(config, logging=replace(config.logging, file_logging=False))


def _tokens(config: object) -> torch.Tensor:
    return (
        torch.arange(16, dtype=torch.long).reshape(2, 8)
        % config.model.model_config.vocab_size
    )


def test_compile_enabled_matches_eager_training_losses() -> None:
    """Compilation may optimize execution, but it must preserve training math."""

    compiled_config = _compile_config()
    eager_config = replace(
        compiled_config,
        model=replace(compiled_config.model, compile_model=False),
    )
    torch.manual_seed(91)
    eager_model = PicotronDecoderModel(eager_config)
    compiled_model = PicotronDecoderModel(compiled_config)
    compiled_model.load_state_dict(copy.deepcopy(eager_model.state_dict()))
    batches = [_tokens(compiled_config)] * 3

    eager_losses = train(eager_model, batches, eager_config, max_steps=3)
    compiled_losses = train(compiled_model, batches, compiled_config, max_steps=3)

    torch.testing.assert_close(
        torch.tensor(compiled_losses), torch.tensor(eager_losses), rtol=1e-5, atol=1e-6
    )


def test_compiled_model_checkpoint_loads_into_eager_model(tmp_path: Path) -> None:
    """Compiled wrapper state must serialize as portable native-model weights."""

    config = _compile_config()
    torch.manual_seed(92)
    original_model = PicotronDecoderModel(config)
    compiled_model = _maybe_compile_model(original_model, config)
    optimizer = AdamW(compiled_model.parameters(), lr=0.002)
    tokens = _tokens(config)
    logits = compiled_model(tokens)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), tokens[:, 1:])
    loss.backward()
    optimizer.step()
    checkpoint_path = tmp_path / "compiled.safetensors"
    save_checkpoint(compiled_model, optimizer, step=1, path=checkpoint_path)

    eager_model = PicotronDecoderModel(config)
    eager_optimizer = AdamW(eager_model.parameters(), lr=0.002)
    assert load_checkpoint(eager_model, eager_optimizer, checkpoint_path) == 1
    for name, value in original_model.state_dict().items():
        torch.testing.assert_close(value, eager_model.state_dict()[name], rtol=0, atol=0)
