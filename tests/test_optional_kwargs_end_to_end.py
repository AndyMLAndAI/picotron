"""End-to-end regression check for configurations without optional model kwargs."""

import torch

from picotron.config.config import PicotronConfig
from picotron.models.toy_model import ToyDecoderModel
from picotron.training.train_loop import train


def test_core_only_config_runs_forward_and_one_training_step() -> None:
    config = PicotronConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_seq_len=8,
        learning_rate=0.001,
        batch_size=2,
        num_epochs=1,
        checkpoint_interval=100,
    )
    model = ToyDecoderModel(config)
    batch = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len))

    logits = model(batch)
    losses = train(model, [batch], config, max_steps=1)

    assert config.model_kwargs == {}
    assert logits.shape == (config.batch_size, config.max_seq_len, config.vocab_size)
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))
