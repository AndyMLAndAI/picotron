"""End-to-end regression check for configurations without optional model kwargs."""

import torch

from config_factory import make_test_config
from picotron.models.toy_model import ToyDecoderModel
from picotron.training.train_loop import train


def test_core_only_config_runs_forward_and_one_training_step() -> None:
    config = make_test_config()
    model = ToyDecoderModel(config)
    batch = torch.randint(
        0,
        config.model.model_config.vocab_size,
        (config.tokens.micro_batch_size, config.tokens.sequence_length),
    )

    logits = model(batch)
    losses = train(model, [batch], config, max_steps=1)

    assert config.model.model_config.model_kwargs == {}
    assert logits.shape == (
        config.tokens.micro_batch_size,
        config.tokens.sequence_length,
        config.model.model_config.vocab_size,
    )
    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


def test_tie_word_embeddings_uses_one_shared_vocabulary_parameter() -> None:
    config = make_test_config(tie_word_embeddings=True)
    model = ToyDecoderModel(config)
    input_ids = torch.randint(
        0,
        config.model.model_config.vocab_size,
        (config.tokens.micro_batch_size, config.tokens.sequence_length),
    )

    logits = model(input_ids)

    assert model.output_projection.weight is model.token_embeddings.weight
    assert logits.shape[-1] == config.model.model_config.vocab_size
