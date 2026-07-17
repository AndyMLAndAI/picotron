"""MoE routing, auxiliary-loss, dense-path, and learning checks."""

import torch

from picotron.config.config import MoEConfig, PicotronConfig
from picotron.models.toy_model import ToyDecoderModel
from picotron.nn.feedforward import SwiGLU
from picotron.nn.moe import MoEFeedForward
from picotron.training.train_loop import train


def _config(moe_config: MoEConfig | None) -> PicotronConfig:
    return PicotronConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_seq_len=8,
        learning_rate=0.003,
        batch_size=2,
        num_epochs=1,
        checkpoint_interval=100,
        moe_config=moe_config,
    )


def test_moe_shape_routing_and_auxiliary_gradients() -> None:
    moe = MoEFeedForward(4, 8, MoEConfig(num_experts=2, top_k=1, aux_loss_coefficient=0.1))
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]))
    hidden_states = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]])

    output, auxiliary_loss = moe(hidden_states)
    (output.sum() + auxiliary_loss).backward()

    assert output.shape == hidden_states.shape
    assert set(moe.last_routing_indices.flatten().tolist()) == {0, 1}
    assert auxiliary_loss.ndim == 0 and auxiliary_loss.requires_grad
    assert all(expert.down_projection.weight.grad is not None for expert in moe.experts)


def test_dense_ffn_path_remains_default() -> None:
    model = ToyDecoderModel(_config(None))

    logits = model(torch.randint(0, 32, (2, 8)))

    assert isinstance(model.layers[0].mlp, SwiGLU)
    assert model.auxiliary_loss is None
    assert logits.shape == (2, 8, 32)


def test_moe_model_training_loss_decreases() -> None:
    config = _config(MoEConfig(num_experts=2, top_k=2, aux_loss_coefficient=0.01))
    model = ToyDecoderModel(config)
    tokens = torch.arange(config.max_seq_len).unsqueeze(0).repeat(config.batch_size, 1)

    losses = train(model, [tokens] * 40, config, max_steps=40)

    assert model.auxiliary_loss is not None
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
