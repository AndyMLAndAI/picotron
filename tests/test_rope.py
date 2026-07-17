"""RoPE math and RoPE-enabled model shape checks."""

import torch

from picotron.config.config import PicotronConfig
from picotron.models.toy_model import ToyDecoderModel
from picotron.nn.rope import apply_rotary_embedding, rotary_cos_sin


def test_rope_dot_product_depends_only_on_relative_position() -> None:
    torch.manual_seed(0)
    head_dim = 8
    base_query = torch.randn(1, 1, 1, head_dim).expand(-1, -1, 12, -1).clone()
    base_key = torch.randn(1, 1, 1, head_dim).expand(-1, -1, 12, -1).clone()
    cosine, sine = rotary_cos_sin(
        12, head_dim, 10_000.0, device=base_query.device, dtype=base_query.dtype
    )
    rotated_base_query = apply_rotary_embedding(base_query, cosine, sine)
    rotated_base_key = apply_rotary_embedding(base_key, cosine, sine)
    relative_one = (rotated_base_query[:, :, 2] * rotated_base_key[:, :, 5]).sum()
    relative_two = (rotated_base_query[:, :, 6] * rotated_base_key[:, :, 9]).sum()

    torch.testing.assert_close(relative_one, relative_two)


def test_rope_enabled_model_forward_shape() -> None:
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
        model_kwargs={"position_embedding_type": "rope", "rope_theta": 500_000.0},
    )
    model = ToyDecoderModel(config)
    input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len))

    logits = model(input_ids)

    assert model.position_embeddings is None
    assert logits.shape == (config.batch_size, config.max_seq_len, config.vocab_size)
