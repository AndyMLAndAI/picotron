"""RoPE math and RoPE-enabled model shape checks."""

import torch

from config_factory import make_test_config
from picotron.models.picotron_decoder import PicotronDecoderModel
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
    config = make_test_config(
        position_embedding_type="rope",
        rope_theta=500_000.0,
    )
    model = PicotronDecoderModel(config)
    input_ids = torch.randint(
        0,
        config.model.model_config.vocab_size,
        (config.tokens.micro_batch_size, config.tokens.sequence_length),
    )

    logits = model(input_ids)

    assert model.position_embeddings is None
    assert logits.shape == (
        config.tokens.micro_batch_size,
        config.tokens.sequence_length,
        config.model.model_config.vocab_size,
    )
