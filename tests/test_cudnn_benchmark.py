"""CPU-safe checks for fixed-shape cuDNN autotuning configuration."""

from __future__ import annotations

import copy
from unittest.mock import patch

import torch

from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.training.train_loop import train
from tests.config_factory import make_test_config


def test_cpu_training_does_not_enable_cudnn_benchmark() -> None:
    """CPU runs must retain their current cuDNN setting without CUDA access."""

    config = make_test_config(train_steps=1)
    model = PicotronDecoderModel(config)
    tokens = torch.arange(16, dtype=torch.long).reshape(2, 8) % 32
    original = torch.backends.cudnn.benchmark
    with patch("torch.cuda.is_available", return_value=False):
        train(model, [tokens], config, max_steps=1)
    assert torch.backends.cudnn.benchmark is original


def test_cudnn_benchmark_toggle_preserves_cpu_training_math() -> None:
    """Autotuning is CUDA-only and cannot alter fixed CPU training results."""

    config = make_test_config(train_steps=2, learning_rate=0.002)
    tokens = torch.arange(16, dtype=torch.long).reshape(2, 8) % 32
    torch.manual_seed(123)
    reference_model = PicotronDecoderModel(config)
    toggled_model = PicotronDecoderModel(config)
    toggled_model.load_state_dict(copy.deepcopy(reference_model.state_dict()))
    original = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        reference_losses = train(reference_model, [tokens] * 2, config, max_steps=2)
        torch.backends.cudnn.benchmark = True
        toggled_losses = train(toggled_model, [tokens] * 2, config, max_steps=2)
    finally:
        torch.backends.cudnn.benchmark = original
    torch.testing.assert_close(torch.tensor(reference_losses), torch.tensor(toggled_losses))
