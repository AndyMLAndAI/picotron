"""CPU verification for the single-device training loop."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from picotron.config.config import load_config
from picotron.models.toy_model import ToyDecoderModel
from picotron.training.train_loop import train


class _RepeatableSyntheticDataset(Dataset[torch.Tensor]):
    """Small deterministic synthetic corpus so optimization can be measured."""

    def __init__(self, sequence: torch.Tensor, size: int) -> None:
        self.sequence = sequence
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        return self.sequence.clone()


def test_training_loss_decreases_on_cpu() -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/toy_model.yaml"
    config = load_config(config_path)
    sequence = (
        torch.arange(config.tokens.sequence_length, dtype=torch.long)
        % config.model.model_config.vocab_size
    )
    dataset = _RepeatableSyntheticDataset(
        sequence, size=config.tokens.micro_batch_size * 80
    )
    data_loader = DataLoader(
        dataset, batch_size=config.tokens.micro_batch_size, shuffle=False
    )
    model = ToyDecoderModel(config)

    losses = train(model, data_loader, config, max_steps=80)

    assert len(losses) == 80
    first_average = sum(losses[:10]) / 10
    last_average = sum(losses[-10:]) / 10
    print(f"first_10_avg={first_average:.6f} last_10_avg={last_average:.6f}")
    assert last_average < first_average
