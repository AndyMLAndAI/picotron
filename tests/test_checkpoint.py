"""CPU verification of exact checkpoint resume behavior."""

from pathlib import Path
from dataclasses import replace

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from picotron.config.config import load_config
from picotron.models.toy_model import ToyDecoderModel
from picotron.serialize.checkpoint import load_checkpoint
from picotron.training.train_loop import train


class _RepeatableTokens(Dataset[torch.Tensor]):
    def __init__(self, sequence: torch.Tensor, size: int) -> None:
        self.sequence = sequence
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.sequence.clone()


def test_checkpoint_resume_preserves_weights_and_loss(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/toy_model.yaml"
    config = replace(load_config(config_path), checkpoint_interval=3)
    sequence = torch.arange(config.max_seq_len, dtype=torch.long) % config.vocab_size
    data_loader = DataLoader(
        _RepeatableTokens(sequence, config.batch_size * 12),
        batch_size=config.batch_size,
        shuffle=False,
    )
    checkpoint_path = tmp_path / "resume.pt"

    torch.manual_seed(11)
    model = ToyDecoderModel(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    first_losses = train(
        model,
        data_loader,
        config,
        optimizer=optimizer,
        max_steps=6,
        checkpoint_path=str(checkpoint_path),
    )
    saved_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}

    fresh_model = ToyDecoderModel(config)
    fresh_optimizer = AdamW(fresh_model.parameters(), lr=config.learning_rate)
    resumed_step = load_checkpoint(fresh_model, fresh_optimizer, checkpoint_path)
    assert resumed_step == config.checkpoint_interval * (6 // config.checkpoint_interval)
    for name, value in saved_weights.items():
        torch.testing.assert_close(value, fresh_model.state_dict()[name], rtol=0, atol=0)

    resumed_losses = train(
        fresh_model,
        data_loader,
        config,
        optimizer=fresh_optimizer,
        resume_from=str(checkpoint_path),
        max_steps=6,
    )
    print(
        f"saved_last_loss={first_losses[-1]:.6f} "
        f"resumed_first_loss={resumed_losses[0]:.6f} "
        f"resumed_last_loss={resumed_losses[-1]:.6f}"
    )
    assert resumed_losses[0] <= first_losses[-1] + 0.25
    assert resumed_losses[-1] < resumed_losses[0]
