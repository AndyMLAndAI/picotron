"""CPU tests for Rich and non-TTY training display paths."""

from dataclasses import replace
from io import StringIO
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from picotron.config.config import load_config
from picotron.logging.display import TrainingDisplay
from picotron.models.toy_model import ToyDecoderModel
from picotron.training.train_loop import train

try:
    from rich.console import Console
except ImportError:
    Console = None


class _RepeatableTokens(Dataset[torch.Tensor]):
    def __init__(self, sequence: torch.Tensor, size: int) -> None:
        self.sequence = sequence
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.sequence.clone()


def _config():
    path = Path(__file__).resolve().parents[1] / "src/picotron/config/toy_model.yaml"
    return replace(load_config(path), num_epochs=10)


def test_display_non_tty_does_not_use_live() -> None:
    config = _config()
    console = Console(force_terminal=False, file=StringIO()) if Console else None
    with TrainingDisplay(config, console=console, plain_interval=2) as display:
        assert not display.use_live
        display.update(step=1, loss=1.0, learning_rate=0.001, tokens_seen=32)
        display.update(step=2, loss=0.9, learning_rate=0.001, tokens_seen=64)
        if display._fallback_progress is not None:
            assert display._fallback_progress.n == 2


def test_display_tty_simulation_does_not_crash() -> None:
    config = _config()
    console = Console(force_terminal=True, file=StringIO()) if Console else None
    with TrainingDisplay(config, console=console) as display:
        display.update(step=1, loss=1.0, learning_rate=0.001, tokens_seen=32)
        if display.use_live:
            assert display._progress is not None
            assert display._progress_task is not None
            display.update(step=2, loss=0.9, learning_rate=0.001, tokens_seen=64)
            assert display._progress.tasks[display._progress_task].completed == 2


def test_training_loss_trend_survives_display() -> None:
    config = _config()
    sequence = torch.arange(config.max_seq_len, dtype=torch.long) % config.vocab_size
    loader = DataLoader(
        _RepeatableTokens(sequence, config.batch_size * 8),
        batch_size=config.batch_size,
        shuffle=False,
    )
    torch.manual_seed(17)
    losses = train(ToyDecoderModel(config), loader, config, max_steps=40)
    first_average = sum(losses[:10]) / 10
    last_average = sum(losses[-10:]) / 10
    print(f"first_10_avg={first_average:.6f} last_10_avg={last_average:.6f}")
    assert last_average < first_average
