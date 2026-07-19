"""CPU checkpoint-resume verification for the separate SFT package."""

from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from picotron.config.config import load_config
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.serialize.checkpoint import save_checkpoint
from picotron.training.train_loop import train
from picotron_sft import run_sft


class _TokenDataset(Dataset[torch.Tensor]):
    def __init__(self, sequence: torch.Tensor, size: int) -> None:
        self.sequence = sequence
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.sequence.clone()


class _SFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, sequence: torch.Tensor, size: int) -> None:
        self.sequence = sequence
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.sequence.clone(), "labels": self.sequence.clone()}


def test_sft_loads_picotron_checkpoint_and_learns_new_data(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/picotron_decoder.yaml"
    loaded_config = load_config(config_path)
    base_config = replace(
        loaded_config,
        tokens=replace(loaded_config.tokens, train_steps=2),
        checkpoints=replace(loaded_config.checkpoints, checkpoint_interval=100),
    )
    pretrain_sequence = torch.arange(base_config.tokens.sequence_length, dtype=torch.long)
    pretrain_loader = DataLoader(
        _TokenDataset(pretrain_sequence, base_config.tokens.micro_batch_size * 4),
        batch_size=base_config.tokens.micro_batch_size,
        shuffle=False,
    )
    torch.manual_seed(29)
    pretrained_model = PicotronDecoderModel(base_config)
    pretrained_optimizer = AdamW(
        pretrained_model.parameters(),
        lr=base_config.optimizer.learning_rate_scheduler.learning_rate,
    )
    train(
        pretrained_model,
        pretrain_loader,
        base_config,
        optimizer=pretrained_optimizer,
        max_steps=4,
    )
    checkpoint_path = tmp_path / "pretrained.pt"
    save_checkpoint(pretrained_model, pretrained_optimizer, step=4, path=checkpoint_path)
    saved_weights = {name: value.detach().clone() for name, value in pretrained_model.state_dict().items()}

    sft_sequence = torch.arange(base_config.tokens.sequence_length, dtype=torch.long).flip(0)
    sft_loader = DataLoader(
        _SFTDataset(sft_sequence, base_config.tokens.micro_batch_size * 10),
        batch_size=base_config.tokens.micro_batch_size,
        shuffle=False,
    )
    resumed_model = PicotronDecoderModel(base_config)
    losses = run_sft(
        resumed_model,
        sft_loader,
        base_checkpoint_path=str(checkpoint_path),
        learning_rate=base_config.optimizer.learning_rate_scheduler.learning_rate,
        batch_size=base_config.tokens.micro_batch_size,
        num_steps=10,
    )
    print(f"resumed_first_loss={losses[0]:.6f} resumed_last_loss={losses[-1]:.6f}")
    assert len(losses) == 10
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3
