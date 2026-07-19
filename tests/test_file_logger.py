"""CPU regression tests for persistent Picotron training logs."""

from __future__ import annotations

import copy
import csv
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from picotron.config.config import GeneralConfig, LoggingConfig
from picotron.logging.file_logger import FileLogger
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.training.train_loop import train
from tests.config_factory import make_test_config


class _RepeatedTokens(Dataset[torch.Tensor]):
    """A deterministic token dataset that makes two short runs comparable."""

    def __init__(self, tokens: torch.Tensor, size: int) -> None:
        self.tokens = tokens
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self.tokens.clone()


def _config(tmp_path: Path, *, file_logging: bool) -> object:
    base = make_test_config(train_steps=6, sequence_length=8, micro_batch_size=2)
    return replace(
        base,
        logging=LoggingConfig(
            file_logging=file_logging,
            file_logging_output_dir=str(tmp_path),
        ),
        general=GeneralConfig(project="picotron", run="file-logger-test", seed=19),
    )


def _loader(config: object) -> DataLoader[torch.Tensor]:
    sequence_length = config.tokens.sequence_length
    vocab_size = config.model.model_config.vocab_size
    sequence = torch.arange(sequence_length, dtype=torch.long) % vocab_size
    dataset = _RepeatedTokens(sequence, size=config.tokens.micro_batch_size * 8)
    return DataLoader(dataset, batch_size=config.tokens.micro_batch_size, shuffle=False)


def test_training_writes_metrics_and_transcript_without_changing_losses(tmp_path: Path) -> None:
    enabled_config = _config(tmp_path, file_logging=True)
    disabled_config = _config(tmp_path, file_logging=False)
    torch.manual_seed(123)
    logged_model = PicotronDecoderModel(enabled_config)
    baseline_model = copy.deepcopy(logged_model)

    logged_losses = train(logged_model, _loader(enabled_config), enabled_config, max_steps=6)
    baseline_losses = train(
        baseline_model, _loader(disabled_config), disabled_config, max_steps=6
    )

    assert logged_losses == pytest.approx(baseline_losses, abs=1e-7)
    run_directory = tmp_path / "file-logger-test"
    metrics_path = run_directory / "metrics.csv"
    log_path = run_directory / "run.log"
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert {"step", "loss", "learning_rate", "tokens_per_second", "elapsed"} <= set(
            reader.fieldnames
        )
    assert len(rows) == 6
    assert [int(row["step"]) for row in rows] == list(range(1, 7))
    transcript = log_path.read_text(encoding="utf-8")
    assert "Startup config:" in transcript
    assert "Starting pretraining run." in transcript
    assert "step=1" in transcript


def test_file_logger_adds_trainer_specific_metric_columns(tmp_path: Path) -> None:
    config = _config(tmp_path, file_logging=True)
    config = replace(config, general=replace(config.general, run="dpo-file-logger-test"))

    with FileLogger(config, method="dpo") as file_logger:
        file_logger.log_step(
            step=1,
            loss=0.5,
            learning_rate=1e-4,
            tokens_seen=8,
            metrics={"chosen_logprob": -1.0, "rejected_logprob": -2.0, "margin": 1.0},
        )
        warnings.warn("Triton fallback test warning", RuntimeWarning)

    with (tmp_path / "dpo-file-logger-test" / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as metrics_file:
        reader = csv.DictReader(metrics_file)
        assert reader.fieldnames is not None
        assert {"chosen_logprob", "rejected_logprob", "margin"} <= set(reader.fieldnames)
    transcript = (tmp_path / "dpo-file-logger-test" / "run.log").read_text(encoding="utf-8")
    assert "Triton fallback test warning" in transcript
