"""Installed command-line entrypoint for Picotron pretraining."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import torch

from picotron.config.config import load_config
from picotron.data.dataloader import create_synthetic_dataloader
from picotron.models.toy_model import ToyDecoderModel
from picotron.parallel.ddp import initialize_distributed
from picotron.training.train_loop import train


def build_parser() -> argparse.ArgumentParser:
    """Build the pretraining CLI parser without executing training."""

    parser = argparse.ArgumentParser(description="Run Picotron synthetic pretraining.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1024,
        help="Number of synthetic sequences in each epoch.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and launch the existing pretraining components."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    torch.manual_seed(config.general.seed)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    initialize_distributed(expected_world_size=config.parallelism.dp)
    model = ToyDecoderModel(config)
    data_loader = create_synthetic_dataloader(config, args.num_sequences)
    train(
        model,
        data_loader,
        config,
        max_steps=args.max_steps,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
        device=device,
    )
