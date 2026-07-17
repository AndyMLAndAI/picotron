"""Command-line entrypoint for single-device synthetic pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picotron.config.config import load_config
from picotron.data.dataloader import create_synthetic_dataloader
from picotron.models.toy_model import ToyDecoderModel
from picotron.parallel.ddp import initialize_distributed
from picotron.training.train_loop import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic Picotron pretraining.")
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
    args = parser.parse_args()

    config = load_config(args.config)
    distributed_info = initialize_distributed()
    model = ToyDecoderModel(config)
    data_loader = create_synthetic_dataloader(config, args.num_sequences)
    train(
        model,
        data_loader,
        config,
        max_steps=args.max_steps,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
        device=(
            torch.device(f"cuda:{distributed_info.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        ),
    )


if __name__ == "__main__":
    main()
