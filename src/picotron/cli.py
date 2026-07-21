"""Installed command-line entrypoint for Picotron pretraining."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import torch

from picotron.config.config import load_config
from picotron.data.dataloader import (
    create_memmap_dataloader,
    create_synthetic_dataloader,
)
from picotron.data.auto_preprocess import materialize_hf_dataset_sources
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.parallel.ddp import initialize_distributed
from picotron.training.train_loop import train


def build_parser() -> argparse.ArgumentParser:
    """Build the pretraining CLI parser without executing training."""

    parser = argparse.ArgumentParser(
        description="Run Picotron pretraining from a configured token cache or synthetic data."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1024,
        help="Number of synthetic sequences; ignored when data.dataset_token_path is set.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
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
    distributed_info = initialize_distributed(
        expected_world_size=config.parallelism.dp
    )
    if distributed_info.rank == 0 and getattr(config, "data", None) is not None:
        config = materialize_hf_dataset_sources(
            config, project_root=Path(__file__).resolve().parents[2]
        )
    if distributed_info.world_size > 1:
        torch.distributed.barrier()
        if distributed_info.rank != 0 and getattr(config, "data", None) is not None:
            config = materialize_hf_dataset_sources(
                config, project_root=Path(__file__).resolve().parents[2]
            )
    model = PicotronDecoderModel(config)
    data_config = getattr(config, "data", None)
    if data_config is not None and data_config.dataset_sources:
        data_loader = create_memmap_dataloader(
            config,
            rank=distributed_info.rank,
            world_size=distributed_info.world_size,
        )
    else:
        data_loader = create_synthetic_dataloader(
            config,
            args.num_sequences,
            seed=config.general.seed,
            rank=distributed_info.rank,
            world_size=distributed_info.world_size,
        )
    train(
        model,
        data_loader,
        config,
        max_steps=args.max_steps,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
        device=device,
    )


if __name__ == "__main__":
    main()
