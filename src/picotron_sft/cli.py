"""Installed command-line entrypoint for Picotron full fine-tuning."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Sequence

from picotron_sft.config import load_sft_config
from picotron_sft.sft_trainer import run_sft


def build_parser() -> argparse.ArgumentParser:
    """Build the SFT CLI parser without loading a model or dataset."""

    parser = argparse.ArgumentParser(description="Run Picotron full fine-tuning.")
    parser.add_argument("--config", required=True, type=Path, help="SFT YAML config path.")
    parser.add_argument(
        "--model-factory",
        required=True,
        help="Import path for a callable accepting PicotronConfig and returning a causal LM.",
    )
    parser.add_argument(
        "--data-loader-factory",
        required=True,
        help="Import path for a callable accepting (dataset_path, PicotronConfig).",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def _load_factory(specification: str) -> Callable[..., Any]:
    """Load a ``module:callable`` factory specification."""

    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Factories must use the 'module:callable' format.")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError(f"Factory '{specification}' is not callable.")
    return factory


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and launch the existing model-agnostic SFT API."""

    args = build_parser().parse_args(argv)
    config = load_sft_config(args.config)
    model = _load_factory(args.model_factory)(config.base_config)
    data_loader = _load_factory(args.data_loader_factory)(
        config.dataset_path, config.base_config
    )
    run_sft(
        model,
        data_loader,
        base_checkpoint_path=str(config.base_checkpoint_path),
        learning_rate=(
            config.base_config.optimizer.learning_rate_scheduler.learning_rate
        ),
        weight_decay=config.base_config.optimizer.weight_decay,
        batch_size=config.base_config.tokens.micro_batch_size,
        num_steps=(
            args.max_steps or config.max_steps or config.base_config.tokens.train_steps
        ),
        display_config=config.base_config,
    )
