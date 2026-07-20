"""Installed command-line entrypoint for Picotron DPO."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Sequence

from picotron_dpo.config import load_dpo_config
from picotron_dpo.dpo_trainer import run_dpo


def build_parser() -> argparse.ArgumentParser:
    """Build the DPO CLI parser without constructing models or datasets."""

    parser = argparse.ArgumentParser(description="Run Picotron Direct Preference Optimization.")
    parser.add_argument("--config", required=True, type=Path, help="DPO YAML config path.")
    parser.add_argument(
        "--model-factory",
        required=True,
        help="Import path for a callable accepting PicotronConfig and returning a causal LM.",
    )
    parser.add_argument(
        "--data-factory",
        required=True,
        help="Import path for a callable accepting (dataset_path, PicotronConfig, tokenizer).",
    )
    parser.add_argument(
        "--tokenizer-factory",
        required=True,
        help="Import path for a callable accepting PicotronConfig and returning a tokenizer.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and call the script-first DPO API."""

    args = build_parser().parse_args(argv)
    config = load_dpo_config(args.config)
    model = _load_factory(args.model_factory)(config.base_config)
    tokenizer = _load_factory(args.tokenizer_factory)(config.base_config)
    dataset = _load_factory(args.data_factory)(config.dataset_path, config.base_config, tokenizer)
    run_dpo(
        model,
        dataset,
        tokenizer=tokenizer,
        base_checkpoint_path=config.base_checkpoint_path,
        beta=config.beta,
        learning_rate=config.base_config.optimizer.learning_rate_scheduler.learning_rate,
        weight_decay=config.base_config.optimizer.weight_decay,
        batch_size=config.base_config.tokens.micro_batch_size,
        max_length=config.base_config.tokens.sequence_length,
        num_steps=args.max_steps or config.max_steps or config.base_config.tokens.train_steps,
        display_config=config.base_config,
    )


def _load_factory(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Factories must use the 'module:callable' format.")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError(f"Factory '{specification}' is not callable.")
    return factory
