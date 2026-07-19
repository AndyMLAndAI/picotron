"""Installed command-line entrypoint for Picotron GRPO."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Sequence

from picotron.config.config import load_config
from picotron_grpo.grpo_trainer import run_grpo


def build_parser() -> argparse.ArgumentParser:
    """Build the GRPO parser without creating a model or loading data."""

    parser = argparse.ArgumentParser(description="Run Picotron Group Relative Policy Optimization.")
    parser.add_argument("--config", required=True, type=Path, help="Nested Picotron YAML config path.")
    parser.add_argument("--model-factory", required=True, help="Import path: module:callable.")
    parser.add_argument("--tokenizer-factory", required=True, help="Import path: module:callable.")
    parser.add_argument("--prompts-factory", required=True, help="Import path: module:callable.")
    parser.add_argument("--reward-fn", required=True, help="Import path: module:callable.")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Construct user-provided model/data callables and run GRPO."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    model = _load_factory(args.model_factory)(config)
    tokenizer = _load_factory(args.tokenizer_factory)(config)
    prompts = _load_factory(args.prompts_factory)(config)
    reward_fn = _load_factory(args.reward_fn)
    run_grpo(
        model,
        prompts,
        reward_fn,
        tokenizer=tokenizer,
        group_size=args.group_size,
        beta=args.beta,
        clip_epsilon=args.clip_epsilon,
        learning_rate=config.optimizer.learning_rate_scheduler.learning_rate,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_steps=args.max_steps or config.tokens.train_steps,
        display_config=config,
    )


def _load_factory(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Factories must use the 'module:callable' format.")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError(f"Factory '{specification}' is not callable.")
    return factory
