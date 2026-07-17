"""CLI entrypoint for model-agnostic Picotron full fine-tuning."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picotron_sft.config import SFTConfig, load_sft_config
from picotron_sft.sft_trainer import run_sft


def _load_factory(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Factories must use the 'module:callable' format.")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError(f"Factory '{specification}' is not callable.")
    return factory


def main() -> None:
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
    args = parser.parse_args()

    config = load_sft_config(args.config)
    model = _load_factory(args.model_factory)(config.base_config)
    data_loader = _load_factory(args.data_loader_factory)(
        config.dataset_path, config.base_config
    )
    run_sft(
        model,
        data_loader,
        base_checkpoint_path=str(config.base_checkpoint_path),
        learning_rate=config.base_config.learning_rate,
        batch_size=config.base_config.batch_size,
        num_steps=args.max_steps or config.max_steps,
        display_config=config.base_config,
    )


if __name__ == "__main__":
    main()
