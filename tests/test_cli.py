"""CLI parsing and delegation tests without launching training."""

from pathlib import Path
from types import SimpleNamespace

from picotron import cli
from picotron_sft import cli as sft_cli


def test_pretraining_cli_parses_config_and_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--num-sequences",
            "12",
            "--max-steps",
            "5",
            "--checkpoint-path",
            "run.pt",
            "--resume-from",
            "old.pt",
        ]
    )

    assert args.config == Path("config.yaml")
    assert args.num_sequences == 12
    assert args.max_steps == 5
    assert args.checkpoint_path == Path("run.pt")
    assert args.resume_from == Path("old.pt")


def test_pretraining_cli_delegates_to_train(monkeypatch) -> None:
    calls = {}
    config = SimpleNamespace(
        general=SimpleNamespace(seed=17),
        parallelism=SimpleNamespace(dp=1),
    )
    distributed = SimpleNamespace(local_rank=0)

    def fake_load_config(path):
        calls["config"] = path
        return config

    def fake_model(loaded_config):
        calls["model_config"] = loaded_config
        return "model"

    def fake_train(*args, **kwargs):
        calls["train"] = (args, kwargs)

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(
        cli, "initialize_distributed", lambda *args, **kwargs: distributed
    )
    monkeypatch.setattr(cli, "PicotronDecoderModel", fake_model)
    monkeypatch.setattr(cli, "create_synthetic_dataloader", lambda loaded_config, count: ("loader", count))
    monkeypatch.setattr(cli, "train", fake_train)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    cli.main(["--config", "config.yaml", "--num-sequences", "3", "--max-steps", "2"])

    assert calls["config"] == Path("config.yaml")
    assert calls["model_config"] is config
    assert calls["train"][0][:3] == ("model", ("loader", 3), config)
    assert calls["train"][1]["max_steps"] == 2


def test_sft_cli_parses_and_delegates(monkeypatch) -> None:
    args = sft_cli.build_parser().parse_args(
        [
            "--config",
            "sft.yaml",
            "--model-factory",
            "demo:make_model",
            "--data-loader-factory",
            "demo:make_data",
            "--max-steps",
            "4",
        ]
    )
    assert args.config == Path("sft.yaml")
    assert args.max_steps == 4

    calls = {}
    base_config = SimpleNamespace(
        optimizer=SimpleNamespace(
            learning_rate_scheduler=SimpleNamespace(learning_rate=0.001)
        ),
        tokens=SimpleNamespace(micro_batch_size=2),
    )
    sft_config = SimpleNamespace(
        base_config=base_config,
        base_checkpoint_path=Path("base.pt"),
        dataset_path=Path("data"),
        max_steps=10,
    )
    monkeypatch.setattr(sft_cli, "load_sft_config", lambda path: sft_config)
    monkeypatch.setattr(sft_cli, "_load_factory", lambda spec: lambda *factory_args: (spec, factory_args))
    monkeypatch.setattr(sft_cli, "run_sft", lambda *args, **kwargs: calls.setdefault("run", (args, kwargs)))

    sft_cli.main(
        [
            "--config",
            "sft.yaml",
            "--model-factory",
            "demo:make_model",
            "--data-loader-factory",
            "demo:make_data",
            "--max-steps",
            "4",
        ]
    )

    assert calls["run"][0][:2] == (
        ("demo:make_model", (base_config,)),
        ("demo:make_data", (Path("data"), base_config)),
    )
    assert calls["run"][1]["num_steps"] == 4
