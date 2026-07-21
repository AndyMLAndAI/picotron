"""CPU checks for CLI orchestration of on-demand Hugging Face token caches."""

from pathlib import Path

from picotron.config.config import DataConfig, DatasetSourceConfig, load_config
from picotron.data.auto_preprocess import materialize_hf_dataset_sources
from tests.config_factory import make_test_config


def test_hf_dataset_source_is_deterministic_and_validates_with_legacy_sources(tmp_path: Path) -> None:
    remote = DatasetSourceConfig(
        hf_name="org/example",
        hf_config="subset",
        target_tokens=12,
        weight=0.7,
    )
    legacy = DatasetSourceConfig(path=str(tmp_path / "existing.uint16"), weight=0.3)
    config = make_test_config(
        data=DataConfig(
            datasets=(remote, legacy),
            tokenizer_name="gpt2",
            token_cache_dir=str(tmp_path / "cache"),
            num_workers=0,
        )
    )

    assert remote.needs_preprocessing
    assert remote.cache_path(tokenizer_name="gpt2", cache_dir=config.data.token_cache_dir) == remote.cache_path(
        tokenizer_name="gpt2", cache_dir=config.data.token_cache_dir
    )
    assert not legacy.needs_preprocessing
    assert config.data.dataset_sources == (remote, legacy)

    config_path = tmp_path / "hf_sources.yaml"
    config_path.write_text(
        """\
checkpoints:
  checkpoint_interval: 1
model:
  model_config:
    vocab_size: 32
    hidden_size: 16
    intermediate_size: 32
    num_hidden_layers: 1
    num_attention_heads: 4
optimizer:
  learning_rate_scheduler:
    learning_rate: 0.001
parallelism: {}
tokens:
  sequence_length: 4
  micro_batch_size: 1
  train_steps: 1
data:
  tokenizer_name: gpt2
  datasets:
    - hf_name: org/example
      hf_config: subset
      target_tokens: 12
      weight: 0.7
    - path: existing.uint16
      weight: 0.3
logging: {}
general: {}
""",
        encoding="utf-8",
    )
    parsed = load_config(config_path)
    assert parsed.data.datasets[0].hf_name == "org/example"
    assert parsed.data.datasets[1].path == "existing.uint16"


def test_existing_hf_cache_skips_preprocessing(tmp_path: Path, monkeypatch) -> None:
    source = DatasetSourceConfig(hf_name="org/example", target_tokens=8)
    config = make_test_config(
        data=DataConfig(
            datasets=(source,),
            tokenizer_name="gpt2",
            token_cache_dir=str(tmp_path / "cache"),
            num_workers=0,
        )
    )
    cache_path = Path(
        source.cache_path(tokenizer_name="gpt2", cache_dir=config.data.token_cache_dir)
    )
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"\0" * (source.target_tokens * 2))

    def should_not_run(*args, **kwargs):
        raise AssertionError("complete caches must not invoke preprocessing")

    monkeypatch.setattr("picotron.data.auto_preprocess.subprocess.run", should_not_run)
    resolved = materialize_hf_dataset_sources(config, project_root=Path(__file__).parents[1])

    assert resolved.data.datasets == (DatasetSourceConfig(path=str(cache_path), weight=1.0),)
