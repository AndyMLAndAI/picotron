"""Hugging Face authentication-resolution regression tests."""

from picotron_sft.model_loading import _with_hf_token


def test_model_loader_uses_environment_token_without_overriding_explicit_value(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "environment-token")

    assert _with_hf_token({})["token"] == "environment-token"
    assert _with_hf_token({"token": "configured-token"})["token"] == "configured-token"
