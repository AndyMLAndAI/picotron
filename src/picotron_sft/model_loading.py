"""Optional Unsloth-first loader with a conservative Hugging Face fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch


LOGGER = logging.getLogger("picotron_sft")

# These model_type values correspond to text model families in Unsloth's
# official catalog. Unknown values deliberately use standard Transformers.
_UNSLOTH_MODEL_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen3",
        "gemma",
        "gemma2",
        "gemma3",
        "phi",
        "phi3",
        "phi4",
        "deepseek_v2",
        "deepseek_v3",
        "falcon",
        "granite",
        "gpt_oss",
    }
)


def load_model(
    model_name: str,
    max_seq_length: int = 1024,
    dtype: torch.dtype | None = None,
    load_in_4bit: bool = False,
    full_finetuning: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a causal LM through Unsloth when safely supported, else Transformers.

    The model type is inspected before calling Unsloth. Any unavailable,
    unsupported, or failing Unsloth path falls back to standard Hugging Face
    loading and logs the reason. ``load_in_4bit`` on the fallback requires a
    compatible bitsandbytes installation; if unavailable it falls back to a
    normal-precision HF load with a warning.
    """

    if not model_name:
        raise ValueError("model_name must be non-empty.")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive.")
    kwargs = _with_hf_token(kwargs)

    unsloth_result = _try_unsloth_load(
        model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        full_finetuning=full_finetuning,
        kwargs=kwargs,
    )
    if unsloth_result is not None:
        return unsloth_result
    return _load_with_transformers(model_name, dtype=dtype, load_in_4bit=load_in_4bit, kwargs=kwargs)


def _try_unsloth_load(
    model_name: str,
    *,
    max_seq_length: int,
    dtype: torch.dtype | None,
    load_in_4bit: bool,
    full_finetuning: bool,
    kwargs: dict[str, Any],
) -> tuple[Any, Any] | None:
    try:
        # Unsloth recommends importing itself before Transformers so its patches
        # are applied before model classes are imported.
        import unsloth  # noqa: F401
        from unsloth import FastModel
        from transformers import AutoConfig
    except ImportError:
        LOGGER.info("Unsloth is not installed; loading '%s' with Transformers.", model_name)
        return None

    try:
        model_type = AutoConfig.from_pretrained(model_name, **_config_kwargs(kwargs)).model_type
    except Exception as error:
        LOGGER.warning(
            "Could not inspect '%s' for Unsloth compatibility (%s); using Transformers.",
            model_name,
            error,
        )
        return None
    if model_type not in _UNSLOTH_MODEL_TYPES:
        LOGGER.info(
            "Model type '%s' is not in Picotron's verified Unsloth allowlist; using Transformers.",
            model_type,
        )
        return None

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            full_finetuning=full_finetuning,
            **kwargs,
        )
    except Exception as error:
        LOGGER.warning(
            "Unsloth could not load '%s' (%s); falling back to Transformers.", model_name, error
        )
        return None
    LOGGER.info("Loaded '%s' through Unsloth FastModel.", model_name)
    return model, tokenizer


def _load_with_transformers(
    model_name: str,
    *,
    dtype: torch.dtype | None,
    load_in_4bit: bool,
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = dict(kwargs)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception as error:
            LOGGER.warning(
                "4-bit HF loading is unavailable (%s); loading '%s' without quantization.",
                error,
                model_name,
            )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception:
        if "quantization_config" not in model_kwargs:
            raise
        LOGGER.warning("4-bit HF loading failed for '%s'; retrying without quantization.", model_name)
        model_kwargs.pop("quantization_config")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **_tokenizer_kwargs(kwargs))
    LOGGER.info("Loaded '%s' through Hugging Face Transformers.", model_name)
    return model, tokenizer


def _config_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in kwargs.items() if name in {"revision", "token", "trust_remote_code"}}


def _tokenizer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in kwargs.items() if name in {"revision", "token", "trust_remote_code"}}


def _with_hf_token(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Apply the standard HF_TOKEN fallback without overriding an explicit token."""

    resolved = dict(kwargs)
    if resolved.get("token") is None:
        environment_token = os.getenv("HF_TOKEN")
        if environment_token:
            resolved["token"] = environment_token
    return resolved
