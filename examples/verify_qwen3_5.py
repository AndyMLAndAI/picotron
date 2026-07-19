"""Verify text-only SFT/DPO against Qwen3.5's multimodal HF model class."""

from _hf_model_verification import ModelVerificationSpec, main


SPEC = ModelVerificationSpec(
    family="Qwen3.5",
    model_id="Qwen/Qwen3.5-9B-Base",
    expected_model_type="qwen3_5",
    generic_causal_lm=False,
    notes=(
        "Official Qwen3.5 checkpoints are multimodal. Its text-only forward path "
        "returns causal-LM logits, so Picotron's trainer contract applies; however "
        "generic load_model() uses AutoModelForCausalLM, so this verifier loads "
        "the checkpoint through AutoModelForMultimodalLM instead."
    ),
    loader_kind="multimodal_lm",
)


if __name__ == "__main__":
    main(SPEC)
