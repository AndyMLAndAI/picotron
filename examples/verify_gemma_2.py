"""Config and opt-in SFT/DPO smoke verification for Gemma 2 2B."""

from _hf_model_verification import ModelVerificationSpec, main


SPEC = ModelVerificationSpec(
    family="Gemma 2",
    model_id="google/gemma-2-2b",
    expected_model_type="gemma2",
    generic_causal_lm=True,
    notes="Text-only Gemma 2 CausalLM; requires accepting Google's Hugging Face license first.",
)


if __name__ == "__main__":
    main(SPEC)
