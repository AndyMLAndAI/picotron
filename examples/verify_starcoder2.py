"""Config and opt-in SFT/DPO smoke verification for StarCoder2-3B."""

from _hf_model_verification import ModelVerificationSpec, main


SPEC = ModelVerificationSpec(
    family="StarCoder2",
    model_id="bigcode/starcoder2-3b",
    expected_model_type="starcoder2",
    generic_causal_lm=True,
    notes="Text-only CausalLM with GQA and sliding-window attention; uses the HF fallback.",
)


if __name__ == "__main__":
    main(SPEC)
