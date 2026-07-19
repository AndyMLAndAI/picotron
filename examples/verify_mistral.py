"""Config and opt-in SFT/DPO smoke verification for Mistral-7B-v0.1."""

from _hf_model_verification import ModelVerificationSpec, main


SPEC = ModelVerificationSpec(
    family="Mistral",
    model_id="mistralai/Mistral-7B-v0.1",
    expected_model_type="mistral",
    generic_causal_lm=True,
    notes="The smallest official base Mistral is 7B; it is a GQA sliding-window CausalLM.",
)


if __name__ == "__main__":
    main(SPEC)
