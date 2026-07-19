"""Config and opt-in SFT/DPO smoke verification for Llama 3.2-1B."""

from _hf_model_verification import ModelVerificationSpec, main


SPEC = ModelVerificationSpec(
    family="Llama 3.2",
    model_id="meta-llama/Llama-3.2-1B",
    expected_model_type="llama",
    generic_causal_lm=True,
    notes="Standard GQA CausalLM; requires accepting Meta's Hugging Face license first.",
)


if __name__ == "__main__":
    main(SPEC)
