"""Config and opt-in SFT/DPO smoke verification for Qwen2.5-0.5B."""

from _hf_model_verification import ModelVerificationSpec, main


SPEC = ModelVerificationSpec(
    family="Qwen2.5",
    model_id="Qwen/Qwen2.5-0.5B",
    expected_model_type="qwen2",
    generic_causal_lm=True,
    notes="Standard text-only CausalLM with GQA and RoPE; load_model may use Unsloth.",
)


if __name__ == "__main__":
    main(SPEC)
