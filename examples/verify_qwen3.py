"""Config and opt-in SFT/DPO smoke verification for Qwen3-0.6B."""

from _hf_model_verification import ModelVerificationSpec, main


SPEC = ModelVerificationSpec(
    family="Qwen3",
    model_id="Qwen/Qwen3-0.6B",
    expected_model_type="qwen3",
    generic_causal_lm=True,
    notes="Standard text-only CausalLM with GQA and RoPE; load_model may use Unsloth.",
)


if __name__ == "__main__":
    main(SPEC)
