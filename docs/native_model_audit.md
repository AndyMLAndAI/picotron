# Native model architecture audit

## Conclusion

`ToyDecoderModel` is Picotron's only native, from-scratch causal-LM
architecture.  `picotron.models` exports only that class, and the CLI creates
only that class.  The other `nn` classes are decoder components, not alternate
models.  `picotron_sft` and `picotron_dpo` can instead accept externally
loaded Hugging Face-compatible models; that is a separate library-integration
path rather than a second Picotron architecture.

## Feature wiring

| Feature | Configuration | Native-model path | Audit result |
| --- | --- | --- | --- |
| RMSNorm and dense SwiGLU | built-in defaults | `DecoderBlock` and `ToyDecoderModel` | Wired. |
| RoPE | `position_embedding_type: rope`, `rope_theta` | Each non-NoPE `DecoderBlock` creates attention with `use_rope=True` | Wired; default position scheme. |
| Learned positions | `position_embedding_type: learned` | `ToyDecoderModel.position_embeddings` | Wired as the alternative to RoPE. |
| Per-layer NoPE | `nope_layers` | The selected layers are built with `use_rope=False` | Wired.  With learned positions selected, all attention layers already omit RoPE, so `nope_layers` has no additional effect. |
| GQA | `num_key_value_heads` (and validated `attention_type: gqa`) | `CausalSelfAttention` creates smaller K/V projections, then expands K/V heads for scores | Wired. |
| Sliding-window attention | `sliding_window_size` | `CausalSelfAttention._causal_mask` combines window and causal masks | Wired for the eager MHA/GQA path.  Schema validation intentionally excludes combining it with MLA. |
| MoE FFN | `moe_config` | Each `DecoderBlock` replaces dense `SwiGLU` with `MoEFeedForward`; the model collects auxiliary losses | Wired for every decoder block.  The pretraining loop adds the collected auxiliary loss to cross entropy. |
| MLA | `attention_type: mla`, `kv_lora_rank` | `_build_attention` selects `MultiHeadLatentAttention` for every block | Wired as an alternate attention module.  It is intentionally a simplified MLA: it reconstructs full K/V for each forward pass and uses conventional RoPE, not DeepSeek's decoupled RoPE subspace. |

The existing focused tests exercise construction and forward paths for RoPE,
NoPE, GQA/sliding-window, MoE, and MLA.  MLA and MoE tests also exercise a
short training-loss decrease.  This audit itself changes documentation only.

## Important limitations found

1. `ToyDecoderModel.forward` accepts only complete `input_ids`; it has no
   `past_key_values`/`use_cache` generation API.  `MultiHeadLatentAttention`
   records a detached `last_kv_cache` for inspection, but that cache is not
   accepted by a later forward call.  Therefore the MLA module's compressed
   cache representation is testable, but its practical autoregressive
   KV-cache reuse is not yet exposed through the native model.
2. `attention_type` is a useful validation/selection field for MLA, but the
   eager attention implementation determines MHA versus GQA from
   `num_key_value_heads`.  Supplying fewer K/V heads while leaving
   `attention_type: mha` still produces GQA.  This is not disconnected code,
   but the two settings are semantically redundant and could be tightened in a
   future config-cleanup task.

## Naming recommendation

The class is more capable than the name `ToyDecoderModel` suggests.  A future
compatible rename to `PicotronDecoderModel` (or `PicotronCausalLM`) would
better communicate that it is the project's configurable native decoder.
This audit deliberately does not rename it.
