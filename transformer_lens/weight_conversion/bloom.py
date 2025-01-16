from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
    CallableWeightConversion,
)

class BloomWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "embed.W_E": "transformer.word_embeddings.weight",
                "embed.ln.w": "transformer.word_embeddings_layernorm.weight",
                "embed.ln.b": "transformer.word_embeddings_layernorm.bias",
                "unembed.W_U": "lm_head.weight.T",
                "ln_final.b": "transformer.ln_f.bias",
                "blocks": ("transformer.h", WeightConversionSet({
                    "ln1.w": "input_layernorm.weight",
                    "ln1.b": "input_layernorm.bias",
                    "attn.W_Q": (
                        "self_attention.query_key_value.weight.T",
                        RearrangeWeightConversion(
                            "m n h ->n m h",
                            input_filter=lambda weight: weight.reshape(cfg.d_model, cfg.n_heads, 3, cfg.d_head)[..., 0, :],
                            n=cfg.n_heads,
                        )
                    ),
                    "attn.W_K": (
                        "self_attention.query_key_value.weight.T",
                        RearrangeWeightConversion(
                            "m n h ->n m h",
                            input_filter=lambda weight: weight.reshape(cfg.d_model, cfg.n_heads, 3, cfg.d_head)[..., 1, :],
                            n=cfg.n_heads,
                        )
                    ),
                    "attn.W_V": (
                        "self_attention.query_key_value.weight.T",
                        RearrangeWeightConversion(
                            "m n h ->n m h",
                            input_filter=lambda weight: weight.reshape(cfg.d_model, cfg.n_heads, 3, cfg.d_head)[..., 2, :],
                            n=cfg.n_heads,
                        )
                    ),
                    "attn.b_Q": (
                        "self_attention.query_key_value.bias",
                        CallableWeightConversion(
                            convert_callable=lambda weight: weight.reshape(cfg.n_heads, 3, cfg.d_head)[..., 0, :],
                        )
                    ),
                    "attn.b_K": (
                        "self_attention.query_key_value.bias",
                        CallableWeightConversion(
                            convert_callable=lambda weight: weight.reshape(cfg.n_heads, 3, cfg.d_head)[..., 1, :],
                        )
                    ),
                    "attn.b_V": (
                        "self_attention.query_key_value.bias",
                        CallableWeightConversion(
                            convert_callable=lambda weight: weight.reshape(cfg.n_heads, 3, cfg.d_head)[..., 2, :],
                        )
                    ),
                    "attn.W_O": (
                        "self_attention.dense.weight.T",
                        RearrangeWeightConversion("(n h) m->n h m", n=cfg.n_heads)
                    ),
                    "attn.b_O": "self_attention.dense.bias",
                    "ln2.w": "post_attention_layernorm.weight",
                    "ln2.b": "post_attention_layernorm.bias",
                    "mlp.W_in": "mlp.dense_h_to_4h.weight.T",
                    "mlp.b_in": "mlp.dense_h_to_4h.bias",
                    "mlp.W_out": "mlp.dense_4h_to_h.weight.T",
                    "mlp.b_out": "mlp.dense_4h_to_h.bias",
                }))
            }
        )
