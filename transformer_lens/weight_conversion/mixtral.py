from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversions.conversion_utils.conversion_steps import DirectWeightConversion, ZerosWeightConversion, WeightConversionSet, RearrangeWeightConversion
from transformer_lens.weight_conversions.conversion_utils import ArchitectureConversion

class MixtralWeightConversion(ArchitectureConversion):
    
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        number_key_value_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else 0
        super().__init__({
            "embed.W_E": DirectWeightConversion("embed_tokens.weight"),
            "pos_embed.W_pos": DirectWeightConversion("wpe.weight"),
            "ln_final.w": DirectWeightConversion("ln_f.weight"),
            "unembed.W_U": DirectWeightConversion("lm_head.weight.T"),
            "unembed.b_U": ZerosWeightConversion(cfg.d_vocab),
            "blocks": WeightConversionSet("layers", {
                "ln1.w": DirectWeightConversion("input_layernorm.weight"),
                "attn.W_Q": RearrangeWeightConversion(
                    "self_attn.q_proj.weight",
                    "(n h) m->n m h",
                    n=cfg.n_heads
                ),
                "attn._W_K": RearrangeWeightConversion(
                    "self_attn.k_proj.weight",
                    "(n h) m->n m h",
                    n=number_key_value_heads
                ),
                "attn._W_V": RearrangeWeightConversion(
                    "self_attn.v_proj.weight",
                    "(n h) m->n m h",
                    n=number_key_value_heads
                ),
                "attn.W_O" : RearrangeWeightConversion(
                    "self_attn.o_proj.weight",
                    "m (n h)->n h m",
                    n=cfg.n_heads
                ),
                "attn.b_O": ZerosWeightConversion(cfg.d_model),
                "attn.b_Q": ZerosWeightConversion(cfg.n_heads, cfg.d_head),
                "attn._b_K": ZerosWeightConversion(number_key_value_heads, cfg.d_head),
                "attn._b_V": ZerosWeightConversion(number_key_value_heads, cfg.d_head),
                "ln2.w": DirectWeightConversion("post_attention_layernorm.weight"),
                "mlp.W_gate.weight": DirectWeightConversion("block_sparse_moe.gate.weight"),
                "mlp.experts": WeightConversionSet("block_sparse_moe.experts", {
                    "W_in.weight": DirectWeightConversion("w3.weight"),
                    "W_gate.weight": DirectWeightConversion("w1.weight"),
                    "W_out.weight": DirectWeightConversion("w2.weight"),
                }),
            })
        })
