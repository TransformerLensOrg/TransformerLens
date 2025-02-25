import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)


class MixtralWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        number_key_value_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else 0
        super().__init__(
            {
                "embed.W_E": "embed_tokens.weight",
                "pos_embed.W_pos": "wpe.weight",
                "ln_final.w": "ln_f.weight",
                "unembed.W_U": "lm_head.weight.T",
                "unembed.b_U": torch.zeros(cfg.d_vocab),
                "blocks": (
                    "layers",
                    WeightConversionSet(
                        {
                            "ln1.w": "input_layernorm.weight",
                            "attn.W_Q": (
                                "self_attn.q_proj.weight",
                                RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_heads),
                            ),
                            "attn._W_K": (
                                "self_attn.k_proj.weight",
                                RearrangeWeightConversion(
                                    "(n h) m->n m h", n=number_key_value_heads
                                ),
                            ),
                            "attn._W_V": (
                                "self_attn.v_proj.weight",
                                RearrangeWeightConversion(
                                    "(n h) m->n m h", n=number_key_value_heads
                                ),
                            ),
                            "attn.W_O": (
                                "self_attn.o_proj.weight",
                                RearrangeWeightConversion("m (n h)->n h m", n=cfg.n_heads),
                            ),
                            "attn.b_O": torch.zeros(cfg.d_model),
                            "attn.b_Q": torch.zeros(cfg.n_heads, cfg.d_head),
                            "attn._b_K": torch.zeros(number_key_value_heads, cfg.d_head),
                            "attn._b_V": torch.zeros(number_key_value_heads, cfg.d_head),
                            "ln2.w": "post_attention_layernorm.weight",
                            "mlp.W_gate.weight": "block_sparse_moe.gate.weight",
                            "mlp.experts": (
                                "block_sparse_moe.experts",
                                WeightConversionSet(
                                    {
                                        "W_in.weight": "w3.weight",
                                        "W_gate.weight": "w1.weight",
                                        "W_out.weight": "w2.weight",
                                    }
                                ),
                            ),
                        }
                    ),
                ),
            }
        )
