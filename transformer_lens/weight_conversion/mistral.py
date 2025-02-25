import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)


class MistralWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        number_key_value_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else 0
        super().__init__(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "lm_head.weight.T",
                "unembed.b_U": torch.zeros(cfg.d_vocab, dtype=cfg.dtype),
                "blocks": (
                    "model.layers",
                    WeightConversionSet(
                        {
                            "ln1.w": "input_layernorm.weight",
                            "attn.W_Q": (
                                "self_attn.q_proj.weight",
                                RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_heads),
                            ),
                            "attn.W_K": (
                                "self_attn.k_proj.weight",
                                RearrangeWeightConversion(
                                    "(n h) m->n m h", n=number_key_value_heads
                                ),
                            ),
                            "attn.W_V": (
                                "self_attn.v_proj.weight",
                                RearrangeWeightConversion(
                                    "(n h) m->n m h", n=number_key_value_heads
                                ),
                            ),
                            "attn.b_Q": torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype),
                            "attn.b_K": torch.zeros(
                                number_key_value_heads, cfg.d_head, dtype=cfg.dtype
                            ),
                            "attn.b_V": torch.zeros(
                                number_key_value_heads, cfg.d_head, dtype=cfg.dtype
                            ),
                            "attn.W_O": (
                                "self_attn.o_proj.weight",
                                RearrangeWeightConversion("m (n h)->n h m", n=cfg.n_heads),
                            ),
                            "attn.b_O": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                            "ln2.w": "post_attention_layernorm.weight",
                            "mlp.W_in": "mlp.up_proj.weight.T",
                            "mlp.W_gate": "mlp.gate_proj.weight.T",
                            "mlp.b_in": torch.zeros(cfg.d_mlp, dtype=cfg.dtype),
                            "mlp.W_out": "mlp.down_proj.weight.T",
                            "mlp.b_out": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                        }
                    ),
                ),
            }
        )
