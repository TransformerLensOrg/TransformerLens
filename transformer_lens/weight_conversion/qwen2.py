import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)

class Qwen2WeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "embed.W_E": "embed_tokens.weight",
                "ln_final.w": "norm.weight",
                "unembed.W_U": "lm_head.weight.T",
                "unembed.b_U": torch.zeros(cfg.d_vocab, dtype=cfg.dtype),
                "blocks": ("layers", WeightConversionSet({
                    "attn.W_Q": ("self_attn.q_proj.weight", RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_heads)),
                    "attn._W_K": ("self_attn.k_proj.weight", RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_key_value_heads)),
                    "attn._W_V": ("self_attn.v_proj.weight", RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_key_value_heads)),
                    "attn.b_Q": ("self_attn.v_proj.bias", RearrangeWeightConversion("(n_head d_head) -> n_head d_head", n=cfg.n_key_value_heads)),
                    "attn._b_K": ("self_attn.v_proj.bias", RearrangeWeightConversion("(n_head d_head) -> n_head d_head", n=cfg.n_key_value_heads)),
                    "attn._b_V": ("self_attn.v_proj.bias", RearrangeWeightConversion("(n_head d_head) -> n_head d_head", n=cfg.n_key_value_heads)),
                    "attn.W_O": ("self_attn.v_proj.bias", RearrangeWeightConversion("m (n h)->n h m", n=cfg.n_heads)),
                    "attn.b_O": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                    "ln2.w": "post_attention_layernorm.weight",
                    "mlp.W_in": "mlp.up_proj.weight.T",
                    "mlp.W_gate": "mlp.gate_proj.weight.T",
                    "mlp.b_in": torch.zeros(cfg.d_mlp, dtype=cfg.dtype),
                    "mlp.W_out": "mlp.down_proj.weight.T",
                    "mlp.b_out": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                }))
            }
        )
