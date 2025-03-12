import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)


class GPTJWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "embed.W_E": "transformer.wte.weight",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight.T",
                "unembed.b_U": "lm_head.bias",
                "blocks": (
                    "transformer.h",
                    WeightConversionSet(
                        {
                            "ln1.w": "ln_1.weight",
                            "ln1.b": "ln_1.bias",
                            "ln2.w": "ln_1.weight",
                            "ln2.b": "ln_1.bias",
                            "attn.W_Q": (
                                "attn.q_proj.weight",
                                RearrangeWeightConversion("(i h) m->i m h", i=cfg.n_heads),
                            ),
                            "attn.W_K": (
                                "attn.k_proj.weight",
                                RearrangeWeightConversion("(i h) m->i m h", i=cfg.n_heads),
                            ),
                            "attn.W_V": (
                                "attn.v_proj.weight",
                                RearrangeWeightConversion("(i h) m->i m h", i=cfg.n_heads),
                            ),
                            "attn.b_Q": torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype),
                            "attn.b_K": torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype),
                            "attn.b_V": torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype),
                            "attn.W_O": (
                                "attn.out_proj.weight",
                                RearrangeWeightConversion("m (i h)->i h m", i=cfg.n_heads),
                            ),
                            "attn.b_O": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                            "mlp.W_in": "mlp.fc_in.weight.T",
                            "mlp.b_in": "mlp.fc_in.bias",
                            "mlp.W_out": "mlp.fc_out.weight.T",
                            "mlp.b_out": "mlp.fc_out.bias",
                        }
                    ),
                ),
            },
            {
            }
        )
