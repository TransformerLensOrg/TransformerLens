import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    RepeatWeightConversion,
    WeightConversionSet,
)


class GPT2LMHeadCustomWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__({
            "embed.W_E": "model.transformer.wte.weight",
            "unembed.W_U": "model.lm_head.weight.T",
            "pos_embed.W_pos": "model.transformer.wpe.weight",
            "ln_final.w": "model.transformer.ln_f.weight",
            "ln_final.b": "model.transformer.ln_f.bias",
            "blocks": ("model.transformer.h", WeightConversionSet({
                "ln1.w": "ln_1.weight",
                "ln1.b": "ln_1.bias",
                "attn.W_Q": ("attn.q_attn.weight", RearrangeWeightConversion("m h -> i m h", i=cfg.n_heads)),
                "attn.W_K": ("attn.kv_attn.weight", RepeatWeightConversion(
                    "m h -> i m h",
                    i=cfg.n_heads,
                    input_filter=lambda weight: torch.tensor_split(weight, 2, dim=1)[0],
                )),
                "attn.W_V": ("attn.kv_attn.weight", RepeatWeightConversion(
                    "m h -> i m h",
                    i=cfg.n_heads,
                    input_filter=lambda weight: torch.tensor_split(weight, 2, dim=1)[1],
                )),
                "attn.b_Q": ("attn.q_attn.bias", RearrangeWeightConversion(
                    "(index head)-> index head",
                    index=cfg.n_heads,
                    head=cfg.d_head,
                )),
                "attn.b_K": ("attn.kv_attn.bias", RepeatWeightConversion(
                    "head -> index head",
                    index=cfg.n_heads,
                    input_filter=lambda weight: torch.tensor_split(weight, 2, dim=0)[0]
                )),
                "attn.b_V": ("attn.kv_attn.bias", RepeatWeightConversion(
                    "head -> index head",
                    index=cfg.n_heads,
                    input_filter=lambda weight: torch.tensor_split(weight, 2, dim=0)[1]
                )),
                "attn.W_O": ("attn.c_proj.weight", RearrangeWeightConversion("(i h) m->i h m", i=cfg.n_heads)),
                "attn.b_O": "attn.c_proj.bias",
                "ln2.w": "ln_2.weight",
                "ln2.b": "ln_2.bias",
                "mlp.W_in": "mlp.c_fc.weight",
                "mlp.b_in": "mlp.c_fc.bias",
                "mlp.W_out": "mlp.c_proj.weight",
                "mlp.b_out": "mlp.c_proj.bias",
            }))
        })
