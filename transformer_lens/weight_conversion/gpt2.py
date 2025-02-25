import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)


class GPT2WeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "embed.W_E": "transformer.wte.weight",
                "pos_embed.W_pos": "transformer.wpe.weight",
                "unembed.W_U": "lm_head.weight.T",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "blocks": (
                    "transformer.h",
                    WeightConversionSet(
                        {
                            "ln1.w": "ln_1.weight",
                            "ln1.b": "ln_1.bias",
                            "attn.W_Q": (
                                "attn.c_attn.weight",
                                RearrangeWeightConversion(
                                    "m (i h)->i m h",
                                    input_filter=lambda weight: torch.tensor_split(
                                        weight, 3, dim=1
                                    )[0],
                                    i=cfg.n_heads,
                                ),
                            ),
                            "attn.W_K": (
                                "attn.c_attn.weight",
                                RearrangeWeightConversion(
                                    "m (i h)->i m h",
                                    input_filter=lambda weight: torch.tensor_split(
                                        weight, 3, dim=1
                                    )[1],
                                    i=cfg.n_heads,
                                ),
                            ),
                            "attn.W_V": (
                                "attn.c_attn.weight",
                                RearrangeWeightConversion(
                                    "m (i h)->i m h",
                                    input_filter=lambda weight: torch.tensor_split(
                                        weight, 3, dim=1
                                    )[2],
                                    i=cfg.n_heads,
                                ),
                            ),
                            "attn.b_Q": (
                                "attn.c_attn.bias",
                                RearrangeWeightConversion(
                                    "(qkv index head)->qkv index head",
                                    output_filter=lambda qkv_bias: qkv_bias[0],
                                    qkv=3,
                                    index=cfg.n_heads,
                                    head=cfg.d_head,
                                ),
                            ),
                            "attn.b_K": (
                                "attn.c_attn.bias",
                                RearrangeWeightConversion(
                                    "(qkv index head)->qkv index head",
                                    output_filter=lambda qkv_bias: qkv_bias[1],
                                    qkv=3,
                                    index=cfg.n_heads,
                                    head=cfg.d_head,
                                ),
                            ),
                            "attn.b_V": (
                                "attn.c_attn.bias",
                                RearrangeWeightConversion(
                                    "(qkv index head)->qkv index head",
                                    output_filter=lambda qkv_bias: qkv_bias[2],
                                    qkv=3,
                                    index=cfg.n_heads,
                                    head=cfg.d_head,
                                ),
                            ),
                            "attn.W_O": (
                                "attn.c_proj.weight",
                                RearrangeWeightConversion(
                                    "(i h) m->i h m",
                                    i=cfg.n_heads,
                                ),
                            ),
                            "attn.b_O": "attn.c_proj.bias",
                            "ln2.w": "ln_2.weight",
                            "ln2.b": "ln_2.bias",
                            "mlp.W_in": "mlp.c_fc.weight",
                            "mlp.b_in": "mlp.c_fc.bias",
                            "mlp.W_out": "mlp.c_proj.weight",
                            "mlp.b_out": "mlp.c_proj.bias",
                        }
                    ),
                ),
            }
        )
