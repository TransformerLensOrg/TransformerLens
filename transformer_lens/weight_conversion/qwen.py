import torch
from torch import nn

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)

class QwenWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "embed.W_E": "wte.weight",
                "ln_final.w": "ln_f.weight",
                "unembed.W_U": "lm_head.weight.T",
                "unembed.b_U": torch.zeros(cfg.d_vocab, dtype=cfg.dtype),
                "blocks": ("h", WeightConversionSet({
                    "ln1.w": "ln_1.weight",
                    "attn.W_Q": ("attn.c_attn.weight", RearrangeWeightConversion(
                        "(n h) m->n m h",
                        input_filter=lambda weight: weight.split(split_size=cfg.d_model, dim=0)[0],
                        n=cfg.n_heads
                    )),
                    "attn.W_K": ("attn.c_attn.weight", RearrangeWeightConversion(
                        "(n h) m->n m h",
                        input_filter=lambda weight: weight.split(split_size=cfg.d_model, dim=0)[1],
                        n=cfg.n_heads
                    )),
                    "attn.W_V": ("attn.c_attn.weight", RearrangeWeightConversion(
                        "(n h) m->n m h",
                        input_filter=lambda weight: weight.split(split_size=cfg.d_model, dim=0)[2],
                        n=cfg.n_heads
                    )),
                    "attn.b_Q": ("attn.c_attn.bias", RearrangeWeightConversion(
                        "(n_head d_head) -> n_head d_head",
                        input_filter=lambda weight: weight.split(split_size=cfg.d_model, dim=0)[0],
                        n=cfg.n_heads
                    )),
                    "attn.b_K": ("attn.c_attn.bias", RearrangeWeightConversion(
                        "(n_head d_head) -> n_head d_head",
                        input_filter=lambda weight: weight.split(split_size=cfg.d_model, dim=0)[1],
                        n=cfg.n_heads
                    )),
                    "attn.b_V": ("attn.c_attn.bias", RearrangeWeightConversion(
                        "(n_head d_head) -> n_head d_head",
                        input_filter=lambda weight: weight.split(split_size=cfg.d_model, dim=0)[2],
                        n=cfg.n_heads,
                    )),
                    "attn.W_O": ("attn.c_proj.weight", RearrangeWeightConversion(
                        "m (n h)->n h m",
                        n=cfg.n_heads,
                    )),
                    "attn.b_O": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                    "ln2.w": "ln_2.weight",
                    "mlp.W_in": "mlp.w1.weight.T",
                    "mlp.W_gate": "mlp.w2.weight.T",
                    "mlp.b_in": torch.zeros(cfg.d_mlp, dtype=cfg.dtype),
                    "mlp.W_out": "mlp.c_proj.weight.T",
                    "mlp.b_out": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                }))
            }
        )
        
    def get_model(self, remote_module: nn.Module) -> dict:
        """The weights for this model are in a variable named transformer

        Args:
            remote_module nn.Module: The module from hugging face

        Returns:
            dict: The model
        """
        return remote_module.transformer