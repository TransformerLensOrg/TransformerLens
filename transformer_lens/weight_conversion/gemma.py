import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    BaseWeightConversion,
    FIELD_SET,
    ArithmeticWeightConversion,
    OperationTypes,
    RearrangeWeightConversion,
    WeightConversionSet,
)

class GemmaWeightNormalizationConversion(BaseWeightConversion):
    def convert(self, input_value):
        return  input_value.float() + torch.ones_like(input_value, dtype=torch.float32)


class GemmaWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:

        super().__init__(
            {
                "ln_final.w": (
                    "model.norm.weight",
                    GemmaWeightNormalizationConversion(),
                ),
                "embed.W_E": (
                    "model.embed_tokens.weight",
                    ArithmeticWeightConversion(
                        OperationTypes.MULTIPLICATION,
                        torch.tensor(cfg.d_model**0.5, dtype=cfg.dtype),
                    ),
                ),
                "unembed.W_U": "model.lm_head.weight.T",
                "unembed.b_U": torch.zeros(cfg.d_vocab),
                "blocks": ("model.layers", self.blocks_conversions(cfg)),
            }
        )

    def blocks_conversions(self, cfg: HookedTransformerConfig) -> WeightConversionSet:
        number_key_value_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else 0
        laynorm_conversion = {}
        if cfg.use_normalization_before_and_after:
            laynorm_conversion = self.normalization_before_and_after_conversions()
        else:
            laynorm_conversion = self.standard_normalization_conversions()

        return WeightConversionSet(
            {
                "mlp.W_in": "mlp.up_proj.weight.T", 
                "mlp.W_gate": "mlp.gate_proj.weight.T",
                "mlp.b_in": torch.zeros(cfg.d_mlp, dtype=cfg.dtype),
                "mlp.W_out": "mlp.down_proj.weight.T",
                "mlp.b_out": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                "attn.b_Q": torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype),
                "attn._b_K": torch.zeros(number_key_value_heads, cfg.d_head, dtype=cfg.dtype),
                "attn._b_V": torch.zeros(number_key_value_heads, cfg.d_head, dtype=cfg.dtype),
                "attn.b_O": torch.zeros(cfg.d_model, dtype=cfg.dtype),
                "ln1.w": (
                    "input_layernorm.weight",
                    GemmaWeightNormalizationConversion(),
                ),
                "attn.W_Q": (
                    "self_attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_heads),
                ),
                "attn._W_K": (
                    "self_attn.k_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_key_value_heads),
                ),
                "attn._W_V": (
                    "self_attn.v_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_key_value_heads),
                ),
                "attn.W_O": (
                    "self_attn.o_proj.weight",
                    RearrangeWeightConversion("m (n h)->n h m", n=cfg.n_heads),
                ),
            }.update(laynorm_conversion)
        )

    def normalization_before_and_after_conversions(self) -> FIELD_SET:
        return {
            "ln1_post.w": (
                "post_attention_layernorm.weight",
                GemmaWeightNormalizationConversion(),
            ),
            "ln2.w": (
                "pre_feedforward_layernorm.weight",
                GemmaWeightNormalizationConversion(),
            ),
            "ln2_post.w": (
                "post_feedforward_layernorm.weight",
                GemmaWeightNormalizationConversion(),
            ),
        }

    def standard_normalization_conversions(self) -> FIELD_SET:
        return {
            "ln2.w": (
                "pre_feedforward_layernorm.weight",
                GemmaWeightNormalizationConversion()
            )
        }
