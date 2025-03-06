import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    FIELD_SET,
    ArithmeticWeightConversion,
    BaseWeightConversion,
    OperationTypes,
    RearrangeWeightConversion,
    WeightConversionSet,
)


class GemmaWeightNormalizationConversion(BaseWeightConversion):
    def convert(self, input_value, *full_context):
        return input_value.float() + torch.ones_like(input_value, dtype=torch.float32)

    def __repr__(self):
        return "Is an addition of 1 to the input tensor\n"


class GemmaWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "unembed.W_U": "lm_head.weight.T",
                "unembed.b_U": torch.zeros(cfg.d_vocab),
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
                "blocks": ("model.layers", self.blocks_conversions(cfg)),
            }
        )

    def blocks_conversions(self, cfg: HookedTransformerConfig) -> WeightConversionSet:
        number_key_value_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else 0
        base_conversion_dict = {
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
        }

        laynorm_conversion = {}
        if cfg.use_normalization_before_and_after:
            laynorm_conversion = self.normalization_before_and_after_conversions()
        else:
            laynorm_conversion = self.standard_normalization_conversions()

        return WeightConversionSet({**base_conversion_dict, **laynorm_conversion})

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
        return {"ln2.w": ("post_attention_layernorm.weight", GemmaWeightNormalizationConversion())}
