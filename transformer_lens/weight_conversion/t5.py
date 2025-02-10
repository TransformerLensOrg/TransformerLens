from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)


class T5WeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "embed.W_E": "encoder.embed_tokens.weight",
                "unembed.W_U": "encoder.embed_tokens.weight.T",
                "encoder.0.attn.rel_pos_bias.weight": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                "encoder": (
                    "encoder.block",
                    WeightConversionSet(
                        {
                            "attn.W_Q": (
                                "layer.0.SelfAttention.q.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "attn.W_K": (
                                "layer.0.SelfAttention.k.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "attn.W_V": (
                                "layer.0.SelfAttention.v.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "attn.W_O": (
                                "layer.0.SelfAttention.o.weight",
                                RearrangeWeightConversion("m (i h) -> i h m", i=cfg.n_heads),
                            ),
                            "ln1.w": "layer.0.layer_norm.weight",
                            "mlp.W_in": (
                                "layer.1.DenseReluDense.wi.weight",
                                RearrangeWeightConversion("mlp model -> model mlp"),
                            ),
                            "mlp.W_out": (
                                "layer.1.DenseReluDense.wo.weight",
                                RearrangeWeightConversion("model mlp -> mlp model"),
                            ),
                            "ln2.w": "layer.1.layer_norm.weight",
                        }
                    ),
                ),
                "encoder_final_ln.w": "encoder.final_layer_norm.weight",
                "decoder.0.attn.rel_pos_bias.weight": "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                "decoder": (
                    "decoder.block",
                    WeightConversionSet(
                        {
                            "attn.W_Q": (
                                "layer.0.SelfAttention.q.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "attn.W_K": (
                                "layer.0.SelfAttention.k.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "attn.W_V": (
                                "layer.0.SelfAttention.v.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "attn.W_O": (
                                "layer.0.SelfAttention.o.weight",
                                RearrangeWeightConversion("m (i h) -> i h m", i=cfg.n_heads),
                            ),
                            "ln1.w": "layer.0.layer_norm.weight",
                            "cross_attn.W_Q": (
                                "layer.1.EncDecAttention.q.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "cross_attn.W_K": (
                                "layer.1.EncDecAttention.k.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "cross_attn.W_V": (
                                "layer.1.EncDecAttention.v.weight",
                                RearrangeWeightConversion("(i h) m -> i m h", i=cfg.n_heads),
                            ),
                            "cross_attn.W_O": (
                                "layer.1.EncDecAttention.o.weight",
                                RearrangeWeightConversion("m (i h) -> i h m", i=cfg.n_heads),
                            ),
                            "ln2.w": "layer.1.layer_norm.weight",
                            "mlp.W_in": (
                                "layer.2.DenseReluDense.wi.weight",
                                RearrangeWeightConversion("mlp model -> model mlp"),
                            ),
                            "mlp.W_out": (
                                "layer.2.DenseReluDense.wo.weight",
                                RearrangeWeightConversion("model mlp -> mlp model"),
                            ),
                            "ln3.w": "layer.2.layer_norm.weight",
                        }
                    ),
                ),
                "decoder_final_ln.w": "decoder.final_layer_norm.weight",
            }
        )
        