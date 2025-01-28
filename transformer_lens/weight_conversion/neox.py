import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)


class NEOXWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "embed.W_E": "gpt_neox.embed_in.weight",
                "ln_final.w": "gpt_neox.final_layer_norm.weight",
                "ln_final.b": "gpt_neox.final_layer_norm.bias",
                "unembed.W_U": "embed_out.weight.T",
                "unembed.b_U": torch.zeros(cfg.d_vocab, dtype=cfg.dtype),
                "blocks": (
                    "neox.gpt_neox.layers",
                    WeightConversionSet(
                        {
                            "ln1.w": "input_layernorm.weight",
                            "ln1.b": "input_layernorm.bias",
                            "attn.W_Q": (
                                "attention.query_key_value.weight",
                                RearrangeWeightConversion(
                                    "(i qkv h) m -> qkv i m h",
                                    output_filter=lambda result: result[0],
                                    i=cfg.n_heads,
                                    qkv=3,
                                ),
                            ),
                            "attn.W_k": (
                                "attention.query_key_value.weight",
                                RearrangeWeightConversion(
                                    "(i qkv h) m -> qkv i m h",
                                    output_filter=lambda result: result[1],
                                    i=cfg.n_heads,
                                    qkv=3,
                                ),
                            ),
                            "attn.W_V": (
                                "attention.query_key_value.weight",
                                RearrangeWeightConversion(
                                    "(i qkv h) m -> qkv i m h",
                                    output_filter=lambda result: result[2],
                                    i=cfg.n_heads,
                                    qkv=3,
                                ),
                            ),
                            "attn.b_Q": (
                                "attention.query_key_value.bias",
                                RearrangeWeightConversion(
                                    "(index qkv head) -> qkv index head",
                                    output_filter=lambda result: result[0],
                                    qkv=3,
                                    index=cfg.n_heads,
                                    head=cfg.d_head,
                                ),
                            ),
                            "attn.b_K": (
                                "attention.query_key_value.bias",
                                RearrangeWeightConversion(
                                    "(index qkv head) -> qkv index head",
                                    output_filter=lambda result: result[1],
                                    qkv=3,
                                    index=cfg.n_heads,
                                    head=cfg.d_head,
                                ),
                            ),
                            "attn.b_V": (
                                "attention.query_key_value.bias",
                                RearrangeWeightConversion(
                                    "(index qkv head) -> qkv index head",
                                    output_filter=lambda result: result[2],
                                    qkv=3,
                                    index=cfg.n_heads,
                                    head=cfg.d_head,
                                ),
                            ),
                            "attn.W_O": (
                                "attention.dense.weight",
                                RearrangeWeightConversion("m (i h) -> i h m", i=cfg.n_heads),
                            ),
                            "attn.b_O": "attention.dense.bias",
                            "ln2.w": "post_attention_layernorm.weight",
                            "ln2.b": "post_attention_layernorm.bias",
                            "mlp.W_in": "mlp.dense_h_to_4h.weight.T",
                            "mlp.b_in": "mlp.dense_h_to_4h.bias",
                            "mlp.W_out": "mlp.dense_4h_to_h.weight.T",
                            "mlp.b_out": "mlp.dense_4h_to_h.bias",
                        }
                    ),
                ),
            }
        )
