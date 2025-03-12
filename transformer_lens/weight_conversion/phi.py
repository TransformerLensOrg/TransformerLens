from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)


class PhiWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks": (
                    "model.layers",
                    WeightConversionSet(
                        {
                            "ln1.w": "input_layernorm.weight",
                            "ln1.b": "input_layernorm.bias",
                            "attn.W_Q": (
                                "self_attn.q_proj.weight",
                                RearrangeWeightConversion(
                                    "(n_head d_head) d_model -> n_head d_model d_head",
                                    n_head=cfg.n_heads,
                                ),
                            ),
                            "attn.W_K": (
                                "self_attn.k_proj.weight",
                                RearrangeWeightConversion(
                                    "(n_head d_head) d_model -> n_head d_model d_head",
                                    n_head=cfg.n_heads,
                                ),
                            ),
                            "attn.W_V": (
                                "self_attn.v_proj.weight",
                                RearrangeWeightConversion(
                                    "(n_head d_head) d_model -> n_head d_model d_head",
                                    n_head=cfg.n_heads,
                                ),
                            ),
                            "attn.b_Q": (
                                "self_attn.q_proj.bias",
                                RearrangeWeightConversion(
                                    "(n_head d_head) -> n_head d_head",
                                    n_head=cfg.n_heads,
                                ),
                            ),
                            "attn.b_K": (
                                "self_attn.k_proj.bias",
                                RearrangeWeightConversion(
                                    "(n_head d_head) -> n_head d_head",
                                    n_head=cfg.n_heads,
                                ),
                            ),
                            "attn.b_V": (
                                "self_attn.v_proj.bias",
                                RearrangeWeightConversion(
                                    "(n_head d_head) -> n_head d_head",
                                    n_head=cfg.n_heads,
                                ),
                            ),
                            "attn.W_O": (
                                "self_attn.dense.weight",
                                RearrangeWeightConversion(
                                    "d_model (n_head d_head) -> n_head d_head d_model",
                                    n_head=cfg.n_heads,
                                ),
                            ),
                            "attn.b_O": "self_attn.dense.bias",
                            "ln2.w": "input_layernorm.weight",
                            "ln2.b": "input_layernorm.bias",
                            "mlp.W_in": "mlp.fc1.weight.T",
                            "mlp.b_in": "mlp.fc1.bias",
                            "mlp.W_out": "mlp.fc2.weight.T",
                            "mlp.b_out": "mlp.fc2.bias",
                        }
                    ),
                ),
                "ln_final.w": "model.final_layernorm.weight",
                "ln_final.b": "model.final_layernorm.bias",
                "unembed.W_U": "lm_head.weight.T",
                "unembed.b_U": "lm_head.bias",
            },
            {
            }
        )
