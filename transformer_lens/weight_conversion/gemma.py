import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_gemma_weights(gemma, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Gemma Models scale embeddings by multiplying by sqrt(d_model), use hidden state type to match
    # HF implementation
    # state_dict["embed.W_E"] = gemma.model.embed_tokens.weight * torch.tensor(
    #     cfg.d_model**0.5, dtype=cfg.dtype
    # )

    # Gemma has no biases anywhere
    for l in range(cfg.n_layers):
        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        # state_dict[f"blocks.{l}.ln1.w"] = gemma.model.layers[
        #     l
        # ].input_layernorm.weight.float() + torch.ones_like(
        #     gemma.model.layers[l].input_layernorm.weight, dtype=torch.float32
        # )
        # if cfg.use_normalization_before_and_after:
        #     # Only applies for Gemma 2
        #     state_dict[f"blocks.{l}.ln1_post.w"] = gemma.model.layers[
        #         l
        # TODO Need to fix this so it pulls in the proper shape
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                gemma.model.layers[l].input_layernorm.weight, dtype=torch.float32
            )

        # W_Q = gemma.model.layers[l].self_attn.q_proj.weight
        # W_K = gemma.model.layers[l].self_attn.k_proj.weight
        # W_V = gemma.model.layers[l].self_attn.v_proj.weight
        # W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        # W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        # W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        # state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        # state_dict[f"blocks.{l}.attn._W_K"] = W_K
        # state_dict[f"blocks.{l}.attn._W_V"] = W_V

        # state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        # state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
        #     cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        # )
        # state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
        #     cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        # )

        # W_O = gemma.model.layers[l].self_attn.o_proj.weight
        # W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        # state_dict[f"blocks.{l}.attn.W_O"] = W_O

        # state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        if not cfg.use_normalization_before_and_after:
            # Only applies for Gemma 1. Confusingly post_attention_layernorm is applied to mlp_input in Gemma 1 and attn_out in Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = gemma.model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                gemma.model.norm.weight, dtype=torch.float32
            )
        else:
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = gemma.model.layers[
                l
            ].pre_feedforward_layernorm.weight.float() + torch.ones_like(
                gemma.model.layers[l].pre_feedforward_layernorm.weight, dtype=torch.float32
            )
            state_dict[f"blocks.{l}.ln2_post.w"] = gemma.model.layers[
                l
            ].post_feedforward_layernorm.weight.float() + torch.ones_like(
                gemma.model.layers[l].post_feedforward_layernorm.weight, dtype=torch.float32
            )

        state_dict[f"blocks.{l}.mlp.W_in"] = gemma.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = gemma.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = gemma.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
    # state_dict["ln_final.w"] = gemma.model.norm.weight.float() + torch.ones_like(
    #     gemma.model.norm.weight, dtype=torch.float32
    # )

    # state_dict["unembed.W_U"] = gemma.lm_head.weight.T
    # state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils import ArchitectureConversion
from transformer_lens.weight_conversion.conversion_utils.conversion_steps import (
    FIELD_SET,
    ArithmeticWeightConversion,
    CallableWeightConversion,
    DirectWeightConversion,
    OperationTypes,
    RearrangeWeightConversion,
    WeightConversionSet,
)


class GemmalWeightConversion(ArchitectureConversion):
    def __init__(self, cfg: HookedTransformerConfig) -> None:

        super().__init__(
            {
                "ln_final.w": (
                    "gemma.model.norm.weight",
                    CallableWeightConversion(
                        lambda weight: weight.float() + torch.ones_like(weight, dtype=torch.float32)
                    ),
                ),
                "embed.W_E": (
                    "model.embed_tokens.weight",
                    ArithmeticWeightConversion(
                        OperationTypes.MULTIPLICATION,
                        torch.tensor(cfg.d_model**0.5, dtype=cfg.dtype),
                    ),
                ),
                "unembed.W_U": ("lm_head.weight.T", DirectWeightConversion()),
                "unembed.b_U": torch.zeros(cfg.d_vocab),
                "blocks": ("layers", self.blocks_conversions(cfg)),
            }
        )

    def blocks_conversions(self, cfg: HookedTransformerConfig) -> WeightConversionSet:
        number_key_value_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else 0
        if cfg.use_normalization_before_and_after:
            laynorm_conversion = self.normalization_before_and_after_conversions(cfg)
        else:
            laynorm_conversion = self.standard_normalization_conversions(cfg)

        return WeightConversionSet(
            {
                "ln1.w": (
                    "input_layernorm.weight",
                    CallableWeightConversion(
                        lambda weight: weight.float() + torch.ones_like(weight, dtype=torch.float32)
                    ),
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
                "attn.b_Q": torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype),
                "attn._b_K": torch.zeros(number_key_value_heads, cfg.d_head, dtype=cfg.dtype),
                "attn._b_V": torch.zeros(number_key_value_heads, cfg.d_head, dtype=cfg.dtype),
                "attn.b_O": torch.zeros(cfg.d_model, dtype=cfg.dtype),
            }
            | laynorm_conversion
        )

    def normalization_before_and_after_conversions(self, cfg: HookedTransformerConfig) -> FIELD_SET:
        return {
            "ln1_post.w": (
                "post_attention_layernorm.weight",
                CallableWeightConversion(
                    lambda weight: weight.float() + torch.ones_like(weight, dtype=torch.float32)
                ),
            )
        }

    def standard_normalization_conversions(self, cfg: HookedTransformerConfig) -> FIELD_SET:
        return {
            "ln2.w": (
                "pre_feedforward_layernorm.weight",
                CallableWeightConversion(
                    lambda weight, model: weight.float() + torch.ones_like(model.norm.weight, dtype=torch.float32)
                )
            )
        }
