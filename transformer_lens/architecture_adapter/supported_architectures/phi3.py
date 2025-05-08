"""Phi3 architecture adapter."""

from typing import cast

import einops
import torch

from transformer_lens.architecture_adapter.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_phi3_weights(phi, cfg: HookedTransformerConfig):
    state_dict = {}
    state_dict["embed.W_E"] = phi.model.embed_tokens.weight

    # Some models with this architecture use Grouped Query Attention, and so for these we need to modify
    # the state dict keys for the K/V attention weight/biases, prepending "_" to the key names.
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""
    # need a cast since MyPy isn't smart enough to realize that using_gqa implies n_key_value_heads is not None
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = phi.model.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

        W = phi.model.layers[l].self_attn.qkv_proj.weight
        q_dim = cfg.n_heads * cfg.d_head
        kv_dim = n_kv_heads * cfg.d_head
        W_Q, W_K, W_V = W.split([q_dim, kv_dim, kv_dim], dim=0)

        W_Q = einops.rearrange(
            W_Q, "(n_head d_head) d_model -> n_head d_model d_head", n_head=cfg.n_heads
        )
        W_K = einops.rearrange(
            W_K, "(n_kv_head d_head) d_model -> n_kv_head d_model d_head", n_kv_head=n_kv_heads
        )
        W_V = einops.rearrange(
            W_V, "(n_kv_head d_head) d_model -> n_kv_head d_model d_head", n_kv_head=n_kv_heads
        )
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
        )

        W_O = phi.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(
            W_O, "d_model (n_head d_head) -> n_head d_head d_model", n_head=cfg.n_heads
        )

        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = phi.model.layers[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

        W = phi.model.layers[l].mlp.gate_up_proj.weight.T
        W_gate, W_in = torch.tensor_split(W, 2, dim=1)
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.W_gate"] = W_gate
        state_dict[f"blocks.{l}.mlp.W_out"] = phi.model.layers[l].mlp.down_proj.weight.T

    state_dict["ln_final.w"] = phi.model.norm.weight

    state_dict["unembed.W_U"] = phi.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


class Phi3ArchitectureAdapter(ArchitectureConversion):
    """Architecture adapter for Phi3 models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Phi3 architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.field_set = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "model.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "model.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "model.layers.{i}.self_attn.q_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "model.layers.{i}.self_attn.k_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "model.layers.{i}.self_attn.v_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.dense.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "model.layers.{i}.self_attn.dense.bias",
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.fc1.weight",
                "blocks.{i}.mlp.b_in": "model.layers.{i}.mlp.fc1.bias",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.fc2.weight",
                "blocks.{i}.mlp.b_out": "model.layers.{i}.mlp.fc2.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "model.final_layernorm.weight",
                "ln_final.b": "model.final_layernorm.bias",
            }
        )
