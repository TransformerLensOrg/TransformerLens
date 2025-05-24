"""Phi-3 architecture adapter."""

from typing import Any, cast

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)


def convert_phi3_weights(phi: Any, cfg: HookedTransformerConfig):
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

        w = phi.model.layers[l].self_attn.qkv_proj.weight
        q_dim = cfg.n_heads * cfg.d_head
        kv_dim = n_kv_heads * cfg.d_head
        w_q, w_k, w_v = w.split([q_dim, kv_dim, kv_dim], dim=0)

        w_q = einops.rearrange(
            w_q, "(n_head d_head) d_model -> n_head d_model d_head", n_head=cfg.n_heads
        )
        w_k = einops.rearrange(
            w_k, "(n_kv_head d_head) d_model -> n_kv_head d_model d_head", n_kv_head=n_kv_heads
        )
        w_v = einops.rearrange(
            w_v, "(n_kv_head d_head) d_model -> n_kv_head d_model d_head", n_kv_head=n_kv_heads
        )
        state_dict[f"blocks.{l}.attn.W_Q"] = w_q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = w_k
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = w_v

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

        w_o = phi.model.layers[l].self_attn.o_proj.weight
        w_o = einops.rearrange(
            w_o, "d_model (n_head d_head) -> n_head d_head d_model", n_head=cfg.n_heads
        )

        state_dict[f"blocks.{l}.attn.W_O"] = w_o
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = phi.model.layers[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

        w = phi.model.layers[l].mlp.gate_up_proj.weight.T
        w_gate, w_in = torch.tensor_split(w, 2, dim=1)
        state_dict[f"blocks.{l}.mlp.W_in"] = w_in
        state_dict[f"blocks.{l}.mlp.W_gate"] = w_gate
        state_dict[f"blocks.{l}.mlp.W_out"] = phi.model.layers[l].mlp.down_proj.weight.T

    state_dict["ln_final.w"] = phi.model.norm.weight

    state_dict["unembed.W_U"] = phi.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


class Phi3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Phi-3 models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Phi-3 architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeWeightConversion("(n_head d_head) d_model -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("transformer.wte", EmbeddingBridge),  # Word token embeddings
            "blocks": (
                "transformer.h",  # Base path for blocks
                {
                    "ln1": ("ln_1", LayerNormBridge),  # Pre-attention layer norm
                    "ln2": ("ln_2", LayerNormBridge),  # Pre-MLP layer norm
                    "attn": ("attn", AttentionBridge),  # Full attention module
                    "attn.c_attn": ("attn.c_attn", AttentionBridge),  # QKV projection
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                },
            ),
            "ln_final": ("transformer.ln_f", LayerNormBridge),  # Final layer norm
            "unembed": ("lm_head", UnembeddingBridge),  # Language model head
        }
