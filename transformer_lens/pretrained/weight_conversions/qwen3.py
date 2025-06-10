from typing import Any

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_qwen3_weights(qwen: Any, cfg: HookedTransformerConfig):
    """Convert Qwen3 weights to TransformerLens format."""
    state_dict = {}

    state_dict["embed.W_E"] = qwen.model.embed_tokens.weight

    if cfg.n_key_value_heads is None:
        gqa_uscore = ""
        n_kv_heads = cfg.n_heads
    else:
        gqa_uscore = "_"
        n_kv_heads = cfg.n_key_value_heads

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = qwen.model.layers[l].input_layernorm.weight

        W_Q = qwen.model.layers[l].self_attn.q_proj.weight
        W_K = qwen.model.layers[l].self_attn.k_proj.weight
        W_V = qwen.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        # Load weights into RMSNorm modules
        state_dict[f"blocks.{l}.attn.q_norm.w"] = qwen.model.layers[l].self_attn.q_norm.weight
        state_dict[f"blocks.{l}.attn.k_norm.w"] = qwen.model.layers[l].self_attn.k_norm.weight

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = qwen.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = qwen.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = qwen.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = qwen.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = qwen.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = qwen.model.norm.weight

    state_dict["unembed.W_U"] = qwen.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
