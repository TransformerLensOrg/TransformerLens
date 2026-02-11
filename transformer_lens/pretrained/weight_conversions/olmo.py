from typing import cast

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_olmo_weights(olmo, cfg: HookedTransformerConfig):
    state_dict = {}

    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)

    assert cfg.d_mlp is not None # keep mypy happy

    base_model = olmo.model

    state_dict["embed.W_E"] = base_model.embed_tokens.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = base_model.layers[l].post_attention_layernorm.weight

        W_Q = base_model.layers[l].self_attn.q_proj.weight
        W_K = base_model.layers[l].self_attn.k_proj.weight
        W_V = base_model.layers[l].self_attn.v_proj.weight

        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        if cfg.use_qk_norm:
            state_dict[f"blocks.{l}.attn.q_norm.w"] = base_model.layers[l].self_attn.q_norm.weight
            state_dict[f"blocks.{l}.attn.k_norm.w"] = base_model.layers[l].self_attn.k_norm.weight

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=W_Q.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype, device=W_K.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype, device=W_V.device
        )

        W_O = base_model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=W_O.device
        )

        state_dict[f"blocks.{l}.ln2.w"] = base_model.layers[l].post_feedforward_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = base_model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = base_model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=base_model.layers[l].mlp.up_proj.weight.device
        )

        state_dict[f"blocks.{l}.mlp.W_out"] = base_model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=base_model.layers[l].mlp.down_proj.weight.device
        )

    state_dict["ln_final.w"] = base_model.norm.weight
    state_dict["unembed.W_U"] = olmo.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(
        cfg.d_vocab, dtype=cfg.dtype, device=olmo.lm_head.weight.device
    )

    return state_dict
