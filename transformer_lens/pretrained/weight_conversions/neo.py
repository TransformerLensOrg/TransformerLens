import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_neo_weights(neo, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = neo.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = neo.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = neo.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = neo.transformer.h[l].ln_1.bias

        W_Q = neo.transformer.h[l].attn.attention.q_proj.weight
        W_K = neo.transformer.h[l].attn.attention.k_proj.weight
        W_V = neo.transformer.h[l].attn.attention.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)

        W_O = neo.transformer.h[l].attn.attention.out_proj.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = neo.transformer.h[l].attn.attention.out_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = neo.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = neo.transformer.h[l].ln_2.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = neo.transformer.h[l].mlp.c_fc.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neo.transformer.h[l].mlp.c_fc.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = neo.transformer.h[l].mlp.c_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neo.transformer.h[l].mlp.c_proj.bias
    state_dict["ln_final.w"] = neo.transformer.ln_f.weight
    state_dict["ln_final.b"] = neo.transformer.ln_f.bias

    state_dict["unembed.W_U"] = neo.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    return state_dict
