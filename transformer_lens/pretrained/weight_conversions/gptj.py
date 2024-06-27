import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_gptj_weights(gptj, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = gptj.transformer.wte.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gptj.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gptj.transformer.h[l].ln_1.bias

        W_Q = gptj.transformer.h[l].attn.q_proj.weight
        W_K = gptj.transformer.h[l].attn.k_proj.weight
        W_V = gptj.transformer.h[l].attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)

        W_O = gptj.transformer.h[l].attn.out_proj.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # Layer Norm 1 and 2 are tied.
        state_dict[f"blocks.{l}.ln2.w"] = state_dict[f"blocks.{l}.ln1.w"]
        state_dict[f"blocks.{l}.ln2.b"] = state_dict[f"blocks.{l}.ln1.b"]

        state_dict[f"blocks.{l}.mlp.W_in"] = gptj.transformer.h[l].mlp.fc_in.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = gptj.transformer.h[l].mlp.fc_in.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = gptj.transformer.h[l].mlp.fc_out.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = gptj.transformer.h[l].mlp.fc_out.bias
    state_dict["ln_final.w"] = gptj.transformer.ln_f.weight
    state_dict["ln_final.b"] = gptj.transformer.ln_f.bias

    state_dict["unembed.W_U"] = gptj.lm_head.weight.T
    # Contains a bias, for some reason?
    state_dict["unembed.b_U"] = gptj.lm_head.bias
    return state_dict
