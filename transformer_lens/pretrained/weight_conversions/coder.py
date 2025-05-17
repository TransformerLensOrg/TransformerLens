import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_coder_weights(model, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = model.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = model.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = model.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = model.transformer.h[l].ln_1.bias

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W_KV = model.transformer.h[l].attn.kv_attn.weight  # [d_model, 2 * d_head]
        W_K, W_V = torch.tensor_split(W_KV, 2, dim=1)
        W_Q = model.transformer.h[l].attn.q_attn.weight  # [d_model, d_model]
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.repeat(W_K, "m h -> i m h", i=cfg.n_heads)
        W_V = einops.repeat(W_V, "m h -> i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        b_Q = einops.rearrange(
            model.transformer.h[l].attn.q_attn.bias,
            "(index head)-> index head",
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        b_KV = model.transformer.h[l].attn.kv_attn.bias  # [2 * d_head]
        b_K, b_V = torch.tensor_split(b_KV, 2, dim=0)
        b_K = einops.repeat(b_K, "head -> index head", index=cfg.n_heads)
        b_V = einops.repeat(b_V, "head -> index head", index=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V

        W_O = model.transformer.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = model.transformer.h[l].attn.c_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = model.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = model.transformer.h[l].ln_2.bias

        W_in = model.transformer.h[l].mlp.c_fc.weight
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = model.transformer.h[l].mlp.c_fc.bias

        W_out = model.transformer.h[l].mlp.c_proj.weight
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = model.transformer.h[l].mlp.c_proj.bias
    state_dict["unembed.W_U"] = model.lm_head.weight.T

    state_dict["ln_final.w"] = model.transformer.ln_f.weight
    state_dict["ln_final.b"] = model.transformer.ln_f.bias
    return state_dict
