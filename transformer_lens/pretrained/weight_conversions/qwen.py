import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_qwen_weights(qwen, cfg: HookedTransformerConfig):
    state_dict = {}
    model = qwen.transformer
    state_dict["embed.W_E"] = model.wte.weight

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = model.h[l].ln_1.weight

        W_Q, W_K, W_V = model.h[l].attn.c_attn.weight.split(split_size=cfg.d_model, dim=0)
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        b_Q, b_K, b_V = model.h[l].attn.c_attn.bias.split(split_size=cfg.d_model, dim=0)
        b_Q = einops.rearrange(
            b_Q,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )
        b_K = einops.rearrange(
            b_K,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )
        b_V = einops.rearrange(
            b_V,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V

        W_O = model.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = model.h[l].ln_2.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = model.h[l].mlp.w1.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = model.h[l].mlp.w2.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = model.h[l].mlp.c_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = model.ln_f.weight

    state_dict["unembed.W_U"] = qwen.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
