import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_baichuan_weights(baichuan, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = baichuan.model.embed_tokens.weight

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = baichuan.model.layers[l].input_layernorm.weight

        W = baichuan.model.layers[l].self_attn.W_pack.weight

        W_split = W.T.reshape(cfg.d_model, cfg.n_heads, 3, cfg.d_head)

        W_Q, W_K, W_V = W_split[..., 0, :], W_split[..., 1, :], W_split[..., 2, :]
        W_Q = einops.rearrange(W_Q, "m n h ->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m n h ->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m n h ->n m h", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=W_Q.device
        )
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(
            cfg.n_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=W_Q.device,
        )
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(
            cfg.n_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=W_Q.device,
        )

        W_O = baichuan.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=W_O.device
        )

        state_dict[f"blocks.{l}.ln2.w"] = baichuan.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = baichuan.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = baichuan.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=W_O.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = baichuan.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=W_O.dtype)

    state_dict["ln_final.w"] = baichuan.model.norm.weight
    state_dict["pos_embed.W_pos"] = baichuan.model.transformer.wpe.weight
    state_dict["unembed.W_U"] = baichuan.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=W_O.dtype)

    return state_dict
