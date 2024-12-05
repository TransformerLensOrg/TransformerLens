import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_phi3_weights(phi, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = phi.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = phi.model.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

        W = phi.model.layers[l].self_attn.qkv_proj.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=0)
        W_Q = einops.rearrange(
            W_Q, "(n_head d_head) d_model -> n_head d_model d_head", n_head=cfg.n_heads
        )
        W_K = einops.rearrange(
            W_K, "(n_head d_head) d_model  -> n_head d_model d_head", n_head=cfg.n_heads
        )
        W_V = einops.rearrange(
            W_V, "(n_head d_head) d_model  -> n_head d_model d_head", n_head=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.W_V"] = W_V
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)

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
