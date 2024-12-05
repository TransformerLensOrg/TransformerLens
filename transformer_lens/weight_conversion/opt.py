import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_opt_weights(opt, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = opt.model.decoder.embed_tokens.weight
    state_dict["pos_embed.W_pos"] = opt.model.decoder.embed_positions.weight[2:, :]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = opt.model.decoder.layers[l].self_attn_layer_norm.weight
        state_dict[f"blocks.{l}.ln1.b"] = opt.model.decoder.layers[l].self_attn_layer_norm.bias

        W_Q = opt.model.decoder.layers[l].self_attn.q_proj.weight
        W_K = opt.model.decoder.layers[l].self_attn.k_proj.weight
        W_V = opt.model.decoder.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(
            W_Q,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )
        W_K = einops.rearrange(
            W_K,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )
        W_V = einops.rearrange(
            W_V,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        q_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.q_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )
        k_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.k_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )
        v_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.v_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = q_bias
        state_dict[f"blocks.{l}.attn.b_K"] = k_bias
        state_dict[f"blocks.{l}.attn.b_V"] = v_bias

        W_O = opt.model.decoder.layers[l].self_attn.out_proj.weight
        W_O = einops.rearrange(
            W_O,
            "d_model (index d_head)->index d_head d_model",
            index=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = opt.model.decoder.layers[l].self_attn.out_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = opt.model.decoder.layers[l].final_layer_norm.weight
        state_dict[f"blocks.{l}.ln2.b"] = opt.model.decoder.layers[l].final_layer_norm.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = opt.model.decoder.layers[l].fc1.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = opt.model.decoder.layers[l].fc2.weight.T

        state_dict[f"blocks.{l}.mlp.b_in"] = opt.model.decoder.layers[l].fc1.bias
        state_dict[f"blocks.{l}.mlp.b_out"] = opt.model.decoder.layers[l].fc2.bias
    state_dict["ln_final.w"] = opt.model.decoder.final_layer_norm.weight
    state_dict["ln_final.b"] = opt.model.decoder.final_layer_norm.bias
    state_dict["unembed.W_U"] = opt.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    return state_dict
