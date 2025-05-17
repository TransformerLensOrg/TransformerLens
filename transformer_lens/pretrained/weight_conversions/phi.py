import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_phi_weights(phi, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = phi.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = phi.model.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = phi.model.layers[l].input_layernorm.bias

        W_Q = phi.model.layers[l].self_attn.q_proj.weight
        W_K = phi.model.layers[l].self_attn.k_proj.weight
        W_V = phi.model.layers[l].self_attn.v_proj.weight
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
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        b_Q = phi.model.layers[l].self_attn.q_proj.bias
        b_K = phi.model.layers[l].self_attn.k_proj.bias
        b_V = phi.model.layers[l].self_attn.v_proj.bias
        b_Q = einops.rearrange(b_Q, "(n_head d_head) -> n_head d_head", n_head=cfg.n_heads)
        b_K = einops.rearrange(b_K, "(n_head d_head) -> n_head d_head", n_head=cfg.n_heads)
        b_V = einops.rearrange(b_V, "(n_head d_head) -> n_head d_head", n_head=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V

        W_O = phi.model.layers[l].self_attn.dense.weight
        W_O = einops.rearrange(
            W_O, "d_model (n_head d_head) -> n_head d_head d_model", n_head=cfg.n_heads
        )

        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = phi.model.layers[l].self_attn.dense.bias

        # Layer Norm 1 and 2 are tied.
        state_dict[f"blocks.{l}.ln2.w"] = state_dict[f"blocks.{l}.ln1.w"]
        state_dict[f"blocks.{l}.ln2.b"] = state_dict[f"blocks.{l}.ln1.b"]

        state_dict[f"blocks.{l}.mlp.W_in"] = phi.model.layers[l].mlp.fc1.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = phi.model.layers[l].mlp.fc1.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = phi.model.layers[l].mlp.fc2.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = phi.model.layers[l].mlp.fc2.bias

    state_dict["ln_final.w"] = phi.model.final_layernorm.weight
    state_dict["ln_final.b"] = phi.model.final_layernorm.bias

    state_dict["unembed.W_U"] = phi.lm_head.weight.T
    state_dict["unembed.b_U"] = phi.lm_head.bias

    return state_dict
