import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_mingpt_weights(old_state_dict, cfg: HookedTransformerConfig):
    # mingpt (https://github.com/karpathy/minGPT) is mostly similar to GPT-2,
    # but doesn't concat the QKV matrices.
    state_dict = {}

    state_dict["embed.W_E"] = old_state_dict["tok_emb.weight"]
    state_dict["pos_embed.W_pos"] = old_state_dict["pos_emb"].squeeze()

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = old_state_dict[f"blocks.{l}.ln1.weight"]
        state_dict[f"blocks.{l}.ln1.b"] = old_state_dict[f"blocks.{l}.ln1.bias"]

        W_Q = old_state_dict[f"blocks.{l}.attn.query.weight"]
        W_K = old_state_dict[f"blocks.{l}.attn.key.weight"]
        W_V = old_state_dict[f"blocks.{l}.attn.value.weight"]
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        q_bias = einops.rearrange(
            old_state_dict[f"blocks.{l}.attn.query.bias"], "(i h)->i h", i=cfg.n_heads
        )
        k_bias = einops.rearrange(
            old_state_dict[f"blocks.{l}.attn.key.bias"], "(i h)->i h", i=cfg.n_heads
        )
        v_bias = einops.rearrange(
            old_state_dict[f"blocks.{l}.attn.value.bias"], "(i h)->i h", i=cfg.n_heads
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = q_bias
        state_dict[f"blocks.{l}.attn.b_K"] = k_bias
        state_dict[f"blocks.{l}.attn.b_V"] = v_bias

        W_O = old_state_dict[f"blocks.{l}.attn.proj.weight"]
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = old_state_dict[f"blocks.{l}.attn.proj.bias"]

        state_dict[f"blocks.{l}.ln2.w"] = old_state_dict[f"blocks.{l}.ln2.weight"]
        state_dict[f"blocks.{l}.ln2.b"] = old_state_dict[f"blocks.{l}.ln2.bias"]

        W_in = old_state_dict[f"blocks.{l}.mlp.0.weight"]
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in.T
        state_dict[f"blocks.{l}.mlp.b_in"] = old_state_dict[f"blocks.{l}.mlp.0.bias"]

        W_out = old_state_dict[f"blocks.{l}.mlp.2.weight"]
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out.T
        state_dict[f"blocks.{l}.mlp.b_out"] = old_state_dict[f"blocks.{l}.mlp.2.bias"]

    state_dict["unembed.W_U"] = old_state_dict["head.weight"].T

    state_dict["ln_final.w"] = old_state_dict["ln_f.weight"]
    state_dict["ln_final.b"] = old_state_dict["ln_f.bias"]

    return state_dict
