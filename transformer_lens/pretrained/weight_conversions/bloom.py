import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_bloom_weights(bloom, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = bloom.transformer.word_embeddings.weight

    # Bloom uses post embedding layer norm
    state_dict["embed.ln.w"] = bloom.transformer.word_embeddings_layernorm.weight
    state_dict["embed.ln.b"] = bloom.transformer.word_embeddings_layernorm.bias

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = bloom.transformer.h[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = bloom.transformer.h[l].input_layernorm.bias

        W = bloom.transformer.h[l].self_attention.query_key_value.weight

        W_split = W.T.reshape(cfg.d_model, cfg.n_heads, 3, cfg.d_head)

        W_Q, W_K, W_V = W_split[..., 0, :], W_split[..., 1, :], W_split[..., 2, :]
        W_Q = einops.rearrange(W_Q, "m n h ->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m n h ->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m n h ->n m h", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = bloom.transformer.h[l].self_attention.query_key_value.bias
        qkv_bias = qkv_bias.reshape(cfg.n_heads, 3, cfg.d_head)

        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[:, 0, :]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[:, 1, :]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[:, 2, :]

        W_O = bloom.transformer.h[l].self_attention.dense.weight.T  # [1024, 1024]
        W_O = einops.rearrange(W_O, "(n h) m->n h m", n=cfg.n_heads)  # [n_heads, d_head, d_model]
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = bloom.transformer.h[l].self_attention.dense.bias

        state_dict[f"blocks.{l}.ln2.w"] = bloom.transformer.h[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = bloom.transformer.h[l].post_attention_layernorm.bias

        W_in = bloom.transformer.h[l].mlp.dense_h_to_4h.weight.T
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = bloom.transformer.h[l].mlp.dense_h_to_4h.bias

        W_out = bloom.transformer.h[l].mlp.dense_4h_to_h.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = bloom.transformer.h[l].mlp.dense_4h_to_h.bias
    state_dict["unembed.W_U"] = bloom.lm_head.weight.T

    state_dict["ln_final.w"] = bloom.transformer.ln_f.weight
    state_dict["ln_final.b"] = bloom.transformer.ln_f.bias
    return state_dict
