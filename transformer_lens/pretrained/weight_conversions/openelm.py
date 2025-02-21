import torch
import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def convert_openelm_weights(openelm, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.d_mlp is not None
    assert cfg.n_key_value_heads is not None

    state_dict["embed.W_E"] = openelm.transformer.token_embeddings.weight

    for l in range(cfg.n_layers):
        WQ = openelm.transformer.layers[l].attn.qkv_proj.weight[:(cfg.n_query_heads[l] * cfg.d_head)]
        WK = openelm.transformer.layers[l].attn.qkv_proj.weight[(cfg.n_query_heads[l] * cfg.d_head) : ((cfg.n_query_heads[l] + cfg.n_key_value_heads[l]) * cfg.d_head)]
        WV = openelm.transformer.layers[l].attn.qkv_proj.weight[-cfg.n_key_value_heads[l] * cfg.d_head:]

        WQ = einops.rearrange(WQ, "(n h) m->n m h", n=cfg.n_query_heads[l])
        WK = einops.rearrange(WK, "(n h) m->n m h", n=cfg.n_key_value_heads[l])
        WV = einops.rearrange(WV, "(n h) m->n m h", n=cfg.n_key_value_heads[l])

        state_dict[f"blocks.{l}.attn.W_Q"] = WQ
        state_dict[f"blocks.{l}.attn._W_K"] = WK
        state_dict[f"blocks.{l}.attn._W_V"] = WV

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads[l], cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads[l], cfg.d_head, dtype=cfg.dtype
        )

        WO = openelm.transformer.layers[l].attn.out_proj.weight
        WO = einops.rearrange(WO, "m (n h)->n h m", n=cfg.n_query_heads[l])
        state_dict[f"blocks.{l}.attn.W_O"] = WO

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.ln2.w"] = openelm.transformer.layers[l].attn_norm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = openelm.transformer.layers[l].ffn.proj_1.weight[:cfg.d_mlps[l], :].T 
        state_dict[f"blocks.{l}.mlp.W_gate"] = openelm.transformer.layers[l].ffn.proj_1.weight[cfg.d_mlps[l]:, :].T 
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlps[l], dtype=cfg.dtype)
        state_dict[f"blocks.{l}.mlp.W_out"] = openelm.transformer.layers[l].ffn.proj_2.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(openelm.transformer.layers[l].ffn.proj_2.weight.shape[0], dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.ln3.w"] = openelm.transformer.layers[l].ffn_norm.weight

    state_dict["ln_final.w"] = openelm.transformer.norm.weight

    state_dict["unembed.W_U"] = openelm.transformer.token_embeddings.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)