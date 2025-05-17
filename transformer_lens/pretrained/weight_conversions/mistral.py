import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_mistral_weights(mistral, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = mistral.model.embed_tokens.weight

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Mistral has no biases anywhere
    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = mistral.model.layers[l].input_layernorm.weight

        W_Q = mistral.model.layers[l].self_attn.q_proj.weight
        W_K = mistral.model.layers[l].self_attn.k_proj.weight
        W_V = mistral.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = mistral.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = mistral.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = mistral.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = mistral.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = mistral.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = mistral.model.norm.weight

    state_dict["unembed.W_U"] = mistral.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
