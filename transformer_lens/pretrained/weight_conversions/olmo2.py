import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM, Olmo2DecoderLayer

def convert_olmo2_weights(olmo2:Olmo2ForCausalLM, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.d_mlp is not None

    state_dict["embed.W_E"] = olmo2.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        olmo2_layer:Olmo2DecoderLayer = olmo2.model.layers[l]

        W_Q = olmo2_layer.self_attn.q_proj.weight
        W_K = olmo2_layer.self_attn.k_proj.weight
        W_V = olmo2_layer.self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V
        state_dict[f"blocks.{l}.attn.q_norm.w"] = olmo2_layer.self_attn.q_norm.weight
        state_dict[f"blocks.{l}.attn.k_norm.w"] = olmo2_layer.self_attn.k_norm.weight

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = olmo2_layer.self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln1.w"] = olmo2_layer.post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = olmo2_layer.mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = olmo2_layer.mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = olmo2_layer.mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = olmo2_layer.post_feedforward_layernorm.weight


    state_dict["ln_final.w"] = olmo2.model.norm.weight

    state_dict["unembed.W_U"] = olmo2.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
