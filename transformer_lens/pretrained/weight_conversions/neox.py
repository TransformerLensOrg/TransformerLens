import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_neox_weights(neox, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = neox.gpt_neox.embed_in.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = neox.gpt_neox.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = neox.gpt_neox.layers[l].input_layernorm.bias

        # For some inexplicable reason, NeoX both uses the concatenated QKV
        # matmul of GPT-2 (afaict this has a neglible performance impact) AND
        # has the flattened axis in the DIFFERENT order of (head_index qkv
        # d_head) - this took me an hour to debug...
        W = neox.gpt_neox.layers[l].attention.query_key_value.weight
        W = einops.rearrange(W, "(i qkv h) m->qkv i m h", i=cfg.n_heads, qkv=3)

        # Fold in layer norm weights
        state_dict[f"blocks.{l}.attn.W_Q"] = W[0]
        state_dict[f"blocks.{l}.attn.W_K"] = W[1]
        state_dict[f"blocks.{l}.attn.W_V"] = W[2]

        qkv_bias = neox.gpt_neox.layers[l].attention.query_key_value.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(index qkv head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        # Fold in layer norm biases
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = neox.gpt_neox.layers[l].attention.dense.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = neox.gpt_neox.layers[l].attention.dense.bias

        state_dict[f"blocks.{l}.ln2.w"] = neox.gpt_neox.layers[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = neox.gpt_neox.layers[l].post_attention_layernorm.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = neox.gpt_neox.layers[l].mlp.dense_h_to_4h.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neox.gpt_neox.layers[l].mlp.dense_h_to_4h.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = neox.gpt_neox.layers[l].mlp.dense_4h_to_h.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neox.gpt_neox.layers[l].mlp.dense_4h_to_h.bias
    state_dict["ln_final.w"] = neox.gpt_neox.final_layer_norm.weight
    state_dict["ln_final.b"] = neox.gpt_neox.final_layer_norm.bias

    state_dict["unembed.W_U"] = neox.embed_out.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)
    return state_dict
