import einops
import torch
from transformers.models.mixtral import MixtralDecoderLayer, MixtralModel

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def _convert_expert_weights(state_dict: dict) -> dict:
    state_dict[f"blocks.{l}.mlp.experts.{e}.W_in"] = (
        model.layers[l].block_sparse_moe.experts[e].w3.weight.T
    )
    state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate"] = (
        model.layers[l].block_sparse_moe.experts[e].w1.weight.T
    )
    state_dict[f"blocks.{l}.mlp.experts.{e}.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)
    state_dict[f"blocks.{l}.mlp.experts.{e}.W_out"] = (
        model.layers[l].block_sparse_moe.experts[e].w2.weight.T
    )
    state_dict[f"blocks.{l}.mlp.experts.{e}.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)
    return state_dict


def _convert_layer_weights(
    cfg: HookedTransformerConfig, state_dict: dict, l: int, layer: MixtralDecoderLayer
) -> dict:
    state_dict[f"blocks.{l}.ln1.w"] = layer.input_layernorm.weight

    W_Q = layer.self_attn.q_proj.weight
    W_K = layer.self_attn.k_proj.weight
    W_V = layer.self_attn.v_proj.weight
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

    W_O = layer.self_attn.o_proj.weight
    W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
    state_dict[f"blocks.{l}.attn.W_O"] = W_O

    state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict[f"blocks.{l}.ln2.w"] = layer.post_attention_layernorm.weight

    state_dict[f"blocks.{l}.mlp.W_gate"] = layer.block_sparse_moe.gate.weight.T

    # The mapping here from wn to W_{in/out/gate} is a bit confusing:
    # w1 -> W_gate
    # w2 -> W_out
    # w3 -> W_in
    # See https://github.com/mistralai/mistral-inference/blob/8598cf582091a596671be31990448e0620017851/mistral/model.py#L128 for reference
    for e in range(cfg.num_experts):
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_in"] = layer.block_sparse_moe.experts[
            e
        ].w3.weight.T
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate"] = layer.block_sparse_moe.experts[
            e
        ].w1.weight.T
        state_dict[f"blocks.{l}.mlp.experts.{e}.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.mlp.experts.{e}.W_out"] = layer.block_sparse_moe.experts[
            e
        ].w2.weight.T
        state_dict[f"blocks.{l}.mlp.experts.{e}.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)
    return state_dict


def convert_mixtral_weights(hf_model: any, cfg: HookedTransformerConfig):
    # The same as Mistral, but with the MLP replaced with MoE
    # As with Mistral, Mixtral has no biases

    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None
    assert cfg.num_experts is not None

    model: MixtralModel = hf_model.model

    state_dict["embed.W_E"] = model.embed_tokens.weight

    for l in range(cfg.n_layers):
        layer = model.layers[l]
        state_dict = _convert_layer_weights(layer)

    state_dict["ln_final.w"] = model.norm.weight.data

    state_dict["unembed.W_U"] = hf_model.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
