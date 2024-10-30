import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_mixtral_weights(mixtral, cfg: HookedTransformerConfig):
    # The same as Mistral, but with the MLP replaced with MoE
    # As with Mistral, Mixtral has no biases

    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None
    assert cfg.num_experts is not None

    state_dict["embed.W_E"] = mixtral.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = mixtral.model.layers[l].input_layernorm.weight

        W_Q = mixtral.model.layers[l].self_attn.q_proj.weight
        W_K = mixtral.model.layers[l].self_attn.k_proj.weight
        W_V = mixtral.model.layers[l].self_attn.v_proj.weight
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

        W_O = mixtral.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = mixtral.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_gate.weight"] = mixtral.model.layers[
            l
        ].block_sparse_moe.gate.weight

        # The mapping here from wn to W_{in/out/gate} is a bit confusing:
        # w1 -> W_gate
        # w2 -> W_out
        # w3 -> W_in
        # See https://github.com/mistralai/mistral-inference/blob/8598cf582091a596671be31990448e0620017851/mistral/model.py#L128 for reference
        for e in range(cfg.num_experts):
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = (
                mixtral.model.layers[l].block_sparse_moe.experts[e].w3.weight
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = (
                mixtral.model.layers[l].block_sparse_moe.experts[e].w1.weight
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = (
                mixtral.model.layers[l].block_sparse_moe.experts[e].w2.weight
            )

    state_dict["ln_final.w"] = mixtral.model.norm.weight.data

    state_dict["unembed.W_U"] = mixtral.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict

class MixtralWeightConversion:
    
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super({
            "embed.W_E": WeightConversionStep.direct("embed_tokens.weight"),
            "pos_embed.W_pos": WeightConversionStep.direct("wpe.weight"),
            "ln_final.w": WeightConversionStep.direct("ln_f.weight"),
            "unembed.W_U": WeightConversionStep.direct("lm_head.weight.T"),
            "unembed.b_U": WeightConversionStep.zeroes(cfg.d_vocab),
            "blocks": WeigntConversionStep.set("layers", cfg.n_layers, {
                "ln1.w": WeightConversionStep.direct("input_layernorm.weight"),
                "attn.W_Q": WeightConversionStep.rearrange(
                    "self_attn.q_proj.weight",
                    "(n h) m->n m h",
                    n=cfg.n_heads
                ),
                "attn._W_K": WeightConversionStep.rearrange(
                    "self_attn.k_proj.weight",
                    "(n h) m->n m h",
                    n=cfg.n_key_value_heads
                ),
                "attn._W_V": WeightConversionStep.rearrange(
                    "self_attn.v_proj.weight",
                    "(n h) m->n m h",
                    n=cfg.n_key_value_heads
                ),
                "attn.W_O" : WeightConversionStep.rearrange(
                    "self_attn.o_proj.weight",
                    "m (n h)->n h m",
                    n=cfg.n_heads
                ),
                "attn.b_O": WeightConversionStep.zeroes(cfg.d_model),
                "attn.b_Q": WeightConversionStep.zeroes(cfg.n_heads, cfg.d_head),
                "attn._b_K": WeightConversionStep.zeroes(cfg.n_key_value_heads, cfg.d_head),
                "attn._b_V": WeightConversionStep.zeroes(cfg.n_key_value_heads, cfg.d_head),
                "ln2.w": WeightConversionStep.direct("post_attention_layernorm.weight"),
                "mlp.W_gate.weight": WeightConversionStep.direct("block_sparse_moe.gate.weight"),
                "mlp.experts": WeigntConversionStep.set("block_sparse_moe.experts", cfg.num_experts, {
                    "W_in.weight": WeightConversionStep.direct("w3.weight"),
                    "W_gate.weight": WeightConversionStep.direct("w1.weight"),
                    "W_out.weight": WeightConversionStep.direct("w2.weight"),
                }),
            })
        })
