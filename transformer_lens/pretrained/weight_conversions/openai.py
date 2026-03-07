"""Weight conversion for OpenAI GPT-OSS models.

GPT-OSS has a unique MoE architecture:
- GptOssExperts stores all expert weights in merged tensors (not individual modules)
- gate_up_proj: (num_experts, hidden_size, 2*expert_dim) with interleaved gate/up columns
- down_proj: (num_experts, expert_dim, hidden_size)
- Router (GptOssTopKRouter) uses weight + bias
"""

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_gpt_oss_weights(gpt_oss, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None
    assert cfg.d_mlp is not None
    assert cfg.num_experts is not None

    state_dict["embed.W_E"] = gpt_oss.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        layer = gpt_oss.model.layers[l]

        # LayerNorms
        state_dict[f"blocks.{l}.ln1.w"] = layer.input_layernorm.weight
        state_dict[f"blocks.{l}.ln2.w"] = layer.post_attention_layernorm.weight

        # Attention
        W_Q = einops.rearrange(layer.self_attn.q_proj.weight, "(n h) m -> n m h", n=cfg.n_heads)
        W_K = einops.rearrange(layer.self_attn.k_proj.weight, "(n h) m -> n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(layer.self_attn.v_proj.weight, "(n h) m -> n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        if layer.self_attn.q_proj.bias is not None:
            state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
                layer.self_attn.q_proj.bias, "(n h) -> n h", n=cfg.n_heads
            )
            state_dict[f"blocks.{l}.attn._b_K"] = einops.rearrange(
                layer.self_attn.k_proj.bias, "(n h) -> n h", n=cfg.n_key_value_heads
            )
            state_dict[f"blocks.{l}.attn._b_V"] = einops.rearrange(
                layer.self_attn.v_proj.bias, "(n h) -> n h", n=cfg.n_key_value_heads
            )
        else:
            state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
            state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype)
            state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype)

        W_O = einops.rearrange(layer.self_attn.o_proj.weight, "m (n h) -> n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        if hasattr(layer.self_attn.o_proj, "bias") and layer.self_attn.o_proj.bias is not None:
            state_dict[f"blocks.{l}.attn.b_O"] = layer.self_attn.o_proj.bias
        else:
            state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # MoE - Router (GPT-OSS uses 'router' with bias)
        state_dict[f"blocks.{l}.mlp.W_gate.weight"] = layer.mlp.router.weight
        state_dict[f"blocks.{l}.mlp.W_gate.bias"] = layer.mlp.router.bias

        # MoE - Experts
        # GPT-OSS stores all experts in merged tensors:
        #   gate_up_proj: (num_experts, hidden_size, 2*expert_dim) - interleaved gate/up
        #   down_proj: (num_experts, expert_dim, hidden_size)
        experts = layer.mlp.experts
        gate_up_proj = experts.gate_up_proj        # (num_experts, hidden_size, 2*expert_dim)
        gate_up_bias = experts.gate_up_proj_bias   # (num_experts, 2*expert_dim)
        down_proj = experts.down_proj              # (num_experts, expert_dim, hidden_size)
        down_bias = experts.down_proj_bias         # (num_experts, hidden_size)

        for e in range(cfg.num_experts):
            # Split interleaved gate_up_proj into separate gate and up (in) projections
            # Even columns → gate path, Odd columns → up/in path
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = gate_up_proj[e, :, ::2].T.contiguous()
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.bias"] = gate_up_bias[e, ::2].contiguous()

            state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = gate_up_proj[e, :, 1::2].T.contiguous()
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.bias"] = gate_up_bias[e, 1::2].contiguous()

            state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = down_proj[e].T.contiguous()
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.bias"] = down_bias[e].contiguous()

    state_dict["ln_final.w"] = gpt_oss.model.norm.weight
    state_dict["unembed.W_U"] = gpt_oss.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
