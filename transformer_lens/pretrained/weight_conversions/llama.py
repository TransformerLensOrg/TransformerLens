from typing import cast

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_llama_weights(llama, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = llama.model.embed_tokens.weight

    # Some models with the Llama architecture use Grouped Query Attention, and so for these we need to modify
    # the state dict keys for the K/V attention weight/biases, prepending "_" to the key names.
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""
    # need a cast since MyPy isn't smart enough to realize that using_gqa implies n_key_value_heads is not None
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)

    # llama has no biases anywhere and deals with everything else roughly like
    # GPTNeoX with different names

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = llama.model.layers[l].input_layernorm.weight

        W_Q = llama.model.layers[l].self_attn.q_proj.weight
        W_K = llama.model.layers[l].self_attn.k_proj.weight
        W_V = llama.model.layers[l].self_attn.v_proj.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )

        W_O = llama.model.layers[l].self_attn.o_proj.weight

        if not cfg.load_in_4bit:
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"blocks.{l}.ln2.w"] = llama.model.layers[l].post_attention_layernorm.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            state_dict[f"blocks.{l}.mlp.W_in"] = llama.model.layers[l].mlp.up_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_gate"] = llama.model.layers[l].mlp.gate_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_out"] = llama.model.layers[l].mlp.down_proj.weight.T
        else:
            state_dict[f"blocks.{l}.mlp.W_in"] = llama.model.layers[l].mlp.up_proj.weight
            state_dict[f"blocks.{l}.mlp.W_gate"] = llama.model.layers[l].mlp.gate_proj.weight
            state_dict[f"blocks.{l}.mlp.W_out"] = llama.model.layers[l].mlp.down_proj.weight

        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

    state_dict["ln_final.w"] = llama.model.norm.weight

    state_dict["unembed.W_U"] = llama.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device)

    return state_dict
