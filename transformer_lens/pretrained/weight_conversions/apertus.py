"""
Apertus is Llama like model architecture from Swiss AI.
convert weights to standardized format for HookedTransformer
"""

from typing import cast

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_apertus_weights(apertus, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = apertus.model.embed_tokens.weight

    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""

    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)


    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = apertus.model.layers[l].attention_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype, device=cfg.device)

        W_Q = apertus.model.layers[l].self_attn.q_proj.weight
        W_K = apertus.model.layers[l].self_attn.k_proj.weight
        W_V = apertus.model.layers[l].self_attn.v_proj.weight

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

        W_O = apertus.model.layers[l].self_attn.o_proj.weight

        if not cfg.load_in_4bit:
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"blocks.{l}.ln2.w"] = apertus.model.layers[l].feedforward_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype, device=cfg.device)

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            state_dict[f"blocks.{l}.mlp.W_in"] = apertus.model.layers[l].mlp.up_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_out"] = apertus.model.layers[l].mlp.down_proj.weight.T
        else:
            state_dict[f"blocks.{l}.mlp.W_in"] = apertus.model.layers[l].mlp.up_proj.weight
            state_dict[f"blocks.{l}.mlp.W_out"] = apertus.model.layers[l].mlp.down_proj.weight

        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        # Extract trainable activation parameters
        mlp = apertus.model.layers[l].mlp
        try:
            if hasattr(mlp, 'act_fn'):
                alpha_p = mlp.act_fn.alpha_p
                alpha_n = mlp.act_fn.alpha_n
                beta = mlp.act_fn.beta
            elif hasattr(mlp, 'act'):
                alpha_p = mlp.act.alpha_p
                alpha_n = mlp.act.alpha_n
                beta = mlp.act.beta
            else:
                alpha_p = mlp.alpha_p
                alpha_n = mlp.alpha_n
                beta = mlp.beta
            state_dict[f"blocks.{l}.mlp.act_fn.alpha_p"] = alpha_p
            state_dict[f"blocks.{l}.mlp.act_fn.alpha_n"] = alpha_n
            state_dict[f"blocks.{l}.mlp.act_fn.beta"] = beta
        except AttributeError:
            # If parameters not found, use defaults
            print(f"Activation parameters not found in layer {l}, using defaults")
            state_dict[f"blocks.{l}.mlp.act_fn.alpha_p"] = torch.tensor(0.8, dtype=cfg.dtype, device=cfg.device)
            state_dict[f"blocks.{l}.mlp.act_fn.alpha_n"] = torch.tensor(0.8, dtype=cfg.dtype, device=cfg.device)
            state_dict[f"blocks.{l}.mlp.act_fn.beta"] = torch.tensor(0.5, dtype=cfg.dtype, device=cfg.device)

    state_dict["ln_final.w"] = apertus.model.norm.weight
    state_dict["ln_final.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype, device=cfg.device)

    state_dict["unembed.W_U"] = apertus.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device)

    return state_dict
