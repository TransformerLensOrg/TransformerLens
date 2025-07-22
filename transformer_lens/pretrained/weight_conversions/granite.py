from typing import cast

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_granite_weights(hf_model, cfg: HookedTransformerConfig):
    """
    Converts the weights of a Hugging Face GraniteForCausalLM model to the format
    used by HookedTransformer
    """
    state_dict = {}

    # Token Embeddings - move to the correct device
    state_dict["embed.W_E"] = hf_model.model.embed_tokens.weight.to(device=cfg.device)

    # Granite architecture use Grouped Query Attention
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)

    for l in range(cfg.n_layers):
        # LayerNorm 1 - move to the correct device
        state_dict[f"blocks.{l}.ln1.w"] = hf_model.model.layers[l].input_layernorm.weight.to(
            device=cfg.device
        )

        # Attention weights
        # Transpose the weights first, then rearrange
        W_Q = hf_model.model.layers[l].self_attn.q_proj.weight.T
        W_K = hf_model.model.layers[l].self_attn.k_proj.weight.T
        W_V = hf_model.model.layers[l].self_attn.v_proj.weight.T
        W_O = hf_model.model.layers[l].self_attn.o_proj.weight.T

        # Reshape weights for TransformerLens internal format
        W_Q = einops.rearrange(W_Q, "m (n h) -> n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (n h) -> n m h", n=n_kv_heads)
        W_V = einops.rearrange(W_V, "m (n h) -> n m h", n=n_kv_heads)
        W_O = einops.rearrange(W_O, "(n h) m -> n h m", n=cfg.n_heads)

        # Move weights to the correct device
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q.to(device=cfg.device)
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K.to(device=cfg.device)
        state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V.to(device=cfg.device)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        # Attention biases (Granite models don't use biases, so we set them to zero)
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            n_kv_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        # LayerNorm 2
        state_dict[f"blocks.{l}.ln2.w"] = hf_model.model.layers[
            l
        ].post_attention_layernorm.weight.to(device=cfg.device)

        # MLP weights for GatedMLP - move to the correct device
        state_dict[f"blocks.{l}.mlp.W_in"] = hf_model.model.layers[l].mlp.up_proj.weight.T.to(
            device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.W_gate"] = hf_model.model.layers[l].mlp.gate_proj.weight.T.to(
            device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.W_out"] = hf_model.model.layers[l].mlp.down_proj.weight.T.to(
            device=cfg.device
        )

        # MLP biases (Granite models don't use biases, so we set them to zero)
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

    # Final LayerNorm
    state_dict["ln_final.w"] = hf_model.model.norm.weight.to(device=cfg.device)

    # Unembedding weights
    state_dict["unembed.W_U"] = hf_model.lm_head.weight.T.to(device=cfg.device)
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device)

    return state_dict
