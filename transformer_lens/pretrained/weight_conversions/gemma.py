import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_gemma3_weights(gemma, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Gemma 3 has a different model structure
    if hasattr(gemma, 'language_model'):
        model = gemma.language_model
    else:
        model = gemma
        
    embed_weight = model.model.embed_tokens.weight.float()
    state_dict["embed.W_E"] = embed_weight
    state_dict["unembed.W_U"] = embed_weight.T.clone()
    
    # Gemma 3 has no biases anywhere
    for l in range(cfg.n_layers):
        # Gemma 3 RMSNorm adds 1 to weights before multiplying by input
        state_dict[f"blocks.{l}.ln1.w"] = model.model.layers[l].input_layernorm.weight.float()
        state_dict[f"blocks.{l}.ln1_post.w"] = model.model.layers[l].post_attention_layernorm.weight.float()


        # Gemma 3 has different attention patterns with query/key normalization
        W_Q = model.model.layers[l].self_attn.q_proj.weight
        W_K = model.model.layers[l].self_attn.k_proj.weight
        W_V = model.model.layers[l].self_attn.v_proj.weight
        
        # First rearrange the weights
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        
        # Apply normalization to query and key states
        q_norm = model.model.layers[l].self_attn.q_norm.weight.float()
        k_norm = model.model.layers[l].self_attn.k_norm.weight.float()
        
        # Reshape normalization weights to match the rearranged dimensions
        # W_Q shape is [n_heads, d_model, d_head]
        state_dict[f"blocks.{l}.attn.q_norm.w"]  = q_norm
        state_dict[f"blocks.{l}.attn.k_norm.w"]  = k_norm #.view(cfg.n_key_value_heads, -1).mean(dim=1)  # [4]

           
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        # Zero biases for attention
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = model.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # Gemma 3 has pre and post feedforward norms
        state_dict[f"blocks.{l}.ln2.w"] = model.model.layers[l].pre_feedforward_layernorm.weight.float()
        state_dict[f"blocks.{l}.ln2_post.w"] = model.model.layers[l].post_feedforward_layernorm.weight.float()


        # Gemma 3 MLP structure with up_proj, gate_proj, and down_proj
        state_dict[f"blocks.{l}.mlp.W_in"] = model.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = model.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.mlp.W_out"] = model.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)


    # Final norm
    state_dict["ln_final.w"] = model.model.norm.weight.float()

    # Output embedding with logit softcapping
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict


def convert_gemma_weights(gemma, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # gemma = gemma.language_model
    # Gemma Models scale embeddings by multiplying by sqrt(d_model), use hidden state type to match
    # HF implementation
    state_dict["embed.W_E"] = gemma.model.embed_tokens.weight * torch.tensor(
        cfg.d_model**0.5, dtype=cfg.dtype
    )

    # Gemma has no biases anywhere
    for l in range(cfg.n_layers):
        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        state_dict[f"blocks.{l}.ln1.w"] = gemma.model.layers[
            l
        ].input_layernorm.weight.float() + torch.ones_like(
            gemma.model.layers[l].input_layernorm.weight, dtype=torch.float32
        )
        if cfg.use_normalization_before_and_after:
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln1_post.w"] = gemma.model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                gemma.model.layers[l].input_layernorm.weight, dtype=torch.float32
            )

        W_Q = gemma.model.layers[l].self_attn.q_proj.weight
        W_K = gemma.model.layers[l].self_attn.k_proj.weight
        W_V = gemma.model.layers[l].self_attn.v_proj.weight
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

        W_O = gemma.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        if not cfg.use_normalization_before_and_after:
            # Only applies for Gemma 1. Confusingly post_attention_layernorm is applied to mlp_input in Gemma 1 and attn_out in Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = gemma.model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                gemma.model.norm.weight, dtype=torch.float32
            )
        else:
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = gemma.model.layers[
                l
            ].pre_feedforward_layernorm.weight.float() + torch.ones_like(
                gemma.model.layers[l].pre_feedforward_layernorm.weight, dtype=torch.float32
            )
            state_dict[f"blocks.{l}.ln2_post.w"] = gemma.model.layers[
                l
            ].post_feedforward_layernorm.weight.float() + torch.ones_like(
                gemma.model.layers[l].post_feedforward_layernorm.weight, dtype=torch.float32
            )

        state_dict[f"blocks.{l}.mlp.W_in"] = gemma.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = gemma.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = gemma.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
    state_dict["ln_final.w"] = gemma.model.norm.weight.float() + torch.ones_like(
        gemma.model.norm.weight, dtype=torch.float32
    )

    state_dict["unembed.W_U"] = gemma.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
