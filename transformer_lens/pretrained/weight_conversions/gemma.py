import einops
import torch

from transformer_lens.config import TransformerLensConfig


def convert_gemma_weights(gemma, cfg: TransformerLensConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Multimodal detection must survive transformers 5.x, which moved
    # Gemma3ForConditionalGeneration's language_model under .model — the old
    # top-level probe silently reported text-only there and converted the
    # multimodal model down the wrong path.
    language_model = getattr(gemma, "language_model", None)
    if language_model is None:
        language_model = getattr(getattr(gemma, "model", None), "language_model", None)

    if language_model is not None:
        # Vision tower is skipped entirely; only the text transformer converts.
        base_model = getattr(language_model, "model", language_model)
    else:
        # Text-only Gemma3ForCausalLM has .model wrapper
        base_model = gemma.model

    # Gemma Models scale embeddings by multiplying by sqrt(d_model), use hidden state type to match
    # HF implementation
    state_dict["embed.W_E"] = base_model.embed_tokens.weight * torch.tensor(
        cfg.d_model**0.5, dtype=cfg.dtype
    )

    # Gemma has no biases anywhere
    for l in range(cfg.n_layers):
        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        state_dict[f"blocks.{l}.ln1.w"] = base_model.layers[
            l
        ].input_layernorm.weight.float() + torch.ones_like(
            base_model.layers[l].input_layernorm.weight, dtype=torch.float32
        )
        if getattr(cfg, "use_normalization_before_and_after", False):
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln1_post.w"] = base_model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].input_layernorm.weight, dtype=torch.float32
            )

        W_Q = base_model.layers[l].self_attn.q_proj.weight
        W_K = base_model.layers[l].self_attn.k_proj.weight
        W_V = base_model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        # Load q_norm and k_norm if they exist (Gemma 3)
        # Gemma3RMSNorm adds 1 to weights in forward(), so we pre-add it here
        if getattr(cfg, "use_qk_norm", False):
            state_dict[f"blocks.{l}.attn.q_norm.w"] = base_model.layers[
                l
            ].self_attn.q_norm.weight.float() + torch.ones_like(
                base_model.layers[l].self_attn.q_norm.weight, dtype=torch.float32
            )
            state_dict[f"blocks.{l}.attn.k_norm.w"] = base_model.layers[
                l
            ].self_attn.k_norm.weight.float() + torch.ones_like(
                base_model.layers[l].self_attn.k_norm.weight, dtype=torch.float32
            )

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=W_Q.device
        )
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=W_K.device
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=W_V.device
        )

        W_O = base_model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=W_O.device
        )

        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        if not getattr(cfg, "use_normalization_before_and_after", False):
            # Only applies for Gemma 1. Confusingly post_attention_layernorm is applied to mlp_input in Gemma 1 and attn_out in Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = base_model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                base_model.norm.weight, dtype=torch.float32
            )
        else:
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = base_model.layers[
                l
            ].pre_feedforward_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].pre_feedforward_layernorm.weight, dtype=torch.float32
            )
            state_dict[f"blocks.{l}.ln2_post.w"] = base_model.layers[
                l
            ].post_feedforward_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].post_feedforward_layernorm.weight, dtype=torch.float32
            )

        state_dict[f"blocks.{l}.mlp.W_in"] = base_model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = base_model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=base_model.layers[l].mlp.up_proj.weight.device
        )

        state_dict[f"blocks.{l}.mlp.W_out"] = base_model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=base_model.layers[l].mlp.down_proj.weight.device
        )

    # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
    state_dict["ln_final.w"] = base_model.norm.weight.float() + torch.ones_like(
        base_model.norm.weight, dtype=torch.float32
    )

    # For multimodal models, lm_head might not exist or be tied to embeddings
    if hasattr(gemma, "lm_head"):
        state_dict["unembed.W_U"] = gemma.lm_head.weight.T
        unembed_device = gemma.lm_head.weight.device
    else:
        # Multimodal models might use tied embeddings
        state_dict["unembed.W_U"] = base_model.embed_tokens.weight.T
        unembed_device = base_model.embed_tokens.weight.device
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=unembed_device)

    return state_dict
