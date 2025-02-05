import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_t5_weights(t5, cfg: HookedTransformerConfig):
    state_dict = {
        "embed.W_E": t5.encoder.embed_tokens.weight,
        "unembed.W_U": t5.encoder.embed_tokens.weight.T,
        "encoder.0.attn.rel_pos_bias.weight": t5.encoder.block[0]
        .layer[0]
        .SelfAttention.relative_attention_bias.weight,
    }

    for l in range(cfg.n_layers):
        block = t5.encoder.block[l]
        state_dict[f"encoder.{l}.attn.W_Q"] = einops.rearrange(
            block.layer[0].SelfAttention.q.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"encoder.{l}.attn.W_K"] = einops.rearrange(
            block.layer[0].SelfAttention.k.weight, "(i h) m -> i m h", i=cfg.n_heads
        )

        state_dict[f"encoder.{l}.attn.W_V"] = einops.rearrange(
            block.layer[0].SelfAttention.v.weight, "(i h) m -> i m h", i=cfg.n_heads
        )

        state_dict[f"encoder.{l}.attn.W_O"] = einops.rearrange(
            block.layer[0].SelfAttention.o.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )
        state_dict[f"encoder.{l}.ln1.w"] = block.layer[0].layer_norm.weight

        # fixme DenseReluDense may be T5DenseGatedActDense instead
        state_dict[f"encoder.{l}.mlp.W_in"] = einops.rearrange(
            block.layer[1].DenseReluDense.wi.weight, "mlp model -> model mlp"
        )

        state_dict[f"encoder.{l}.mlp.W_out"] = einops.rearrange(
            block.layer[1].DenseReluDense.wo.weight, "model mlp -> mlp model"
        )
        state_dict[f"encoder.{l}.ln2.w"] = block.layer[1].layer_norm.weight

    state_dict["encoder_final_ln.w"] = t5.encoder.final_layer_norm.weight

    state_dict["decoder.0.attn.rel_pos_bias.weight"] = (
        t5.decoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
    )

    for l in range(cfg.n_layers):
        block = t5.decoder.block[l]
        state_dict[f"decoder.{l}.attn.W_Q"] = einops.rearrange(
            block.layer[0].SelfAttention.q.weight, "(i h) m -> i m h", i=cfg.n_heads
        )

        state_dict[f"decoder.{l}.attn.W_K"] = einops.rearrange(
            block.layer[0].SelfAttention.k.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"decoder.{l}.attn.W_V"] = einops.rearrange(
            block.layer[0].SelfAttention.v.weight, "(i h) m -> i m h", i=cfg.n_heads
        )

        state_dict[f"decoder.{l}.attn.W_O"] = einops.rearrange(
            block.layer[0].SelfAttention.o.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )

        state_dict[f"decoder.{l}.ln1.w"] = block.layer[0].layer_norm.weight

        state_dict[f"decoder.{l}.cross_attn.W_Q"] = einops.rearrange(
            block.layer[1].EncDecAttention.q.weight, "(i h) m -> i m h", i=cfg.n_heads
        )

        state_dict[f"decoder.{l}.cross_attn.W_K"] = einops.rearrange(
            block.layer[1].EncDecAttention.k.weight, "(i h) m -> i m h", i=cfg.n_heads
        )

        state_dict[f"decoder.{l}.cross_attn.W_V"] = einops.rearrange(
            block.layer[1].EncDecAttention.v.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"decoder.{l}.cross_attn.W_O"] = einops.rearrange(
            block.layer[1].EncDecAttention.o.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )
        state_dict[f"decoder.{l}.ln2.w"] = block.layer[1].layer_norm.weight

        # fixme DenseReluDense may be T5DenseGatedActDense instead
        state_dict[f"decoder.{l}.mlp.W_in"] = einops.rearrange(
            block.layer[2].DenseReluDense.wi.weight, "mlp model -> model mlp"
        )
        state_dict[f"decoder.{l}.mlp.W_out"] = einops.rearrange(
            block.layer[2].DenseReluDense.wo.weight, "model mlp -> mlp model"
        )
        state_dict[f"decoder.{l}.ln3.w"] = block.layer[2].layer_norm.weight

    state_dict["decoder_final_ln.w"] = t5.decoder.final_layer_norm.weight

    return state_dict
