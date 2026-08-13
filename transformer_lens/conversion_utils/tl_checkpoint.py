"""Convert TransformerLens (HookedTransformer) checkpoints to native bridge format.

Use this to load legacy TL-format checkpoints into a TransformerBridge.boot_native() model.
"""

from __future__ import annotations

import einops
import torch

from transformer_lens.config import TransformerBridgeConfig


def convert_tl_checkpoint(
    tl_state_dict: dict[str, torch.Tensor],
    cfg: TransformerBridgeConfig,
) -> dict[str, torch.Tensor]:
    """Convert a HookedTransformer state_dict to native bridge format.

    Args:
        tl_state_dict: State dict in TL format (keys like embed.W_E, blocks.0.attn.W_Q)
        cfg: The TransformerBridgeConfig used for boot_native

    Returns:
        State dict in native bridge format (keys like tok_embed.weight, layers.0.attn.q.weight)

    Example::

        cfg = TransformerBridgeConfig(n_layers=8, d_model=512, ...)
        bridge = TransformerBridge.boot_native(cfg)
        tl_sd = torch.load("model.pth")
        native_sd = convert_tl_checkpoint(tl_sd, cfg)
        bridge.load_state_dict(native_sd)
    """
    out = {}
    n_heads = cfg.n_heads
    n_layers = cfg.n_layers

    # Embeddings
    if "embed.W_E" in tl_state_dict:
        out["tok_embed.weight"] = tl_state_dict["embed.W_E"]
    if "pos_embed.W_pos" in tl_state_dict:
        out["pos.weight"] = tl_state_dict["pos_embed.W_pos"]

    # Blocks/layers
    for layer in range(n_layers):
        tl_prefix = f"blocks.{layer}"
        native_prefix = f"layers.{layer}"

        # LayerNorm 1
        if f"{tl_prefix}.ln1.w" in tl_state_dict:
            out[f"{native_prefix}.ln1.weight"] = tl_state_dict[f"{tl_prefix}.ln1.w"]
        if f"{tl_prefix}.ln1.b" in tl_state_dict:
            out[f"{native_prefix}.ln1.bias"] = tl_state_dict[f"{tl_prefix}.ln1.b"]

        # LayerNorm 2
        if f"{tl_prefix}.ln2.w" in tl_state_dict:
            out[f"{native_prefix}.ln2.weight"] = tl_state_dict[f"{tl_prefix}.ln2.w"]
        if f"{tl_prefix}.ln2.b" in tl_state_dict:
            out[f"{native_prefix}.ln2.bias"] = tl_state_dict[f"{tl_prefix}.ln2.b"]

        # Attention weights: TL [n_heads, d_model, d_head] -> Native [n_heads*d_head, d_model]
        for qkv in ["Q", "K", "V"]:
            tl_key = f"{tl_prefix}.attn.W_{qkv}"
            native_key = f"{native_prefix}.attn.{qkv.lower()}.weight"
            if tl_key in tl_state_dict:
                w = tl_state_dict[tl_key]
                out[native_key] = einops.rearrange(
                    w, "head d_model d_head -> (head d_head) d_model"
                )

        # Attention biases: TL [n_heads, d_head] -> Native [n_heads*d_head]
        for qkv in ["Q", "K", "V"]:
            tl_key = f"{tl_prefix}.attn.b_{qkv}"
            native_key = f"{native_prefix}.attn.{qkv.lower()}.bias"
            if tl_key in tl_state_dict:
                b = tl_state_dict[tl_key]
                out[native_key] = einops.rearrange(b, "head d_head -> (head d_head)")

        # Output projection: TL W_O [n_heads, d_head, d_model] -> Native [d_model, n_heads*d_head]
        tl_key = f"{tl_prefix}.attn.W_O"
        native_key = f"{native_prefix}.attn.o.weight"
        if tl_key in tl_state_dict:
            w = tl_state_dict[tl_key]
            out[native_key] = einops.rearrange(w, "head d_head d_model -> d_model (head d_head)")

        # Output projection bias
        tl_key = f"{tl_prefix}.attn.b_O"
        native_key = f"{native_prefix}.attn.o.bias"
        if tl_key in tl_state_dict:
            out[native_key] = tl_state_dict[tl_key]

        # MLP weights: TL W_in [d_model, d_mlp] -> Native fc_in.weight [d_mlp, d_model]
        tl_key = f"{tl_prefix}.mlp.W_in"
        native_key = f"{native_prefix}.mlp.fc_in.weight"
        if tl_key in tl_state_dict:
            out[native_key] = tl_state_dict[tl_key].T

        tl_key = f"{tl_prefix}.mlp.b_in"
        native_key = f"{native_prefix}.mlp.fc_in.bias"
        if tl_key in tl_state_dict:
            out[native_key] = tl_state_dict[tl_key]

        # MLP W_out [d_mlp, d_model] -> fc_out.weight [d_model, d_mlp]
        tl_key = f"{tl_prefix}.mlp.W_out"
        native_key = f"{native_prefix}.mlp.fc_out.weight"
        if tl_key in tl_state_dict:
            out[native_key] = tl_state_dict[tl_key].T

        tl_key = f"{tl_prefix}.mlp.b_out"
        native_key = f"{native_prefix}.mlp.fc_out.bias"
        if tl_key in tl_state_dict:
            out[native_key] = tl_state_dict[tl_key]

    # Final LayerNorm
    if "ln_final.w" in tl_state_dict:
        out["ln_out.weight"] = tl_state_dict["ln_final.w"]
    if "ln_final.b" in tl_state_dict:
        out["ln_out.bias"] = tl_state_dict["ln_final.b"]

    # Unembed: TL W_U [d_model, d_vocab] -> Native head.weight [d_vocab, d_model]
    if "unembed.W_U" in tl_state_dict:
        out["head.weight"] = tl_state_dict["unembed.W_U"].T

    # Unembed bias (rare but handle it)
    if "unembed.b_U" in tl_state_dict:
        out["head.bias"] = tl_state_dict["unembed.b_U"]

    return out
