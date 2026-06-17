from __future__ import annotations

from typing import Dict

import einops
import torch
from transformers import ViTForImageClassification


def convert_vit_weights(
    vit_model: ViTForImageClassification,
    cfg,
    include_embeddings: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert Hugging Face ViTForImageClassification weights into the
    TransformerLens-style state_dict used by HookedVisualEncoder.

    Assumptions:
    - Attention weights are stored as:
        W_Q/W_K/W_V: [n_heads, d_model, d_head]
        W_O:         [n_heads, d_head, d_model]
    - MLP weights follow the same convention as the BERT converter.
    - If include_embeddings=False, this only converts weights that exist
      in your current HookedVisualEncoder (blocks + final LN + classifier).

    Returns
    -------
    Dict[str, torch.Tensor]
        A state dict compatible with model.load_state_dict(..., strict=False).
    """
    state_dict: Dict[str, torch.Tensor] = {}

    vit = vit_model.vit

    # -------------------------
    # Optional embeddings
    # -------------------------
    # Only include these if your TL model actually has matching parameters.
    # In your current code you assign HF embeddings directly, so you can skip them.
    if include_embeddings:
        # HF ViT embeddings:
        #   cls_token:           [1, 1, d_model]
        #   position_embeddings:  [1, seq_len, d_model]
        #   patch_embeddings.projection: Conv2d weights [d_model, C, P, P]
        state_dict["embeddings.cls_token"] = vit.embeddings.cls_token
        state_dict["embeddings.position_embeddings"] = vit.embeddings.position_embeddings
        state_dict["embeddings.patch_embeddings.projection.weight"] = (
            vit.embeddings.patch_embeddings.projection.weight
        )
        state_dict["embeddings.patch_embeddings.projection.bias"] = (
            vit.embeddings.patch_embeddings.projection.bias
        )

    # -------------------------
    # Transformer blocks
    # -------------------------
    for l in range(cfg.n_layers):
        block = vit.layer[l]

        # Pre-attention LN
        state_dict[f"blocks.{l}.ln1.w"] = block.layernorm_before.weight
        state_dict[f"blocks.{l}.ln1.b"] = block.layernorm_before.bias

        # QKV projections
        # HF Linear weight shape: [out_features, in_features] = [d_model, d_model]
        # TL expects per-head tensors.
        q_w = einops.rearrange(
            block.attention.attention.query.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )
        k_w = einops.rearrange(
            block.attention.attention.key.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )
        v_w = einops.rearrange(
            block.attention.attention.value.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )

        q_b = einops.rearrange(
            block.attention.attention.query.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )
        k_b = einops.rearrange(
            block.attention.attention.key.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )
        v_b = einops.rearrange(
            block.attention.attention.value.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )

        state_dict[f"blocks.{l}.attn.W_Q"] = q_w
        state_dict[f"blocks.{l}.attn.W_K"] = k_w
        state_dict[f"blocks.{l}.attn.W_V"] = v_w
        state_dict[f"blocks.{l}.attn.b_Q"] = q_b
        state_dict[f"blocks.{l}.attn.b_K"] = k_b
        state_dict[f"blocks.{l}.attn.b_V"] = v_b

        # Attention output projection
        # HF weight: [d_model, d_model]
        # TL: [n_heads, d_head, d_model]
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            block.attention.output.dense.weight,
            "m (h d) -> h d m",
            h=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = block.attention.output.dense.bias

        # Post-attention / pre-MLP LN
        state_dict[f"blocks.{l}.ln2.w"] = block.layernorm_after.weight
        state_dict[f"blocks.{l}.ln2.b"] = block.layernorm_after.bias

        # MLP
        # HF intermediate.dense: [d_mlp, d_model]
        # Your TL BERT converter stores W_in as [d_model, d_mlp]
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            block.intermediate.dense.weight,
            "mlp model -> model mlp",
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = block.intermediate.dense.bias

        # HF output.dense: [d_model, d_mlp]
        # TL stores W_out as [d_mlp, d_model]
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            block.output.dense.weight,
            "model mlp -> mlp model",
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = block.output.dense.bias

    # -------------------------
    # Final encoder norm
    # -------------------------
    state_dict["layernorm.w"] = vit.layernorm.weight
    state_dict["layernorm.b"] = vit.layernorm.bias

    # -------------------------
    # Classification head
    # -------------------------
    # HF classifier: nn.Linear(d_model, num_labels)
    # TL ClassifierHead forward uses F.linear(residual, self.W.T, self.b),
    # so store W as [d_model, num_labels].
    if hasattr(vit_model, "classifier") and vit_model.classifier is not None:
        state_dict["classifier.W"] = vit_model.classifier.weight.T
        state_dict["classifier.b"] = vit_model.classifier.bias

    return state_dict

def convert_vit_model_weights(
    vit: ViTModel,
    cfg,
    include_embeddings: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert Hugging Face ViTModel weights into the
    TransformerLens-style state_dict used by HookedVisualEncoder.

    Assumptions:
    - Attention weights are stored as:
        W_Q/W_K/W_V: [n_heads, d_model, d_head]
        W_O:         [n_heads, d_head, d_model]
    - MLP weights follow the same convention as the BERT converter.
    - If include_embeddings=False, this only converts weights that exist
      in your current HookedVisualEncoder (blocks + final LN + classifier).

    Returns
    -------
    Dict[str, torch.Tensor]
        A state dict compatible with model.load_state_dict(..., strict=False).
    """
    state_dict: Dict[str, torch.Tensor] = {}

    # -------------------------
    # Optional embeddings
    # -------------------------
    # Only include these if your TL model actually has matching parameters.
    # In your current code you assign HF embeddings directly, so you can skip them.
    if include_embeddings:
        # HF ViT embeddings:
        #   cls_token:           [1, 1, d_model]
        #   position_embeddings:  [1, seq_len, d_model]
        #   patch_embeddings.projection: Conv2d weights [d_model, C, P, P]
        state_dict["embeddings.cls_token"] = vit.embeddings.cls_token
        state_dict["embeddings.position_embeddings"] = vit.embeddings.position_embeddings
        state_dict["embeddings.patch_embeddings.projection.weight"] = (
            vit.embeddings.patch_embeddings.projection.weight
        )
        state_dict["embeddings.patch_embeddings.projection.bias"] = (
            vit.embeddings.patch_embeddings.projection.bias
        )

    # -------------------------
    # Transformer blocks
    # -------------------------
    for l in range(cfg.n_layers):
        block = vit.layer[l]

        # Pre-attention LN
        state_dict[f"blocks.{l}.ln1.w"] = block.layernorm_before.weight
        state_dict[f"blocks.{l}.ln1.b"] = block.layernorm_before.bias

        # QKV projections
        # HF Linear weight shape: [out_features, in_features] = [d_model, d_model]
        # TL expects per-head tensors.
        q_w = einops.rearrange(
            block.attention.attention.query.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )
        k_w = einops.rearrange(
            block.attention.attention.key.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )
        v_w = einops.rearrange(
            block.attention.attention.value.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )

        q_b = einops.rearrange(
            block.attention.attention.query.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )
        k_b = einops.rearrange(
            block.attention.attention.key.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )
        v_b = einops.rearrange(
            block.attention.attention.value.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )

        state_dict[f"blocks.{l}.attn.W_Q"] = q_w
        state_dict[f"blocks.{l}.attn.W_K"] = k_w
        state_dict[f"blocks.{l}.attn.W_V"] = v_w
        state_dict[f"blocks.{l}.attn.b_Q"] = q_b
        state_dict[f"blocks.{l}.attn.b_K"] = k_b
        state_dict[f"blocks.{l}.attn.b_V"] = v_b

        # Attention output projection
        # HF weight: [d_model, d_model]
        # TL: [n_heads, d_head, d_model]
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            block.attention.output.dense.weight,
            "m (h d) -> h d m",
            h=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = block.attention.output.dense.bias

        # Post-attention / pre-MLP LN
        state_dict[f"blocks.{l}.ln2.w"] = block.layernorm_after.weight
        state_dict[f"blocks.{l}.ln2.b"] = block.layernorm_after.bias

        # MLP
        # HF intermediate.dense: [d_mlp, d_model]
        # Your TL BERT converter stores W_in as [d_model, d_mlp]
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            block.intermediate.dense.weight,
            "mlp model -> model mlp",
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = block.intermediate.dense.bias

        # HF output.dense: [d_model, d_mlp]
        # TL stores W_out as [d_mlp, d_model]
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            block.output.dense.weight,
            "model mlp -> mlp model",
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = block.output.dense.bias

    # -------------------------
    # Final encoder norm
    # -------------------------
    state_dict["layernorm.w"] = vit.layernorm.weight
    state_dict["layernorm.b"] = vit.layernorm.bias

    return state_dict
