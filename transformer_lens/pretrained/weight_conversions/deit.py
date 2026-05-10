from __future__ import annotations

from typing import Dict

import einops
import torch
from transformers import DeiTForImageClassificationWithTeacher


def convert_deit_weights(
    deit_model: DeiTForImageClassificationWithTeacher,
    cfg,
    include_embeddings: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert a Hugging Face DeiTForImageClassificationWithTeacher checkpoint
    into a TransformerLens-style state_dict.

    Assumed printed HF structure:
        deit_model.deit.encoder.layer[l]
        deit_model.deit.embeddings
        deit_model.deit.layernorm
        deit_model.cls_classifier
        deit_model.distillation_classifier

    TransformerLens-style target names assumed:
        blocks.{l}.ln1.{w,b}
        blocks.{l}.attn.{W_Q,W_K,W_V,b_Q,b_K,b_V,W_O,b_O}
        blocks.{l}.ln2.{w,b}
        blocks.{l}.mlp.{W_in,b_in,W_out,b_out}
        layernorm.{w,b}
        classifier.{W,b}
        distillation_classifier.{W,b}
    """
    state_dict: Dict[str, torch.Tensor] = {}

    deit = deit_model.deit

    if include_embeddings:
        # Keep only if your TL module actually has these parameters.
        state_dict["embeddings.cls_token"] = deit.embeddings.cls_token
        state_dict["embeddings.position_embeddings"] = deit.embeddings.position_embeddings
        state_dict["embeddings.patch_embeddings.projection.weight"] = (
            deit.embeddings.patch_embeddings.projection.weight
        )
        state_dict["embeddings.patch_embeddings.projection.bias"] = (
            deit.embeddings.patch_embeddings.projection.bias
        )

    for l in range(cfg.n_layers):
        block = deit.encoder.layer[l]

        # -------------------------
        # LayerNorm before attention
        # -------------------------
        state_dict[f"blocks.{l}.ln1.w"] = block.layernorm_before.weight
        state_dict[f"blocks.{l}.ln1.b"] = block.layernorm_before.bias

        # -------------------------
        # Attention QKV
        # HF structure:
        #   block.attention.attention.query
        #   block.attention.attention.key
        #   block.attention.attention.value
        # Each is nn.Linear(d_model, d_model)
        # -------------------------
        query = block.attention.attention.query
        key = block.attention.attention.key
        value = block.attention.attention.value

        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            query.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            key.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            value.weight,
            "(h d) m -> h m d",
            h=cfg.n_heads,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            query.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            key.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            value.bias,
            "(h d) -> h d",
            h=cfg.n_heads,
        )

        # -------------------------
        # Attention output projection
        # HF:
        #   block.attention.output.dense
        # weight shape: [d_model, d_model]
        # TL expects [n_heads, d_head, d_model]
        # -------------------------
        attn_out = block.attention.output.dense
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            attn_out.weight,
            "m (h d) -> h d m",
            h=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = attn_out.bias

        # -------------------------
        # LayerNorm after attention
        # -------------------------
        state_dict[f"blocks.{l}.ln2.w"] = block.layernorm_after.weight
        state_dict[f"blocks.{l}.ln2.b"] = block.layernorm_after.bias

        # -------------------------
        # MLP
        # HF:
        #   block.intermediate.dense
        #   block.output.dense
        # -------------------------
        fc1 = block.intermediate.dense
        fc2 = block.output.dense

        # store W_in as [d_model, d_mlp]
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            fc1.weight,
            "mlp model -> model mlp",
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = fc1.bias

        # store W_out as [d_mlp, d_model]
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            fc2.weight,
            "model mlp -> mlp model",
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = fc2.bias

    # -------------------------
    # Final encoder norm
    # -------------------------
    state_dict["layernorm.w"] = deit.layernorm.weight
    state_dict["layernorm.b"] = deit.layernorm.bias

    # -------------------------
    # Classification heads
    # -------------------------
    if hasattr(deit_model, "cls_classifier") and deit_model.cls_classifier is not None:
        state_dict["classifier.W"] = deit_model.cls_classifier.weight.T
        state_dict["classifier.b"] = deit_model.cls_classifier.bias

    if (
        hasattr(deit_model, "distillation_classifier")
        and deit_model.distillation_classifier is not None
    ):
        state_dict["distillation_classifier.W"] = deit_model.distillation_classifier.weight.T
        state_dict["distillation_classifier.b"] = deit_model.distillation_classifier.bias

    return state_dict
