"""Qwen3.5 vision encoder bridge components.

Bridges for the Qwen3.5 vision tower (``model.visual``, a ``Qwen3_5VisionModel``).
Unlike SigLIP/CLIP towers, the Qwen vision tower is structured as:
- patch_embed: Conv3d patch embedding (video-capable)
- pos_embed: learned position embedding
- rotary_pos_emb: vision RoPE (no params; not bridged)
- blocks[]: norm1/norm2 (LayerNorm), attn (fused qkv + proj), mlp (linear_fc1/linear_fc2)
- merger: Qwen3_5VisionPatchMerger — the vision->text projector, bridged separately as
  ``vision_projector`` so each HF module has exactly one owning bridge.
"""
from typing import Any, Dict, Optional

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class Qwen3_5VisionBlockBridge(GeneralizedComponent):
    """Bridge for a single Qwen3.5 vision block.

    Submodules: norm1/norm2 (LayerNorm), attn (fused qkv + proj), mlp (linear_fc1/linear_fc2).
    Norms are wrapped as black-box GeneralizedComponents (hook around the native LayerNorm,
    no recomputation), since NormalizationBridge recomputes the norm and would need the
    vision LayerNorm's eps — keeping them native preserves exact parity.
    """

    is_list_item: bool = True
    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_post": "hook_out",
        "hook_attn_in": "attn.hook_in",
        "hook_attn_out": "attn.hook_out",
        "hook_mlp_in": "mlp.hook_in",
        "hook_mlp_out": "mlp.hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the Qwen3.5 vision block bridge.

        Args:
            name: Component name relative to the vision tower (e.g., "blocks").
            config: Optional configuration object.
            submodules: Optional override/extension of the default submodules.
        """
        default_submodules: Dict[str, GeneralizedComponent] = {
            "norm1": GeneralizedComponent(name="norm1"),
            "norm2": GeneralizedComponent(name="norm2"),
            "attn": GeneralizedComponent(
                name="attn",
                submodules={
                    "qkv": LinearBridge(name="qkv"),
                    "proj": LinearBridge(name="proj"),
                },
            ),
            "mlp": GeneralizedComponent(
                name="mlp",
                submodules={
                    "linear_fc1": LinearBridge(name="linear_fc1"),
                    "linear_fc2": LinearBridge(name="linear_fc2"),
                },
            ),
        }
        if submodules:
            default_submodules.update(submodules)
        super().__init__(name, config, submodules=default_submodules)


class Qwen3_5VisionEncoderBridge(GeneralizedComponent):
    """Bridge for the complete Qwen3.5 vision tower (``model.visual``).

    Decomposes patch embedding, position embedding, and the transformer blocks for
    interpretability. The merger (vision->text projection) is bridged separately as the
    adapter's ``vision_projector`` to keep one bridge per HF module.
    """

    hook_aliases = {
        "hook_vision_embed": "patch_embed.hook_out",
        "hook_vision_out": "hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the Qwen3.5 vision encoder bridge.

        Args:
            name: The HF module path for the vision tower (e.g., "model.visual").
            config: Optional configuration object.
            submodules: Optional override/extension of the default submodules.
        """
        default_submodules: Dict[str, GeneralizedComponent] = {
            "patch_embed": GeneralizedComponent(name="patch_embed"),
            "pos_embed": GeneralizedComponent(name="pos_embed"),
            "blocks": Qwen3_5VisionBlockBridge(name="blocks"),
        }
        if submodules:
            default_submodules.update(submodules)
        super().__init__(name, config, submodules=default_submodules)
