"""Qwen3.5 vision-tower bridges (``model.visual``).

The Qwen vision tower differs from SigLIP/CLIP, so it needs its own bridge. The merger
(vision->text projector) is bridged separately as the adapter's ``vision_projector``, and
the paramless ``rotary_pos_emb`` is left native.
"""
from typing import Any, Dict, Optional

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class Qwen3_5VisionBlockBridge(GeneralizedComponent):
    """Bridge for a single Qwen3.5 vision block.

    Norms stay black-box (hooked, not recomputed): NormalizationBridge would recompute
    with the wrong eps and break parity.
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
    """Bridge for the Qwen3.5 vision tower (``model.visual``); merger is bridged separately."""

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
        default_submodules: Dict[str, GeneralizedComponent] = {
            "patch_embed": GeneralizedComponent(name="patch_embed"),
            "pos_embed": GeneralizedComponent(name="pos_embed"),
            "blocks": Qwen3_5VisionBlockBridge(name="blocks"),
        }
        if submodules:
            default_submodules.update(submodules)
        super().__init__(name, config, submodules=default_submodules)
