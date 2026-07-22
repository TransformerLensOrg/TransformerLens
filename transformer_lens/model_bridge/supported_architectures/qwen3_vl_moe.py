"""Qwen3-VL-MoE architecture adapter.

Alibaba's Qwen3-VL-MoE (``Qwen3VLMoeForConditionalGeneration``): the
Qwen3-VL layout (DeepStack vision tower, interleaved-mRoPE text decoder)
with a sparse-MoE MLP — batched experts plus a parameter-only top-k
router returning a (logits, scores, indices) tuple, so the router stays
unwrapped and MoEBridge delegates the block. Layers listed in
``mlp_only_layers`` hold a dense gated MLP under the same name.
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    MoEBridge,
    MoERouterBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_vl import (
    Qwen3VLArchitectureAdapter,
)


class Qwen3VLMoeArchitectureAdapter(Qwen3VLArchitectureAdapter):
    """Architecture adapter for Qwen3VLMoeForConditionalGeneration models."""

    def _build_mlp_bridge(self) -> Any:
        """Sparse MoE block; gate/experts absent on dense mlp_only_layers."""
        return MoEBridge(
            name="mlp",
            config=self.cfg,
            submodules={
                "gate": MoERouterBridge(name="gate", optional=True),
                "experts": MoEBridge(name="experts", config=self.cfg, optional=True),
            },
        )
