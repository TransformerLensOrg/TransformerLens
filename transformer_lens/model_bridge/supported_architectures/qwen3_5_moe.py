"""Qwen3.5-MoE architecture adapter.

Hybrid linear-attention (GatedDeltaNet) + full-attention with sparse MoE MLP
(256 experts, top-8 routing, shared expert in public checkpoints). Same hybrid
design as Qwen3.5 dense and the same MoE block family as Qwen3-Next.

Two adapters: text-only ``Qwen3_5MoeForCausalLM`` and the vision-language
``Qwen3_5MoeForConditionalGeneration`` (text backbone nested under
``model.language_model`` plus the Qwen3.5 vision tower).
"""

from transformer_lens.model_bridge.generalized_components import (
    LinearBridge,
    MoEBridge,
    MoERouterBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
    Qwen3_5ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_5_multimodal import (
    Qwen3_5MultimodalArchitectureAdapter,
)


def _sparse_moe_mlp(adapter):
    """Qwen3.5 sparse MoE block: 3-tuple router, batched experts, shared expert."""
    return MoEBridge(
        name="mlp",
        config=adapter.cfg,
        submodules={
            "gate": MoERouterBridge(name="gate"),
            "experts": MoEBridge(name="experts", config=adapter.cfg),
            "shared_expert": adapter._gated_mlp(name="shared_expert"),
            "shared_expert_gate": LinearBridge(name="shared_expert_gate"),
        },
    )


class Qwen3_5MoeArchitectureAdapter(Qwen3_5ArchitectureAdapter):
    """Text-only Qwen3.5-MoE: hybrid GatedDeltaNet + full attention, sparse MoE MLP."""

    _multimodal_arch_name: str = "Qwen3_5MoeForConditionalGeneration"

    def _build_mlp_bridge(self):
        """Sparse MoE MLP (router + batched experts + shared expert)."""
        return _sparse_moe_mlp(self)


class Qwen3_5MoeMultimodalArchitectureAdapter(Qwen3_5MultimodalArchitectureAdapter):
    """Vision-language adapter for Qwen3_5MoeForConditionalGeneration.

    Reuses the Qwen3.5 multimodal wiring (language model under
    ``model.language_model`` + vision tower) with the MLP swapped for sparse MoE.
    """

    def _build_mlp_bridge(self):
        """Sparse MoE MLP (router + batched experts + shared expert)."""
        return _sparse_moe_mlp(self)
