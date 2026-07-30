"""Qwen2-MoE architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    LinearBridge,
    MoEBridge,
    MoERouterBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)


class Qwen2MoeRouterBridge(MoERouterBridge):
    """Tuple-preserving router bridge for ``Qwen2MoeTopKRouter``."""


class Qwen2MoeArchitectureAdapter(Qwen2ArchitectureAdapter):
    """Architecture adapter for Qwen2-MoE models.

    Qwen2-MoE uses the Qwen2 attention stack plus a sparse MoE MLP with an
    always-on shared expert path.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen2-MoE architecture adapter."""
        super().__init__(cfg)

        self.cfg.attn_implementation = "eager"

        if self.component_mapping is None:
            raise ValueError("Qwen2 component mapping was not initialized")

        blocks = self.component_mapping["blocks"]
        blocks.submodules["mlp"] = MoEBridge(
            name="mlp",
            config=self.cfg,
            submodules={
                "gate": Qwen2MoeRouterBridge(name="gate"),
                "experts": MoEBridge(name="experts", config=self.cfg),
                "shared_expert": self._gated_mlp(name="shared_expert"),
                "shared_expert_gate": LinearBridge(name="shared_expert_gate"),
            },
        )
