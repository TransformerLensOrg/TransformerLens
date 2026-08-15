"""Qwen3Next architecture adapter.

Hybrid linear-attention (GatedDeltaNet) + full-attention with sparse MoE MLP.
3 linear-attn layers per 1 full-attn layer. Extends Qwen3 base with
optional attention mapping, MoE MLP, and fold_ln disabled.
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    LinearBridge,
    MoEBridge,
    MoERouterBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


class Qwen3NextArchitectureAdapter(Qwen3ArchitectureAdapter):
    """Hybrid linear-attention + full-attention with sparse MoE MLP.

    Same hybrid design as Qwen3.5 but with MoE instead of dense MLP.
    """

    def __init__(self, cfg: Any) -> None:
        setattr(cfg, "gated_q_proj", True)
        super().__init__(cfg, hybrid=True)

    def _build_mlp_bridge(self):
        """Sparse MoE MLP (router + batched experts + shared expert)."""
        return MoEBridge(
            name="mlp",
            config=self.cfg,
            sparse_required=("gate",),
            submodules={
                # Plain-tensor SparseMoeBlock: router observability comes from
                # the gate submodule. Dense fallback layers (mlp_only_layers /
                # decoder_sparse_step) bind the projections below (#1645).
                "gate": MoERouterBridge(name="gate", optional=True),
                "shared_expert": self._gated_mlp(name="shared_expert", optional=True),
                "shared_expert_gate": LinearBridge(name="shared_expert_gate", optional=True),
                "dense_gate": LinearBridge(name="gate_proj", optional=True),
                "dense_in": LinearBridge(name="up_proj", optional=True),
                "dense_out": LinearBridge(name="down_proj", optional=True),
            },
        )
