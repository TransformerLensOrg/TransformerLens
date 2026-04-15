"""Qwen3Next architecture adapter.

Hybrid linear-attention (GatedDeltaNet) + full-attention with sparse MoE MLP.
3 linear-attn layers per 1 full-attn layer. Extends Qwen3 base with
optional attention mapping, MoE MLP, and fold_ln disabled.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import MoEBridge
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


class Qwen3NextArchitectureAdapter(Qwen3ArchitectureAdapter):
    """Hybrid linear-attention + full-attention with sparse MoE MLP.

    Same hybrid design as Qwen3.5 but with MoE instead of dense MLP.
    Inherits Qwen3 config/attention structure.
    """

    def __init__(self, cfg: Any) -> None:
        ArchitectureAdapter.__init__(self, cfg)
        self._setup_qwen3_config(cfg)
        self.supports_fold_ln = False
        setattr(self.cfg, "gated_q_proj", True)  # q_proj outputs [Q|gate] interleaved per head
        self.weight_processing_conversions: dict = {}
        self.component_mapping = self._build_component_mapping(hybrid=True)

    def _build_mlp_bridge(self):
        """Sparse MoE MLP (router + batched experts + shared expert)."""
        return MoEBridge(name="mlp", config=self.cfg)

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slice query half from gated q_proj.weight for weight-space analysis."""
        return self._preprocess_gated_q_proj(state_dict, self.cfg.n_heads, self.cfg.d_head)
