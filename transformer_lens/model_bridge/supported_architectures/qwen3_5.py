"""Qwen3.5 architecture adapter.

Hybrid linear-attention (GatedDeltaNet) + full-attention with dense gated MLP.
3 linear-attn layers per 1 full-attn layer. Extends Qwen3 base with
optional attention mapping and fold_ln disabled.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


class Qwen3_5ArchitectureAdapter(Qwen3ArchitectureAdapter):
    """Hybrid linear-attention + full-attention with dense gated MLP.

    Inherits Qwen3 config/attention/MLP structure. Differences:
    - supports_fold_ln = False (LN target varies by layer type)
    - Attention is optional (absent on linear-attention layers)
    - Gated q_proj (2x wide) requires preprocess_weights slicing
    - No weight_processing_conversions until attn is fully wired
    """

    def __init__(self, cfg: Any) -> None:
        # Call grandparent to set self.cfg, then configure ourselves
        ArchitectureAdapter.__init__(self, cfg)
        self._setup_qwen3_config(cfg)
        self.supports_fold_ln = False
        setattr(self.cfg, "gated_q_proj", True)  # q_proj outputs [Q|gate] interleaved per head
        self.weight_processing_conversions: dict = {}
        self.component_mapping = self._build_component_mapping(hybrid=True)

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Swap multimodal Qwen3_5Config for text-only Qwen3_5TextConfig.

        Published checkpoints carry architectures=['Qwen3_5ForConditionalGeneration'].
        We replace config with text_config so AutoModelForCausalLM loads the
        text-only Qwen3_5ForCausalLM.
        """
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "text_config"):
            model_kwargs["config"] = config.text_config

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slice query half from gated q_proj.weight for weight-space analysis.

        In processed mode, W_Q is the pure query projection (for composition
        scores, logit lens). Gate signal available in unprocessed mode via
        hook_q_gate.
        """
        return self._preprocess_gated_q_proj(state_dict, self.cfg.n_heads, self.cfg.d_head)
