"""Mamba-1 architecture adapter.

Wraps HF's MambaForCausalLM (state-spaces/mamba-*-hf). Uses the wrap-don't-
reimplement pattern: the HF MambaMixer.slow_forward runs as-is; submodule
bridges on in_proj/conv1d/x_proj/dt_proj/out_proj capture projection activations.

Mamba has no attention, no position embeddings, and stateful cache semantics
(MambaCache mutated in place). Phase 1 delegates generation to hf_generate via
the `is_stateful` flag to sidestep KV-cache incompatibility.
"""
from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    LinearBridge,
    RMSNormalizationBridge,
    SSMBlockBridge,
    SSMMixerBridge,
    UnembeddingBridge,
)


class MambaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Mamba-1 models (MambaForCausalLM)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Core config setup
        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.positional_embedding_type = "none"
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.final_rms = True

        # Mamba-specific SSM config fields (state_size, conv_kernel, expand,
        # time_step_rank, intermediate_size) are propagated from the HF config
        # via _HF_PASSTHROUGH_ATTRS in sources/transformers.py, so they are
        # already available on self.cfg.

        # Stateful flag: signals that generation should delegate to hf_generate
        # because the HF cache type (MambaCache) is not compatible with the
        # standard KV-cache path in TransformerBridge.generate().
        self.cfg.is_stateful = True

        # No attention weight conversions — Mamba has no Q/K/V/O.
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="backbone.embeddings"),
            "blocks": SSMBlockBridge(
                name="backbone.layers",
                submodules={
                    "norm": RMSNormalizationBridge(name="norm", config=self.cfg),
                    "mixer": SSMMixerBridge(
                        name="mixer",
                        config=self.cfg,
                        submodules={
                            "in_proj": LinearBridge(name="in_proj"),
                            "conv1d": DepthwiseConv1DBridge(name="conv1d"),
                            "x_proj": LinearBridge(name="x_proj"),
                            "dt_proj": LinearBridge(name="dt_proj"),
                            "out_proj": LinearBridge(name="out_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="backbone.norm_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
