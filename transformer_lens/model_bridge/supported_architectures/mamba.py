"""Architecture adapter for HF's MambaForCausalLM (Mamba-1)."""
from typing import Any

import torch

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
    """Wraps HF's MambaForCausalLM. No attention, no positional embeddings.

    SSM config fields (state_size, conv_kernel, expand, time_step_rank,
    intermediate_size) are propagated from the HF config via
    ``_HF_PASSTHROUGH_ATTRS`` in sources/transformers.py.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.positional_embedding_type = "none"
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.final_rms = True

        # Routes bridge.generate() through the dedicated SSM cache loop.
        self.cfg.is_stateful = True

        # No Q/K/V/O weights to rearrange.
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

    def create_stateful_cache(
        self,
        hf_model: Any,
        batch_size: int,
        device: Any,
        dtype: torch.dtype,
    ) -> Any:
        """Build a MambaCache for the stateful generation loop."""
        from transformers.models.mamba.modeling_mamba import MambaCache

        return MambaCache(hf_model.config, batch_size, device=device, dtype=dtype)
