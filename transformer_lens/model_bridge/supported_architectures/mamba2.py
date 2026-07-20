"""Architecture adapter for HF's Mamba2ForCausalLM, plus the effective attention helper."""
import warnings
from typing import Any, Dict, Optional, Union

import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedRMSNormBridge,
    LinearBridge,
    RMSNormalizationBridge,
    SSM2MixerBridge,
    SSMBlockBridge,
    UnembeddingBridge,
)


class Mamba2ArchitectureAdapter(ArchitectureAdapter):
    """Wraps HF's Mamba2ForCausalLM.

    Differs from Mamba-1 at the mixer level: fused in_proj (no x_proj/dt_proj),
    two-input inner norm, multi-head structure with ``num_heads``/``head_dim``/
    ``n_groups``, and an ``[num_heads]``-shaped ``dt_bias``. Shares
    ``SSMBlockBridge``, ``DepthwiseConv1DBridge``, and the stateful generation
    loop with Mamba-1.
    """

    # White-box forward: P1 is exact vs raw HF (mixer delegates to HF); P2/P3 skip
    # without a HookedTransformer; P4 is generation.
    applicable_phases: list[int] = [1, 2, 3, 4]

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.positional_embedding_type = "none"
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.final_rms = True
        self.cfg.is_stateful = True

        # Most SSM config fields come from _HF_PASSTHROUGH_ATTRS. Mamba2Config
        # has no `intermediate_size` field, so we compute it from expand and
        # derive conv_dim from that. setattr() avoids mypy attr-defined errors
        # since cfg is duck-typed for architecture-specific extensions.
        expand = getattr(self.cfg, "expand", 2)
        hidden_size = self.cfg.d_model
        intermediate_size = expand * hidden_size
        setattr(self.cfg, "intermediate_size", intermediate_size)

        num_heads = self.cfg.n_heads
        state_size = getattr(self.cfg, "state_size", 128)
        n_groups = getattr(self.cfg, "n_groups", 1)
        conv_dim = intermediate_size + 2 * n_groups * state_size
        setattr(self.cfg, "conv_dim", conv_dim)

        # HF splits in_proj 5 ways but two d_mlp slots are always size 0.
        # Stored so the integration test can catch a future HF change that
        # introduces non-zero d_mlp.
        in_proj_out_features = 2 * intermediate_size + conv_dim + num_heads
        setattr(self.cfg, "expected_in_proj_out_features", in_proj_out_features)

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="backbone.embeddings"),
            "blocks": SSMBlockBridge(
                name="backbone.layers",
                submodules={
                    "norm": RMSNormalizationBridge(name="norm", config=self.cfg),
                    "mixer": SSM2MixerBridge(
                        name="mixer",
                        config=self.cfg,
                        submodules={
                            "in_proj": LinearBridge(name="in_proj"),
                            "conv1d": DepthwiseConv1DBridge(name="conv1d"),
                            # TL calls this "inner_norm" to disambiguate from
                            # the block-level norm; name="norm" is the HF path.
                            "inner_norm": GatedRMSNormBridge(name="norm"),
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
        """Build a cache for the stateful generation loop."""
        from transformers.cache_utils import DynamicCache
        from transformers.models.mamba2 import modeling_mamba2

        cache_cls = getattr(modeling_mamba2, "Mamba2Cache", None)
        if cache_cls is not None:
            return cache_cls(hf_model.config, batch_size, device=device, dtype=dtype)

        return DynamicCache(config=hf_model.config)


def compute_effective_attention(
    bridge: TransformerBridge,
    cache: ActivationCache,
    layer: Optional[int] = None,
    include_dt_scaling: bool = False,
) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
    """Mamba-2 effective attention for one or all layers.

    .. deprecated::
        Use the family-agnostic ``cache.compute_ssm_effective_attention(layer=...)``
        instead. This thin wrapper delegates to it and ignores ``bridge`` (the
        cache already knows its model).
    """
    warnings.warn(
        "mamba2.compute_effective_attention is deprecated; use "
        "cache.compute_ssm_effective_attention(layer=..., include_dt_scaling=...).",
        DeprecationWarning,
        stacklevel=2,
    )
    return cache.compute_ssm_effective_attention(layer=layer, include_dt_scaling=include_dt_scaling)


def compute_ssm_state(
    bridge: TransformerBridge,
    cache: ActivationCache,
    layer: Optional[int] = None,
    time_step: Optional[int] = None,
) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
    """Reconstruct the recurrent SSM state ``S`` for one or all Mamba-2 layers.

    .. deprecated::
        Use ``cache.compute_ssm_state(layer=..., time_step=...)`` instead. This
        thin wrapper delegates to it and ignores ``bridge``.
    """
    warnings.warn(
        "mamba2.compute_ssm_state is deprecated; use "
        "cache.compute_ssm_state(layer=..., time_step=...).",
        DeprecationWarning,
        stacklevel=2,
    )
    return cache.compute_ssm_state(layer=layer, time_step=time_step)
