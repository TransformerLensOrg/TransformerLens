"""Architecture adapter for HF's Mamba2ForCausalLM, plus the effective attention helper."""
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
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

if TYPE_CHECKING:
    from transformer_lens.model_bridge.bridge import TransformerBridge


class Mamba2ArchitectureAdapter(ArchitectureAdapter):
    """Wraps HF's Mamba2ForCausalLM.

    Differs from Mamba-1 at the mixer level: fused in_proj (no x_proj/dt_proj),
    two-input inner norm, multi-head structure with ``num_heads``/``head_dim``/
    ``n_groups``, and an ``[num_heads]``-shaped ``dt_bias``. Shares
    ``SSMBlockBridge``, ``DepthwiseConv1DBridge``, and the stateful generation
    loop with Mamba-1.
    """

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
        """Build a Mamba2Cache for the stateful generation loop."""
        from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

        return Mamba2Cache(hf_model.config, batch_size, device=device, dtype=dtype)


def compute_effective_attention(
    bridge: "TransformerBridge",
    cache: Mapping[str, torch.Tensor],
    layer: Optional[int] = None,
    include_dt_scaling: bool = False,
) -> torch.Tensor:
    """Compute Mamba-2 effective attention M = L ⊙ (C B^T) for one or all layers.

    Wraps ``SSM2MixerBridge.compute_effective_attention`` so callers don't have
    to repeat the layer index, and adds all-layers stacking when ``layer`` is
    None.

    Args:
        bridge: A loaded Mamba-2 ``TransformerBridge``.
        cache: ActivationCache from ``run_with_cache`` with in_proj and conv1d
            hooks populated for every requested layer.
        layer: Specific block index, or None for all layers stacked.
        include_dt_scaling: See ``SSM2MixerBridge.compute_effective_attention``.

    Returns:
        Shape ``[batch, num_heads, seq, seq]`` for a single layer, or
        ``[n_layers, batch, num_heads, seq, seq]`` when layer is None.

    Raises:
        TypeError: If any targeted block's mixer isn't an ``SSM2MixerBridge``.

    Example::

        from transformer_lens.model_bridge.supported_architectures.mamba2 import (
            compute_effective_attention,
        )

        M5 = compute_effective_attention(bridge, cache, layer=5)
        M_all = compute_effective_attention(bridge, cache)
    """
    if layer is not None:
        mixer = bridge.blocks[layer].mixer
        if not isinstance(mixer, SSM2MixerBridge):
            raise TypeError(
                f"Layer {layer} mixer is {type(mixer).__name__}, not "
                "SSM2MixerBridge. compute_effective_attention requires a "
                "Mamba-2 bridge."
            )
        return mixer.compute_effective_attention(
            cache, layer_idx=layer, include_dt_scaling=include_dt_scaling
        )

    matrices = []
    for layer_idx, block in enumerate(bridge.blocks):
        mixer = block.mixer
        if not isinstance(mixer, SSM2MixerBridge):
            raise TypeError(
                f"Layer {layer_idx} mixer is {type(mixer).__name__}, not "
                "SSM2MixerBridge. compute_effective_attention requires a "
                "Mamba-2 bridge."
            )
        matrices.append(
            mixer.compute_effective_attention(
                cache, layer_idx=layer_idx, include_dt_scaling=include_dt_scaling
            )
        )
    return torch.stack(matrices, dim=0)
