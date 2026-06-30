"""Architecture adapter for HF's Mamba2ForCausalLM, plus the effective attention helper."""
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

    # Phases 1-3 are transformer-shaped (component/weight comparison) and don't
    # fit SSMs; component-level coverage lives in integration tests:
    # tests/integration/model_bridge/test_mamba2_adapter.py. Phase 4 (generation
    # + text-quality) needs no component comparison, so it applies.
    applicable_phases: list[int] = [4]

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
    """Compute Mamba-2 effective attention M = L ⊙ (C B^T) for one or all layers.

    Wraps ``SSM2MixerBridge.compute_effective_attention`` so callers don't have
    to repeat the layer index, and resolves SSM layers across all families
    (full Mamba-2 and hybrids like NemotronH / GraniteMoeHybrid) when ``layer``
    is None. Non-SSM layers (attention / MLP / MoE) are skipped structurally.

    Args:
        bridge: A loaded ``TransformerBridge`` whose SSM layers use ``.mixer``.
        cache: ActivationCache from ``run_with_cache`` with in_proj and conv1d
            hooks populated for every SSM layer.
        layer: Specific block index, or None for every SSM layer.
        include_dt_scaling: See ``SSM2MixerBridge.compute_effective_attention``.

    Returns:
        For a single ``layer``: ``[batch, num_heads, seq, seq]``.
        For ``layer=None``: a stacked ``[n_layers, batch, num_heads, seq, seq]``
        tensor when *every* block is an SSM layer (full Mamba-2), otherwise a
        ``{layer_idx: [batch, num_heads, seq, seq]}`` dict over the SSM layers
        (heterogeneous hybrids).

    Raises:
        TypeError: If the requested ``layer`` has no ``SSM2MixerBridge`` mixer.
        RuntimeError: If ``layer=None`` finds no cached SSM mixer layers.

    Example::

        from transformer_lens.model_bridge.supported_architectures.mamba2 import (
            compute_effective_attention,
        )

        M5 = compute_effective_attention(bridge, cache, layer=5)
        M_all = compute_effective_attention(bridge, cache)
    """
    if layer is not None:
        mixer = getattr(bridge.blocks[layer], "mixer", None)
        if not isinstance(mixer, SSM2MixerBridge):
            raise TypeError(
                f"Layer {layer} has no SSM2MixerBridge mixer (got "
                f"{type(mixer).__name__}); compute_effective_attention requires "
                "a Mamba-2 / SSM layer."
            )
        return mixer.compute_effective_attention(
            cache, layer_idx=layer, include_dt_scaling=include_dt_scaling
        )

    # All-layers: enumerate SSM mixer layers structurally (blocks_with checks
    # _modules, so attention layers without a `.mixer` slot are skipped). On a
    # hybrid where every block carries a passthrough `.mixer` (NemotronH), the
    # cached-hook check below filters out the non-Mamba mixers.
    results: Dict[int, torch.Tensor] = {}
    for layer_idx, block in bridge.blocks_with("mixer"):
        mixer = block.mixer
        if not isinstance(mixer, SSM2MixerBridge):
            continue
        in_proj_key = f"blocks.{layer_idx}.mixer.in_proj.hook_out"
        conv1d_key = f"blocks.{layer_idx}.mixer.conv1d.hook_out"
        if in_proj_key not in cache or conv1d_key not in cache:
            continue
        results[layer_idx] = mixer.compute_effective_attention(
            cache, layer_idx=layer_idx, include_dt_scaling=include_dt_scaling
        )

    if not results:
        raise RuntimeError(
            "compute_effective_attention found no SSM mixer layers with cached "
            "in_proj/conv1d hooks. Run `run_with_cache()` on a Mamba-2 / SSM "
            "bridge first."
        )

    n_blocks = len(bridge.blocks)
    if len(results) == n_blocks and list(results) == list(range(n_blocks)):
        return torch.stack([results[i] for i in range(n_blocks)], dim=0)
    return results
