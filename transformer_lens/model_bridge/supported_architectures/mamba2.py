"""Mamba-2 architecture adapter.

Wraps HF's Mamba2ForCausalLM. Separate from the Mamba-1 adapter because the
Mamba-2 mixer has a fundamentally different submodule set:
- No `x_proj`/`dt_proj` (fused into `in_proj`)
- Has `inner_norm` (a `MambaRMSNormGated` taking two inputs)
- Has `dt_bias` nn.Parameter (Mamba-1 uses `dt_proj` bias instead)
- Multi-head structure with `num_heads`, `head_dim`, `n_groups`

Shared with Mamba-1: `SSMBlockBridge`, `DepthwiseConv1DBridge`, and the
`is_stateful` generation fallback.
"""
from typing import Any

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


class Mamba2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Mamba-2 models (Mamba2ForCausalLM)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Core config setup
        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.positional_embedding_type = "none"
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.final_rms = True
        self.cfg.is_stateful = True

        # Mamba-2 config fields propagated via _HF_PASSTHROUGH_ATTRS:
        # state_size, conv_kernel, expand, n_groups, chunk_size.
        # num_heads (-> n_heads) and head_dim (-> d_head) are mapped by
        # map_default_transformer_lens_config.

        # intermediate_size is NOT a field on Mamba2Config — compute it from
        # expand * hidden_size. conv_dim is likewise derived. Both are stored
        # as dynamic attributes on cfg via setattr (cfg is duck-typed for
        # architecture-specific fields; mypy can't see these statically).
        expand = getattr(self.cfg, "expand", 2)
        hidden_size = self.cfg.d_model
        intermediate_size = expand * hidden_size
        setattr(self.cfg, "intermediate_size", intermediate_size)

        num_heads = self.cfg.n_heads
        state_size = getattr(self.cfg, "state_size", 128)
        n_groups = getattr(self.cfg, "n_groups", 1)
        conv_dim = intermediate_size + 2 * n_groups * state_size
        setattr(self.cfg, "conv_dim", conv_dim)

        # Plan Step 2.4: HF's in_proj 5-way split has two d_mlp slots that are
        # always size 0 in current configs. Stored for test introspection; if
        # a future HF release introduces non-zero d_mlp, the integration test
        # assertion will catch the in_proj shape mismatch.
        in_proj_out_features = 2 * intermediate_size + conv_dim + num_heads
        setattr(self.cfg, "expected_in_proj_out_features", in_proj_out_features)

        # No attention weight conversions — Mamba-2 has no Q/K/V/O.
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
                            # HF calls this submodule `norm` on the mixer; we
                            # rename to `inner_norm` to disambiguate from the
                            # block-level `norm`. The `name=` arg is the
                            # remote HF path, not the TL name.
                            "inner_norm": GatedRMSNormBridge(name="norm"),
                            "out_proj": LinearBridge(name="out_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="backbone.norm_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
