"""BD3LM (Block Diffusion Language Model) architecture adapter."""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    SymbolicBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.block import (
    DelegatedAttentionBlockBridge,
)


class BD3LMArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for BD3LM (Block Diffusion LM, ICLR 2025).

    BD3LM uses adaLN conditioning on diffusion timesteps, a custom Rotary
    embedding, joint QKV projections, and non-causal block-diffusion masking.
    Because adaLN modulation varies per-timestep, it cannot be folded into
    weights — the adapter uses ``DelegatedAttentionBlockBridge`` to delegate
    each ``DDiTBlock.forward()`` wholesale to the original HF module.
    Hooks fire at block boundaries and on mapped submodules.
    """

    # Phases 1–3 cover component mapping, weight conversion, and forward-pass
    # parity.  Phase 4 (autoregressive generation) is excluded because BD3LM
    # uses iterative diffusion sampling.
    applicable_phases: list[int] = [1, 2, 3]

    # BD3LM uses diffusion sampling, not autoregressive generation.
    supports_generation: bool = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # ── Config attributes ──────────────────────────────────────────
        self.cfg.normalization_type = "LN"
        self.cfg.uses_rms_norm = False
        self.cfg.positional_embedding_type = "none"  # custom Rotary, not HF
        self.cfg.gated_mlp = False  # standard GELU MLP, not gated
        self.cfg.attn_only = False
        self.cfg.final_rms = False  # final norm is custom LayerNorm

        # BD3LM-specific config fields.  These live on the HF config and are
        # forwarded via _HF_PASSTHROUGH_ATTRS; we also store them explicitly
        # so unit tests can assert them without loading a real model.
        block_size = getattr(self.cfg, "block_size", 4)
        setattr(self.cfg, "block_size", block_size)

        cond_dim = getattr(self.cfg, "cond_dim", 128)
        setattr(self.cfg, "cond_dim", cond_dim)

        adaln = getattr(self.cfg, "adaln", True)
        setattr(self.cfg, "adaln", adaln)

        cross_attn = getattr(self.cfg, "cross_attn", True)
        setattr(self.cfg, "cross_attn", cross_attn)

        # Compute d_mlp from the MLP ratio (hardcoded to 4 in DDiTBlock).
        mlp_ratio = 4
        d_mlp = mlp_ratio * self.cfg.d_model
        setattr(self.cfg, "d_mlp", d_mlp)

        # ── Weight processing conversions ──────────────────────────────
        # Wrap-don't-reimplement: no weight rearrangement needed since we
        # delegate forward to the original modules.
        self.weight_processing_conversions = {}

        # ── Component mapping ──────────────────────────────────────────
        # DelegatedAttentionBlockBridge delegates forward() wholesale to the
        # original DDiTBlock (wrap-don't-reimplement), but exposes standard
        # attn/mlp-shaped hook aliases (hook_resid_mid, hook_mlp_in) instead
        # of the SSM-specific hook_mixer_in alias that SSMBlockBridge would
        # have incorrectly implied for this attn+mlp architecture. Submodule
        # bridges map to the HF module paths relative to each block.
        #
        # Module hierarchy (HF paths relative to backbone.blocks[i]):
        #   norm1          → pre-attention LayerNorm
        #   attn_qkv       → joint Q/K/V projection (no bias)
        #   attn_out        → output projection (no bias)
        #   adaLN_modulation → conditioning linear (cond_dim → 6*dim)
        #   norm2          → pre-MLP LayerNorm
        #   mlp.0          → MLP in-projection (bias)
        #   mlp.2          → MLP out-projection (bias)
        self.component_mapping = {
            "embed": EmbeddingBridge(name="backbone.vocab_embed"),
            "blocks": DelegatedAttentionBlockBridge(
                name="backbone.blocks",
                submodules={
                    "ln1": NormalizationBridge(name="norm1", config=self.cfg),
                    "attn": SymbolicBridge(
                        submodules={
                            "qkv": LinearBridge(name="attn_qkv"),
                            "o": LinearBridge(name="attn_out"),
                        },
                    ),
                    "adaln_modulation": LinearBridge(name="adaLN_modulation"),
                    "ln2": NormalizationBridge(name="norm2", config=self.cfg),
                    "mlp": MLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="0"),
                            "out": LinearBridge(name="2"),
                        },
                    ),
                },
            ),
            "sigma_map": MLPBridge(
                name="backbone.sigma_map.mlp",
                config=self.cfg,
                submodules={
                    "in": LinearBridge(name="0"),
                    "out": LinearBridge(name="2"),
                },
            ),
            "ln_final": NormalizationBridge(
                name="backbone.output_layer.norm_final", config=self.cfg
            ),
            "unembed": UnembeddingBridge(name="backbone.output_layer.linear"),
        }

    def prepare_model(self, hf_model: Any) -> None:
        """Patch BD3LM quirks that prevent standard bridge construction.

        Three issues must be fixed before the bridge can wrap the model:

        1. ``vocab_embed`` is an ``nn.Parameter``, not ``nn.Embedding``, so it
           lacks a ``.weight`` attribute that ``EmbeddingBridge`` expects.
        2. The ``flex`` attention backend crashes on CPU; fall back to ``sdpa``
           and regenerate ``block_diff_mask`` for the new backend.
        3. The HF ``forward()`` does not accept ``output_attentions`` and other
           kwargs the bridge unconditionally injects; patch at runtime because
           no other hook point allows filtering them before HF's forward call.
        """
        # Patch vocab_embed to have a weight attribute for EmbeddingBridge
        if hasattr(hf_model, "backbone") and hasattr(hf_model.backbone, "vocab_embed"):
            embed_mod = hf_model.backbone.vocab_embed
            if not hasattr(embed_mod, "weight") and hasattr(embed_mod, "embedding"):
                embed_mod.weight = embed_mod.embedding

        # Fix attention backend for CPU and ensure mask is on a real device
        if hasattr(hf_model, "backbone"):
            backend = getattr(self.cfg, "attn_backend", "sdpa")
            if not torch.cuda.is_available() and backend == "flex":
                backend = "sdpa"
                setattr(self.cfg, "attn_backend", "sdpa")
                for b in hf_model.backbone.blocks:
                    b.attn_backend = "sdpa"
            if hasattr(hf_model.backbone, "gen_mask"):
                hf_model.backbone.gen_mask(
                    getattr(self.cfg, "model_length", getattr(self.cfg, "n_ctx", 2048)),
                    getattr(hf_model.backbone, "block_size", 4),
                    attn_backend=backend,
                )

        # Patch the model's forward method to filter out unsupported kwargs (like output_attentions)
        original_forward = hf_model.forward
        import inspect

        def patched_forward(*args, **kwargs):
            sig = inspect.signature(original_forward)
            valid_params = set(sig.parameters.keys())

            # Check if VAR_KEYWORD (like **kwargs) is accepted
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            if not accepts_var_keyword:
                kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return original_forward(*args, **kwargs)

        hf_model.forward = patched_forward

    def convert_weights(self) -> dict[str, torch.Tensor]:
        """Return empty dict — delegation means no weight rearrangement."""
        return {}
