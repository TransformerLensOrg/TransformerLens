"""Jamba hybrid attention+Mamba architecture adapter.

Supports ``JambaForCausalLM`` (e.g. ``ai21labs/Jamba-tiny-random``,
``ai21labs/AI21-Jamba-Reasoning-3B``, ``ai21labs/Jamba-v0.1``).

Architecture overview:
- Heterogeneous layers from ``config.layers_block_type`` — each element is
  either ``"mamba"`` (``JambaMambaDecoderLayer``) or ``"attention"``
  (``JambaAttentionDecoderLayer``). Attention layers recur every
  ``attn_layer_period`` starting at ``attn_layer_offset`` (classically 1/8).
- Every layer has the same residual skeleton: pre-norm (``input_layernorm``) →
  mixer (``.mamba`` *or* ``.self_attn``) → residual → pre-FFN norm
  (``pre_ff_layernorm``) → ``.feed_forward`` (dense SwiGLU ``JambaMLP`` or
  ``JambaSparseMoeBlock`` when ``layers_num_experts[i] > 1``) → residual.
- The Mamba mixer is **Mamba-1** (``JambaMambaMixer``): ``in_proj`` / ``conv1d``
  / ``x_proj`` / ``dt_proj`` / ``out_proj``, plus Jamba-specific
  ``dt_layernorm`` / ``b_layernorm`` / ``c_layernorm`` on the selective params.
- Attention is GQA **without RoPE** — absolute-position-free; no model-level
  rotary module.
- Generation threads a unified ``DynamicCache`` via ``past_key_values``
  (attention KV + Mamba conv/recurrent states). The Mamba mixer receives that
  same object as ``cache_params=past_key_values``.

Key adapter decisions:
- ``BlockBridge`` (not ``SSMBlockBridge``): two norms and a real post-mixer
  residual make transformer-shaped hooks (``hook_resid_mid`` via ``ln2``)
  meaningful — same choice as Falcon-H1 / GraniteMoeHybrid.
- ``SSMMixerBridge`` (PR #1481 Mamba-1 interp surface) wraps ``.mamba`` under
  the canonical ``.mixer`` dict key so ``find_ssm_mixer`` /
  ``compute_effective_attention`` / ``eager_scan`` resolve it. Inner
  projections match Mamba-1; the three selective-param RMSNorms are mapped
  so reconstruction and the opt-in ``eager_scan`` path apply them (Jamba
  fork of stock Mamba-1). Default forward still delegates to HF
  (bit-identical).
- Attention and mixer are ``optional=True`` so setup skips the absent branch
  per layer type.
- FFN: dense ``GatedMLPBridge`` when ``num_experts <= 1`` (Reasoning-3B);
  ``MoEBridge`` passthrough with optional dense projections *and* router when
  ``num_experts > 1`` (tiny-random / 52B), since dense and MoE layers share
  the HF path ``.feed_forward``.
- ``is_stateful = False``: generation uses the standard ``past_key_values``
  path (same rationale as Zamba2 / Falcon-H1). Setting ``is_stateful`` would
  route through the pure-Mamba ``cache_params`` loop and diverge.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    SSMMixerBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


def _make_optional(component: GeneralizedComponent) -> GeneralizedComponent:
    """Mark a GeneralizedComponent submodule as optional.

    Some bridge classes (e.g. ``RMSNormalizationBridge``) do not forward
    ``optional`` through their own ``__init__``. ``component_setup.py`` reads
    ``getattr(submodule, "optional", False)`` at setup time.
    """
    component.optional = True
    return component


class JambaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ``JambaForCausalLM``.

    Interleaved attention + Mamba-1 layers with optional sparse MoE FFN.
    Attention and Mamba streams are separate optional slots so each can be
    ablated independently.
    """

    # P1: exact passthrough vs raw HF; P2/P3 skip HT comparison; P4 generation.
    applicable_phases: list[int] = [1, 2, 3, 4]

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        # No RoPE — JambaAttention is position-embedding-free.
        self.cfg.positional_embedding_type = "none"
        self.cfg.final_rms = True
        # Dense FFN is SwiGLU (gate_proj / up_proj / down_proj, silu).
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        # Standard past_key_values DynamicCache path (not pure-Mamba cache_params).
        self.cfg.is_stateful = False

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        # Tokenizer prepends BOS on __call__ for released AI21 checkpoints.
        self.cfg.default_prepend_bos = True

        # Per-layer mixer type list for analysis tools (same name as NemotronH /
        # GraniteMoeHybrid / Zamba2). Prefer the HF property; fall back to empty.
        layers_block_type = list(
            getattr(cfg, "layers_block_type", None) or getattr(cfg, "layer_types", None) or []
        )
        # HF layer_types uses linear_attention/full_attention; normalize to TL names.
        _LAYER_TYPE_TO_TL = {
            "linear_attention": "mamba",
            "full_attention": "attention",
            "mamba": "mamba",
            "attention": "attention",
        }
        setattr(
            self.cfg,
            "layers_block_type",
            [_LAYER_TYPE_TO_TL.get(t, t) for t in layers_block_type],
        )

        # Mamba-1 dimensional config (mirrors MambaArchitectureAdapter fields
        # that SSMMixerBridge.compute_* reads from the wrapped HF mixer; also
        # surfaced on cfg for tooling).
        mamba_expand = int(getattr(cfg, "mamba_expand", 2))
        mamba_d_state = int(getattr(cfg, "mamba_d_state", 16))
        mamba_d_conv = int(getattr(cfg, "mamba_d_conv", 4))
        intermediate_size = mamba_expand * int(self.cfg.d_model)
        setattr(self.cfg, "mamba_expand", mamba_expand)
        setattr(self.cfg, "mamba_d_state", mamba_d_state)
        setattr(self.cfg, "mamba_d_conv", mamba_d_conv)
        setattr(self.cfg, "mamba_dt_rank", getattr(cfg, "mamba_dt_rank", None))
        setattr(self.cfg, "intermediate_size", intermediate_size)
        setattr(self.cfg, "state_size", mamba_d_state)
        setattr(self.cfg, "conv_kernel", mamba_d_conv)
        setattr(self.cfg, "expand", mamba_expand)

        num_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", 1) or 1
        setattr(self.cfg, "num_experts", num_experts)

        # Heterogeneous attn/Mamba layers + native HF attention layout: LN folding
        # expects rearranged QKV and uniform per-layer attn, so disable (same as
        # GraniteMoeHybrid / other SSM hybrids).
        self.supports_fold_ln = False
        self.weight_processing_conversions = {}
        self.component_mapping = self._build_component_mapping(num_experts=int(num_experts))

    def _build_mamba_bridge(self) -> SSMMixerBridge:
        """Mamba-1 mixer under canonical ``.mixer``; HF path is ``.mamba``."""
        return SSMMixerBridge(
            name="mamba",
            config=self.cfg,
            optional=True,
            submodules={
                "in_proj": LinearBridge(name="in_proj"),
                "conv1d": DepthwiseConv1DBridge(name="conv1d"),
                "x_proj": LinearBridge(name="x_proj"),
                "dt_proj": LinearBridge(name="dt_proj"),
                "out_proj": LinearBridge(name="out_proj"),
                # Jamba-only selective-param norms (absent on stock Mamba-1).
                "dt_layernorm": _make_optional(
                    RMSNormalizationBridge(name="dt_layernorm", config=self.cfg)
                ),
                "b_layernorm": _make_optional(
                    RMSNormalizationBridge(name="b_layernorm", config=self.cfg)
                ),
                "c_layernorm": _make_optional(
                    RMSNormalizationBridge(name="c_layernorm", config=self.cfg)
                ),
            },
        )

    def _build_attention_bridge(self) -> AttentionBridge:
        """GQA attention without RoPE; keep HF's forward for cache parity."""
        return AttentionBridge(
            name="self_attn",
            config=self.cfg,
            optional=True,
            maintain_native_attention=True,
            requires_attention_mask=True,
            requires_position_embeddings=False,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
            },
        )

    def _build_ffn_bridge(self, num_experts: int) -> GatedMLPBridge | MoEBridge:
        """Dense SwiGLU or mixed dense/MoE sharing HF path ``.feed_forward``."""
        if num_experts > 1:
            # Mixed layers: MoEBridge passthrough handles both tensor and
            # (hidden, router_scores) returns. Optional children skip the
            # wrong feed_forward type per layer.
            return MoEBridge(
                name="feed_forward",
                config=self.cfg,
                submodules={
                    "gate": LinearBridge(name="gate_proj", optional=True),
                    "in": LinearBridge(name="up_proj", optional=True),
                    "out": LinearBridge(name="down_proj", optional=True),
                    "router": LinearBridge(name="router", optional=True),
                },
            )
        return GatedMLPBridge(
            name="feed_forward",
            config=self.cfg,
            submodules={
                "gate": LinearBridge(name="gate_proj"),
                "in": LinearBridge(name="up_proj"),
                "out": LinearBridge(name="down_proj"),
            },
        )

    def _build_component_mapping(self, num_experts: int) -> dict:
        return {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="pre_ff_layernorm", config=self.cfg),
                    "attn": self._build_attention_bridge(),
                    "mixer": self._build_mamba_bridge(),
                    "mlp": self._build_ffn_bridge(num_experts),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.final_layernorm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
