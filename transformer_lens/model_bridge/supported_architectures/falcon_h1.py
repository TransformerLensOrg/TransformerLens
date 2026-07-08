"""Falcon-H1 parallel-hybrid architecture adapter.

Supports ``FalconH1ForCausalLM`` (e.g. ``tiiuae/Falcon-H1-0.5B-Base``).

Architecture overview:
- Every block runs **GQA attention and a Mamba-2 mixer in parallel** on the same
  ``input_layernorm`` output; their results are summed into the residual stream
  alongside scalar multipliers. A second norm (``pre_ff_layernorm``) precedes a
  SwiGLU feed-forward MLP. This is the parallel hybrid that distinguishes
  Falcon-H1 from heterogeneous-layer hybrids like NemotronH (one mixer type per
  layer) — here both branches are present in *every* block.
- RoPE via a model-level ``model.rotary_emb`` module (very large ``rope_theta``).
- ``mamba_rms_norm=false`` on the released checkpoints — the Mamba inner gated
  RMSNorm is **absent**, so ``inner_norm`` is declared optional.
- ~12 scalar multipliers (``embedding_multiplier``, ``attention_out_multiplier``,
  ``key_multiplier``, ``ssm_*_multiplier``, ``mlp_multipliers``,
  ``lm_head_multiplier`` …). HF applies all of these in its own forward, so in
  raw (passthrough) mode the bridge inherits them for free — no weight folding is
  needed for forward parity. Folding would only matter for compatibility mode.

Key adapter decisions:
- ``BlockBridge`` is the block container: the block has two norms
  (``input_layernorm`` + ``pre_ff_layernorm``) and a real post-attention
  residual, so transformer-shaped hooks (``hook_resid_mid`` via ``ln2``) are
  meaningful — unlike single-norm SSM blocks that need ``SSMBlockBridge``.
  ``BlockBridge.forward`` delegates the whole block to HF, so the parallel
  attn+mamba combine happens natively.
- ``SSM2MixerBridge`` wraps the Mamba-2 mixer (passthrough forward). Its inner
  ``in_proj`` / ``conv1d`` / ``out_proj`` are mapped for hookability; the gated
  ``inner_norm`` is optional (absent when ``mamba_rms_norm=false``).
- ``applicable_phases = []``: ``verify_models`` is transformer-shaped and does
  not meaningfully cover SSM hybrids. Correctness is gated by the integration
  parity test instead (the standard set for Mamba/Mamba2/NemotronH).
- Generation uses the **standard transformer KV-cache path**, not the bridge's
  stateful-Mamba loop. Falcon-H1's HF forward carries both attention KV and the
  Mamba conv/recurrent state inside one unified ``past_key_values`` cache, so the
  default path matches HF bit-for-bit. Setting ``is_stateful`` would instead
  route generation through the pure-Mamba ``cache_params`` convention, which this
  hybrid does not use, and diverges from HF after the first decode step.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    GatedRMSNormBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SSM2MixerBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


def _make_optional(component: GeneralizedComponent) -> GeneralizedComponent:
    """Mark a GeneralizedComponent submodule as optional.

    Some bridge classes (e.g. ``GatedRMSNormBridge``) do not forward ``optional``
    through their own ``__init__`` even though ``GeneralizedComponent`` supports
    it. Setting the attribute directly is safe because ``component_setup.py``
    reads ``getattr(submodule, "optional", False)`` at setup time.
    """
    component.optional = True
    return component


class FalconH1ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ``FalconH1ForCausalLM``.

    Parallel hybrid: every block runs GQA attention and a Mamba-2 mixer side by
    side, then a SwiGLU MLP. Both branches are mapped on every block so each
    sub-path is independently hookable for ablation studies.
    """

    # verify_models is transformer-shaped and would need a dedicated refactor to
    # cover SSM hybrids. Forward-pass correctness lives in the integration test:
    # tests/integration/model_bridge/test_falcon_h1_adapter.py
    applicable_phases: list[int] = []

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # SwiGLU feed-forward (gate_proj / up_proj / down_proj, silu activation).
        # FalconH1RMSNorm stores its epsilon as `variance_epsilon` (like Llama).
        setattr(self.cfg, "eps_attr", "variance_epsilon")

        # Mamba-2 dimensional config. Falcon-H1 specifies the inner SSM width
        # directly via `mamba_d_ssm` (= mamba_n_heads * mamba_d_head), unlike
        # Mamba2's `expand`-derived intermediate size. conv_dim follows the
        # Mamba-2 layout: inner width + 2 groups of state for B and C.
        mamba_d_ssm = getattr(cfg, "mamba_d_ssm", self.cfg.d_model)
        mamba_n_groups = getattr(cfg, "mamba_n_groups", 1)
        mamba_d_state = getattr(cfg, "mamba_d_state", 128)
        mamba_n_heads = getattr(cfg, "mamba_n_heads", 0)
        conv_dim = mamba_d_ssm + 2 * mamba_n_groups * mamba_d_state
        setattr(self.cfg, "mamba_intermediate_size", mamba_d_ssm)
        setattr(self.cfg, "conv_dim", conv_dim)
        # HF fuses gate, hidden_BC, and dt into one in_proj output; stored so a
        # future HF layout change is caught by the integration test.
        setattr(
            self.cfg, "expected_in_proj_out_features", 2 * mamba_d_ssm + conv_dim + mamba_n_heads
        )

        # Raw (passthrough) mode delegates the full forward to HF, which applies
        # all of Falcon-H1's scalar multipliers natively — no weight folding for
        # forward parity.
        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="pre_ff_layernorm", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mamba": SSM2MixerBridge(
                        name="mamba",
                        config=self.cfg,
                        submodules={
                            "in_proj": LinearBridge(name="in_proj"),
                            "conv1d": DepthwiseConv1DBridge(name="conv1d"),
                            # HF names the inner gated norm "norm"; TL aliases it
                            # to "inner_norm" to disambiguate from the block norm.
                            # Absent when mamba_rms_norm=false, so optional.
                            "inner_norm": _make_optional(GatedRMSNormBridge(name="norm")),
                            "out_proj": LinearBridge(name="out_proj"),
                        },
                    ),
                    "mlp": GatedMLPBridge(
                        name="feed_forward",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.final_layernorm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire the model-level rotary embedding onto the attention bridges."""
        if not hasattr(hf_model.model, "rotary_emb"):
            return

        rotary_emb = hf_model.model.rotary_emb

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        try:
            attn_bridge = self.get_generalized_component("blocks.0.attn")
            attn_bridge.set_rotary_emb(rotary_emb)
        except (AttributeError, KeyError, ValueError):
            pass
