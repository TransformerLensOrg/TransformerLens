"""Gemma 3n text-only architecture adapter.

Bridges the text path of the full tri-modal ``Gemma3nForConditionalGeneration``
(``model.language_model`` + ``lm_head``); the vision/audio towers stay referenced but
unbridged (see the vision+audio follow-up). The decoder layers run on a stacked AltUp
4-stream residual, so blocks use ``AltUpBlockBridge`` rather than ``BlockBridge``. All
math is deferred to HF; submodules are decomposed only for hooks (parity-safe delegation).
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AltUpBlockBridge,
    AttentionBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class _SparsityPreservingGatedMLPBridge(GatedMLPBridge):
    """GatedMLPBridge that always delegates, even under processed weights.

    Gemma3nTextMLP applies `_gaussian_topk` activation sparsity (0.95 on the
    first layers of E2B/E4B) before the activation; the functional
    processed-weights branch computes plain act(gate)*up and inflates those
    layers' output ~10x. Delegation stays numerically correct in compat mode
    because set_processed_weights has already written the processed tensors
    into the wrapped Linears — and the gate/in/out hooks still fire from
    inside HF's forward, which is all the bridge swap wanted.
    """

    def set_processed_weights(self, weights, verbose: bool = False) -> None:
        super().set_processed_weights(weights, verbose=verbose)
        self._use_processed_weights = False


class Gemma3nArchitectureAdapter(ArchitectureAdapter):
    """Text-only adapter for Gemma 3n (`Gemma3nForConditionalGeneration`)."""

    _testing_lm_attr = "model.language_model"
    _testing_wire_rotary = False

    # The full model includes a timm-based vision tower (TimmWrapperModel), so timm is needed
    # even for text-only use (the towers stay referenced).
    required_libraries: list[str] = ["timm"]
    required_libraries_group: str = "multimodal"

    # Phase 3 (processed/compatibility mode) folds LN into a single residual stream, which
    # AltUp's 4-stream residual can't represent. Phases 1 (HF parity), 2 (hooks), and 4 (text
    # quality) do apply and pass.
    applicable_phases: list[int] = [1, 2, 4]

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.is_multimodal = False
        self.cfg.gated_mlp = True
        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        self.cfg.rmsnorm_uses_offset = True  # Gemma RMSNorm uses (1.0 + weight)
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.attn_implementation = "eager"
        # AltUp + per-layer-embedding residual topology isn't fold-safe.
        self.supports_fold_ln = False
        self.weight_processing_conversions: dict = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.language_model.rotary_emb"),
            "blocks": AltUpBlockBridge(
                name="model.language_model.layers",
                config=self.cfg,
                submodules={
                    "input_layernorm": GeneralizedComponent(name="input_layernorm"),
                    "post_attention_layernorm": GeneralizedComponent(
                        name="post_attention_layernorm"
                    ),
                    "pre_feedforward_layernorm": GeneralizedComponent(
                        name="pre_feedforward_layernorm"
                    ),
                    "post_feedforward_layernorm": GeneralizedComponent(
                        name="post_feedforward_layernorm"
                    ),
                    "post_per_layer_input_norm": GeneralizedComponent(
                        name="post_per_layer_input_norm"
                    ),
                    "altup": GeneralizedComponent(name="altup"),
                    "laurel": GeneralizedComponent(name="laurel"),
                    "per_layer_input_gate": GeneralizedComponent(name="per_layer_input_gate"),
                    "per_layer_projection": GeneralizedComponent(name="per_layer_projection"),
                    # AttentionBridge for the per-head hook surface a bare
                    # component lacks. Delegated semantics: pattern AND scores
                    # both fire HF's post-softmax weights (no pre-softmax tensor
                    # exists, unlike gemma1/2/3); writes to them are inert.
                    "self_attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        maintain_native_attention=True,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            # The last num_kv_shared_layers layers reuse earlier KV and
                            # drop their own k/v projections and norms.
                            "k": LinearBridge(name="k_proj", optional=True),
                            "v": LinearBridge(name="v_proj", optional=True),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": GeneralizedComponent(name="q_norm"),
                            "k_norm": GeneralizedComponent(name="k_norm", optional=True),
                            "v_norm": GeneralizedComponent(name="v_norm", optional=True),
                        },
                    ),
                    "mlp": _SparsityPreservingGatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": GeneralizedComponent(name="model.language_model.norm"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
