"""Gemma 4 architecture adapter.

Bridges the text path of ``Gemma4ForConditionalGeneration``
(``model.language_model`` + ``lm_head``) and the vision pipeline. For the standard
variants (E2B / E4B / 31B / 26B-A4B) the vision encoder (``model.vision_tower``) and
projector (``model.embed_vision``) are both bridged, enabling Phase 7 multimodal testing.

The same adapter also covers ``Gemma4UnifiedForConditionalGeneration`` (the
encoder-free 12B variant, transformers >= 5.10): its text decoder is a strict
structural subset — same module paths, no PLE and no MoE, both optional here.
It is still multimodal but has no ``vision_tower`` — ``model.embed_vision`` is the
full vision pipeline (raw-patch projection), mapped as the projector only.

Per-layer structure is heterogeneous across the family, so all math is deferred to HF
and submodules are decomposed only for hooks (parity-safe delegation):

- **KV sharing** (E2B/E4B): the last ``num_kv_shared_layers`` layers reuse earlier KV
  states and drop their own ``k_proj`` / ``v_proj`` / ``k_norm`` / ``v_norm``.
- **K==V attention** (31B / 26B-A4B): global-attention layers share key and value
  weights (``attention_k_eq_v``) and have no ``v_proj``.
- **Per-Layer Embeddings** (E2B/E4B): each layer mixes in a per-layer input via
  ``per_layer_input_gate`` / ``per_layer_projection`` / ``post_per_layer_input_norm``.
- **MoE** (26B-A4B): layers add a ``router`` + batched ``experts`` block in parallel
  with the dense MLP, sandwiched by three extra norms.

Unlike Gemma 1-3, ``Gemma4RMSNorm`` multiplies by ``weight`` directly — there is no
``(1.0 + weight)`` offset.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    DelegatedAttentionBlockBridge,
    EmbeddingBridge,
    LinearBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class Gemma4ArchitectureAdapter(ArchitectureAdapter):
    """Adapter for Gemma 4 (`Gemma4ForConditionalGeneration` — multimodal, or
    `Gemma4UnifiedForConditionalGeneration` — text-only 12B)."""

    # Phase 3 (processed/compatibility mode) folds LN into a single residual stream,
    # which the PLE residual mix, per-layer `layer_scalar` buffers, and the MoE branch
    # can't represent. Phases 1 (HF parity), 2 (hooks), and 4 (text quality) apply.
    applicable_phases: list[int] = [1, 2, 4]

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Both variants are multimodal (take pixel_values). The difference:
        # - Gemma4ForConditionalGeneration: vision_tower (encoder) + embed_vision (projector)
        # - Gemma4UnifiedForConditionalGeneration (12B): embed_vision only — encoder-free
        #   embedder that does raw-patch projection without an attention-based vision encoder.
        arch = getattr(cfg, "architecture", "") or ""
        self._is_unified = "Gemma4Unified" in arch
        self.cfg.is_multimodal = True

        if hasattr(cfg, "vision_config"):
            vcfg = cfg.vision_config
            self.cfg.vision_hidden_size = getattr(vcfg, "hidden_size", None)
            self.cfg.vision_num_layers = getattr(vcfg, "num_hidden_layers", None)
            self.cfg.vision_num_heads = getattr(vcfg, "num_attention_heads", None)
            self.cfg.mm_tokens_per_image = getattr(cfg, "vision_soft_tokens_per_image", 256)

        self.cfg.gated_mlp = True
        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        # Gemma4RMSNorm scales by weight directly — no (1 + weight) offset, unlike Gemma 1-3.
        self.cfg.rmsnorm_uses_offset = False
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.attn_implementation = "eager"
        # PLE / layer_scalar / MoE residual topology isn't fold-safe.
        self.supports_fold_ln = False
        self.weight_processing_conversions: dict = {}

        # Vision components. Gemma4ForConditionalGeneration has a separate vision
        # encoder (model.vision_tower) + projector (model.embed_vision). The 12B
        # unified variant is encoder-free — model.embed_vision is the full vision
        # pipeline (raw-patch projection), so it maps as the projector with no encoder.
        _vision_mapping: dict[str, Any] = {
            "vision_projector": VisionProjectionBridge(name="model.embed_vision"),
        }
        if not self._is_unified:
            _vision_mapping = {
                "vision_encoder": GeneralizedComponent(name="model.vision_tower"),
                **_vision_mapping,
            }

        self.component_mapping = {
            **_vision_mapping,
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            # Single rotary module serving both layer types (full / sliding) via a
            # per-layer-type forward kwarg, with separate rope parameters per type.
            "rotary_emb": RotaryEmbeddingBridge(name="model.language_model.rotary_emb"),
            "blocks": DelegatedAttentionBlockBridge(
                name="model.language_model.layers",
                submodules={
                    # Sandwich norms: ln1/ln1_post around attention, ln2/ln2_post
                    # around the MLP (same shape as Gemma 2/3).
                    "ln1": GeneralizedComponent(name="input_layernorm"),
                    "ln1_post": GeneralizedComponent(name="post_attention_layernorm"),
                    "ln2": GeneralizedComponent(name="pre_feedforward_layernorm"),
                    "ln2_post": GeneralizedComponent(name="post_feedforward_layernorm"),
                    # PLE residual mix — present only when hidden_size_per_layer_input > 0
                    # (E2B/E4B; absent on 31B and 26B-A4B).
                    "per_layer_input_gate": GeneralizedComponent(
                        name="per_layer_input_gate", optional=True
                    ),
                    "per_layer_projection": GeneralizedComponent(
                        name="per_layer_projection", optional=True
                    ),
                    "post_per_layer_input_norm": GeneralizedComponent(
                        name="post_per_layer_input_norm", optional=True
                    ),
                    # MoE branch — present only when enable_moe_block (26B-A4B).
                    "router": GeneralizedComponent(name="router", optional=True),
                    "experts": GeneralizedComponent(name="experts", optional=True),
                    "pre_feedforward_layernorm_2": GeneralizedComponent(
                        name="pre_feedforward_layernorm_2", optional=True
                    ),
                    "post_feedforward_layernorm_1": GeneralizedComponent(
                        name="post_feedforward_layernorm_1", optional=True
                    ),
                    "post_feedforward_layernorm_2": GeneralizedComponent(
                        name="post_feedforward_layernorm_2", optional=True
                    ),
                    "attn": GeneralizedComponent(
                        name="self_attn",
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            # KV-shared layers (E2B/E4B) drop k/v projections and norms;
                            # K==V layers (31B / 26B-A4B global attention) drop v_proj.
                            "k": LinearBridge(name="k_proj", optional=True),
                            "v": LinearBridge(name="v_proj", optional=True),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": GeneralizedComponent(name="q_norm"),
                            "k_norm": GeneralizedComponent(name="k_norm", optional=True),
                            "v_norm": GeneralizedComponent(name="v_norm", optional=True),
                        },
                    ),
                    "mlp": GeneralizedComponent(
                        name="mlp",
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

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention so bridge and HF match (sliding/full layer mix)."""
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"
        language_model = getattr(getattr(hf_model, "model", None), "language_model", None)
        if language_model is not None and hasattr(language_model, "layers"):
            for layer in language_model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"
