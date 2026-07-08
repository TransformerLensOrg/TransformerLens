"""RecurrentGemma (Griffin) architecture adapter.

Supports ``RecurrentGemmaForCausalLM`` (e.g. ``google/recurrentgemma-2b``,
``google/recurrentgemma-9b``), the open instance of the **Griffin** architecture.

Architecture overview:
- Heterogeneous layers defined by ``config.block_types`` (default pattern
  ``("recurrent", "recurrent", "attention")`` repeated over ``num_hidden_layers``).
  Each ``model.layers.{i}`` is a ``RecurrentGemmaDecoderLayer`` whose
  ``temporal_block`` is *either*:
    * ``RecurrentGemmaRecurrentBlock`` — the RG-LRU real-gated linear recurrence
      (``linear_x`` / ``linear_y`` / ``conv_1d`` / ``rg_lru`` / ``linear_out``), or
    * ``RecurrentGemmaSdpaAttention`` — local sliding-window GQA attention with
      partial rotary (``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` / ``rotary_emb``).
- Every layer additionally has ``temporal_pre_norm`` (pre-norm before the temporal
  block), ``channel_pre_norm`` (pre-norm before the MLP), and a gated MLP
  ``mlp_block`` (``gate_proj`` / ``up_proj`` / ``down_proj``, GELU-tanh).
- ``model.final_norm`` + ``lm_head``.

Gemma-family numerics (shared with Gemma-2/3): RMSNorm applies ``(1.0 + weight)``
(``rmsnorm_uses_offset``), token embeddings are scaled by ``sqrt(hidden_size)`` at
runtime inside the HF forward, and the final logits are tanh-soft-capped at
``config.logits_soft_cap`` (30.0).

Key adapter decision (mirrors ``Lfm2MoeArchitectureAdapter``): because the
``temporal_block`` substructure varies per layer (recurrent vs. attention), we
wrap each decoder layer as a whole with residual-stream hooks only, rather than
pretending every layer has a homogeneous attention/MLP substructure. This keeps
execution correct on both layer types. Finer-grained RG-LRU state hooks
(``hook_ssm_write`` / ``hook_ssm_state`` style) are a natural follow-up.

``applicable_phases = [4]``: the whole-layer bridge exposes only residual hooks,
so the component-level comparisons in phases 1-3 do not apply; phase 4
(generation + text quality) does.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)


class RecurrentGemmaBlockBridge(BlockBridge):
    """Whole-layer RecurrentGemma bridge exposing only residual-stream hooks.

    RecurrentGemma interleaves RG-LRU recurrent layers and local-attention layers.
    Wrapping the HF decoder layer as a whole preserves correct execution while
    avoiding unresolved standard attention/MLP aliases on the recurrent layers
    (which have no q/k/v/o or gate/up/down substructure).
    """

    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_post": "hook_out",
    }


class RecurrentGemmaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ``RecurrentGemmaForCausalLM`` (Griffin).

    Hybrid RG-LRU recurrence + local sliding-window attention. The temporal-block
    type per layer is determined by ``config.block_types[layer_idx % len(block_types)]``.
    """

    # Whole-layer residual hooks only; phases 1-3 compare component substructure
    # this adapter intentionally does not expose. Phase 4 (generation) applies.
    applicable_phases: list[int] = [4]

    def __init__(self, cfg: Any) -> None:
        """Initialize the RecurrentGemma architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # RG-LRU + local attention both use RoPE-style handling internally on the
        # attention layers; there is no model-level rotary module (it lives inside
        # each attention temporal_block), so we do not wire a rotary_emb component.
        # Gemma-family gated MLP (gate_proj -> GELU-tanh(up) -> down).
        # Gemma RMSNorm uses (1.0 + weight); see
        # https://github.com/huggingface/transformers/pull/29402
        self.cfg.rmsnorm_uses_offset = True

        # Gemma models were not trained with BOS tokens prepended.
        self.cfg.default_prepend_bos = False

        norm_eps = getattr(cfg, "rms_norm_eps", None)
        if norm_eps is not None:
            self.cfg.eps = norm_eps

        # Final logits are tanh-soft-capped at config.logits_soft_cap (30.0).
        logits_soft_cap = getattr(cfg, "logits_soft_cap", None)
        if logits_soft_cap is not None:
            self.cfg.output_logits_soft_cap = logits_soft_cap

        # Expose the per-layer temporal-block pattern for analysis tools, using the
        # canonical `layers_block_type` name as on Nemotron-H / Granite.
        block_types = list(getattr(cfg, "block_types", None) or [])
        setattr(self.cfg, "block_types", block_types)
        setattr(self.cfg, "layers_block_type", block_types)

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": RecurrentGemmaBlockBridge(name="model.layers", config=self.cfg),
            "ln_final": RMSNormalizationBridge(name="model.final_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Force eager attention when the HF config exposes the implementation knob."""
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

    def prepare_model(self, hf_model: Any) -> None:
        """Force eager attention on the loaded HF model when supported."""
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"
