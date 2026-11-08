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
``config.logits_soft_cap`` (30.0). The embedding scale is HF's ``model.normalizer``
buffer, which is stored in bfloat16 (``sqrt(2560)`` becomes 50.5); ``hook_embed``
carries the scaled embedding that enters the residual stream, as on Gemma-2/3.

Key adapter decision: every layer is ``temporal_pre_norm`` -> ``temporal_block``
-> residual add -> ``channel_pre_norm`` -> ``mlp_block`` -> residual add, so the
norms, the temporal block and the MLP are bridged on every layer. Because the
``temporal_block`` varies per layer (RG-LRU vs. local attention), it is wrapped
as a passthrough ``mixer`` (``hook_mixer_in`` / ``hook_mixer_out``) rather than
an attention bridge; ``cfg.layers_block_type`` says which one a layer has.
Per-head attention hooks and RG-LRU state hooks (``hook_ssm_write`` /
``hook_ssm_state`` style) are a natural follow-up.

``applicable_phases = [4]``: the temporal blocks are not decomposed into
components, so the component-level comparisons in phases 1-3 do not apply;
phase 4 (generation + text quality) does.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class RecurrentGemmaBlockBridge(BlockBridge):
    """RecurrentGemma layer: a temporal block (RG-LRU or local attention) and a gated MLP.

    The temporal block is bridged as a passthrough ``mixer``, so the layer has no
    attention aliases; it writes ``mixer.hook_out`` and ``mlp.hook_out`` to the
    residual stream.
    """

    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_mid": "ln2.hook_in",
        "hook_resid_post": "hook_out",
        "hook_mixer_in": "mixer.hook_in",
        "hook_mixer_out": "mixer.hook_out",
        "hook_mlp_out": "mlp.hook_out",
    }


def _embedding_normalizer(d_model: int) -> float:
    """HF's ``RecurrentGemmaModel.normalizer``: ``sqrt(hidden_size)`` stored in bfloat16."""
    return float(torch.tensor(d_model**0.5, dtype=torch.bfloat16))


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
        # Hookable attention needs eager; the base prepare hooks force it through
        # from_pretrained and onto the loaded config.
        self.cfg.attn_implementation = "eager"
        # RG-LRU + local attention both use RoPE-style handling internally on the
        # attention layers; there is no model-level rotary module (it lives inside
        # each attention temporal_block), so we do not wire a rotary_emb component.
        # Gemma-family gated MLP (gate_proj -> GELU-tanh(up) -> down).
        # Gemma RMSNorm uses (1.0 + weight); see
        # https://github.com/huggingface/transformers/pull/29402
        self.cfg.rmsnorm_uses_offset = True

        # Gemma models were not trained with BOS tokens prepended.
        self.cfg.default_prepend_bos = False

        # RecurrentGemmaMlp is intermediate_size // 2 wide (gate_proj / up_proj outputs).
        intermediate_size = getattr(cfg, "intermediate_size", None)
        if isinstance(intermediate_size, int):
            self.cfg.d_mlp = intermediate_size // 2

        norm_eps = getattr(cfg, "rms_norm_eps", None)
        if norm_eps is not None:
            self.cfg.eps = norm_eps

        # Final logits are tanh-soft-capped at config.logits_soft_cap (30.0).
        logits_soft_cap = getattr(cfg, "logits_soft_cap", None)
        if logits_soft_cap is not None:
            self.cfg.output_logits_soft_cap = logits_soft_cap

        # Canonical per-layer pattern (as on Nemotron-H). Read the builder's
        # expanded layers_block_type; the raw block_types pattern is not
        # propagated and would clobber it with [] on real boots.
        setattr(self.cfg, "layers_block_type", self._canonical_layer_types(cfg))

        self.component_mapping = {
            # HF multiplies the embedding by model.normalizer after embed_tokens.
            "embed": EmbeddingBridge(
                name="model.embed_tokens",
                output_contribution_scale=_embedding_normalizer(self.cfg.d_model),
            ),
            "blocks": RecurrentGemmaBlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="temporal_pre_norm", config=self.cfg),
                    # RG-LRU recurrent block or local attention, per cfg.layers_block_type.
                    "mixer": GeneralizedComponent(name="temporal_block"),
                    "ln2": RMSNormalizationBridge(name="channel_pre_norm", config=self.cfg),
                    "mlp": self._gated_mlp(name="mlp_block"),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.final_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_model(self, hf_model: Any) -> None:
        """Take the embedding scale from the loaded model's ``normalizer`` buffer."""
        super().prepare_model(hf_model)
        normalizer = getattr(getattr(hf_model, "model", None), "normalizer", None)
        embed = self.component_mapping["embed"] if self.component_mapping else None
        if isinstance(normalizer, torch.Tensor) and isinstance(embed, EmbeddingBridge):
            embed._output_contribution_scale = float(normalizer)
