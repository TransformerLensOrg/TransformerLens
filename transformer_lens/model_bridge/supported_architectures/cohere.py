"""Cohere architecture adapter.

Supports CohereForCausalLM models (Command-R family) with:
- Parallel attention+MLP sharing a single input_layernorm (no post_attention_layernorm)
- True LayerNorm (CohereLayerNorm) with weight but no bias
- GQA (grouped-query attention) with separate Q/K/V/O projections
- Gated SwiGLU MLP (gate_proj, up_proj, down_proj)
- Logit scaling: output logits multiplied by config.logit_scale (default 1/16)
- Tied embed/unembed weights by default (tie_word_embeddings=True)
- Interleaved RoPE via CohereRotaryEmbedding (delegated to HF module)
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class CohereArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Cohere models (CohereForCausalLM).

    Architectural quirks vs. standard decoder-only models:
    - Single input_layernorm per block; NO post_attention_layernorm.
      Attention and MLP both read the SAME normed hidden states (parallel).
    - CohereLayerNorm is true LayerNorm (mean-subtracting), NOT RMSNorm.
      It has a weight parameter but NO bias parameter.
    - Logit scale: CohereForCausalLM.forward multiplies logits by logit_scale
      (default 0.0625 = 1/16). Folded into unembed.weight via preprocess_weights.
    - Rotary embeddings use repeat_interleave instead of cat-split (delegated to HF).

    Optional parameters (absent from state_dict by default):
    - blocks.{i}.attn.b_Q/b_K/b_V/b_O — no bias on projections (attention_bias=False)
    - blocks.{i}.mlp.b_gate/b_in/b_out  — no bias on MLP projections
    - blocks.{i}.ln1.b                   — CohereLayerNorm has no bias
    - ln_final.b                         — CohereLayerNorm has no bias
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Cohere architecture adapter."""
        super().__init__(cfg)

        # --- Normalization ---
        # CohereLayerNorm is true LayerNorm (subtracts mean), NOT RMSNorm.
        # uses_rms_norm=False tells NormalizationBridge to subtract the mean.
        # eps_attr="variance_epsilon": CohereLayerNorm stores eps as self.variance_epsilon.
        self.cfg.normalization_type = "LN"
        self.cfg.uses_rms_norm = False
        self.cfg.eps_attr = "variance_epsilon"
        self.cfg.final_rms = False

        # --- Position embeddings and MLP ---
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False

        # --- Parallel block: single norm, no post_attention_layernorm ---
        self.cfg.parallel_attn_mlp = True

        # --- Tokenizer: BOS is prepended by default ---
        # CohereTokenizerFast has add_bos_token=False but HF's __call__ with
        # add_special_tokens=True (the default) prepends BOS. Verified against
        # trl-internal-testing/tiny-CohereForCausalLM.
        self.cfg.default_prepend_bos = True

        # --- GQA: n_key_value_heads ---
        # sources/transformers.py copies num_key_value_heads generically.
        # Re-read here to ensure it's set on cfg for _qkvo_weight_conversions.
        n_kv = getattr(cfg, "n_key_value_heads", None)
        if n_kv is not None:
            self.cfg.n_key_value_heads = n_kv

        # --- Weight processing conversions ---
        # Standard GQA-aware Q/K/V/O rearrangements (same as Llama/Qwen2).
        # n_kv is already set on self.cfg; _qkvo_weight_conversions reads it via
        # getattr(self.cfg, "n_key_value_heads", None) when called with no args.
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        # --- Logit scale ---
        # CohereConfig.logit_scale is typed float | None; apply explicit None-check
        # so cfg.logit_scale is always a plain float (never None).
        # logit_scale is not a declared field on TransformerBridgeConfig; it is a
        # Cohere-specific dynamic attribute accessed later in preprocess_weights.
        _ls = getattr(cfg, "logit_scale", None)
        self.cfg.logit_scale = float(_ls) if _ls is not None else 0.0625  # type: ignore[attr-defined]

        # --- RoPE theta (informational metadata) ---
        # CohereRotaryEmbedding reads config.rope_parameters["rope_theta"] directly;
        # store it in cfg.rotary_base so TL config accurately reflects the model.
        # TransformerBridgeConfig stores rotary_base as int, matching its declared type.
        _rope_params = getattr(cfg, "rope_parameters", None) or {}
        if isinstance(_rope_params, dict):
            _theta = _rope_params.get("rope_theta", getattr(cfg, "default_theta", 10000.0))
        else:
            _theta = getattr(cfg, "default_theta", 10000.0)
        self.cfg.rotary_base = int(_theta)

        # --- Component mapping ---
        # Block structure follows Falcon's parallel_attn=True, num_ln_in_parallel_attn=1
        # mode: single ln1 feeds both attn and MLP; NO ln2.
        # Submodule shapes follow Llama: separate q/k/v/o projections and SwiGLU MLP.
        # Rotary and attention both delegate to HF modules, preserving Cohere's
        # repeat_interleave RoPE convention without re-implementing it in TL.
        self.component_mapping = {
            # Embedding: model.embed_tokens (same root as Llama, not transformer.* like Falcon)
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            # Rotary embedding: top-level, delegates to CohereRotaryEmbedding.
            # Pattern matches llama.py:75 and falcon.py:154 — NOT inside blocks.
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    # Single pre-norm only — Cohere has no post_attention_layernorm.
                    # NormalizationBridge handles weight-only CohereLayerNorm correctly:
                    # it checks `hasattr(original_component, "bias") and bias is not None`
                    # before adding bias, so the missing bias attribute is silently skipped.
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    # No "ln2" — parallel block, same normed input goes to attn AND mlp.
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
                    # GatedMLPBridge: gate/in/out matches Llama's gate_proj/up_proj/down_proj.
                    # Optional use_qk_norm is handled transparently by HF's
                    # CohereAttention.forward delegation (no extra submodules needed).
                    "mlp": GatedMLPBridge(
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
            # Final LayerNorm (CohereLayerNorm, weight-only) at model.norm
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            # Unembed: lm_head. logit_scale is folded into weight in preprocess_weights.
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fold logit_scale into unembed weights before ProcessWeights runs.

        bridge.py lines 726-732 clone unembed.weight before calling this, so
        scaling does not affect the tied embed.weight.
        logit_scale=1.0 is a no-op (skipped for efficiency).
        """
        scale: float = getattr(self.cfg, "logit_scale")  # always set by __init__
        if scale != 1.0:
            for key in ("unembed.weight", "unembed.bias"):
                if key in state_dict:
                    orig_dtype = state_dict[key].dtype
                    state_dict[key] = (state_dict[key].float() * scale).to(orig_dtype)
        return state_dict

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set rotary embedding reference on attention bridges for component testing.

        CohereRotaryEmbedding lives at hf_model.model.rotary_emb. The bridge
        delegates to it directly, preserving the repeat_interleave RoPE convention
        without re-implementing it in TL.

        Pattern matches llama.py and qwen2.py.
        """
        rotary_emb = hf_model.model.rotary_emb

        # Set on actual bridge instances in the live model (if available)
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template so get_generalized_component() calls work
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
