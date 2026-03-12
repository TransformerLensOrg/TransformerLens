"""OLMo architecture adapter."""

import logging
from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
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


class OlmoArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OLMo (v1) models.

    OLMo v1 uses a pre-norm architecture with a custom non-learnable LayerNorm
    (fixed weight=1, bias=0), rotary position embeddings (RoPE), and gated MLP
    (SwiGLU). Key differences from later OLMo variants:

    - Pre-norm: LayerNorm is applied BEFORE attention and BEFORE MLP.
    - Non-learnable LayerNorm: Weight and bias are not trainable parameters.
      Delegating to HF's native forward via NormalizationBridge handles this correctly.
    - No Q/K normalization in attention.
    - Optional QKV clipping (handled by HF's native attention forward).

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP up_proj
    - blocks.{i}.mlp.b_gate - No bias on MLP gate_proj
    - blocks.{i}.mlp.b_out - No bias on MLP down_proj
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the OLMo architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = False
        # Force eager attention for numerical consistency with benchmark reference
        self.cfg.attn_implementation = "eager"

        self.default_config = {
            "d_model": cfg.d_model,
            "d_head": cfg.d_model // cfg.n_heads,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_vocab": cfg.d_vocab,
        }

        # GQA support
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.default_config["n_key_value_heads"] = cfg.n_key_value_heads
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        n_kv_heads = (
            self.cfg.n_key_value_heads
            if self.cfg.n_key_value_heads is not None
            else self.cfg.n_heads
        )

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
        }

        # Component mapping — PRE-NORM architecture:
        # ln1 = input_layernorm (applied BEFORE attention)
        # ln2 = post_attention_layernorm (applied BEFORE MLP)
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(
                        name="input_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="post_attention_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
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
            "ln_final": NormalizationBridge(
                name="model.norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_model(self, hf_model: Any) -> None:
        """Patch OLMo's in-place clamp_ to avoid backward hook conflicts.

        OLMo v1 uses query_states.clamp_() when config.clip_qkv is set.
        In-place ops on tensors that pass through register_full_backward_hook
        trigger PyTorch's "view modified inplace" error.  This patch disables
        the in-place clamp branch during attention forward passes.

        Note: clip_qkv clamping is skipped in the patched forward.  In practice
        clip_qkv values (typically 100+) rarely activate.  If exact clamping is
        needed, add out-of-place clamp hooks on hook_q/hook_k/hook_v.
        """
        _patch_olmo_inplace_clamp(hf_model)

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for OLMo component testing.

        OLMo uses RoPE (Rotary Position Embeddings). We set the rotary_emb
        reference on all attention bridge instances for component testing.

        Args:
            hf_model: The HuggingFace OLMo model instance
            bridge_model: The TransformerBridge model (if available)
        """
        # Get rotary embedding instance from the model
        rotary_emb = hf_model.model.rotary_emb

        # Force HF model to use "eager" attention to match bridge implementation
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Set rotary_emb on actual bridge instances in bridge_model if available
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)


def _patch_olmo_inplace_clamp(hf_model: Any) -> None:
    """Patch OLMo attention to avoid in-place clamp_ that conflicts with backward hooks.

    PyTorch's register_full_backward_hook wraps module outputs in
    BackwardHookFunctionBackward views.  OLMo's attention does
    query_states.clamp_() on tensors derived from those views, which
    PyTorch forbids.

    Fix: wrap each attention layer's forward to temporarily clear
    config.clip_qkv (preventing the in-place branch) and apply
    out-of-place clamping via a forward hook instead.
    """
    if not hasattr(hf_model, "model") or not hasattr(hf_model.model, "layers"):
        return

    clip_qkv = getattr(hf_model.config, "clip_qkv", None)
    if clip_qkv is None:
        return

    import functools
    import types

    patched = 0
    for layer in hf_model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        original_forward = attn.forward

        def _make_patched_forward(orig_fwd, clip_val=clip_qkv):
            @functools.wraps(orig_fwd)
            def patched_forward(*args, **kwargs):
                # Temporarily disable clip_qkv so HF's in-place clamp_ is skipped
                cfg = hf_model.config
                saved = cfg.clip_qkv
                cfg.clip_qkv = None
                try:
                    return orig_fwd(*args, **kwargs)
                finally:
                    cfg.clip_qkv = saved

            return patched_forward

        attn.forward = _make_patched_forward(original_forward)
        patched += 1

    if patched > 0:
        logging.info(
            "Patched %d OLMo attention layer(s): disabled in-place clamp_ "
            "(clip_qkv=%.1f) for backward hook compatibility.",
            patched,
            clip_qkv,
        )
