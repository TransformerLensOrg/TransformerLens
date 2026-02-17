"""OpenELM architecture adapter."""

import sys
from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)


class OpenElmArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Apple OpenELM models.

    OpenELM uses a unique architecture with per-layer varying head counts and FFN
    dimensions. Key characteristics:

    - Combined QKV projection (qkv_proj) with per-layer varying Q/KV head counts
    - Gated MLP with combined gate+up projection (proj_1) and per-layer FFN sizes
    - RMSNorm normalization
    - Full rotary embeddings (per-layer, not shared)
    - Optional Q/K RMSNorm (normalize_qk_projections=True)
    - Weight tying (share_input_output_layers=True typically)
    - Model root is 'transformer' (not 'model')
    - Requires trust_remote_code=True (custom HF code)

    The native HF attention handles all per-layer dimension variations, RoPE,
    GQA group repeat, and Q/K normalization internally. The bridge delegates
    to the native forward for correct computation.

    Note: Individual Q/K/V hooks are not available since the model uses a combined
    QKV projection. Attention-level hooks (hook_attn_in, hook_attn_out) are provided.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the OpenELM architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True

        self.default_config = {
            "d_model": cfg.d_model,
            "d_head": getattr(cfg, "head_dim", cfg.d_model // cfg.n_heads),
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_vocab": cfg.d_vocab,
        }

        # GQA support
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.default_config["n_key_value_heads"] = cfg.n_key_value_heads
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        # OpenELM doesn't ship its own tokenizer — uses LLaMA tokenizer.
        # Use NousResearch mirror (ungated) to avoid access restrictions.
        self.cfg.tokenizer_name = "NousResearch/Llama-2-7b-hf"

        # No weight processing conversions needed - native attention handles all
        # per-layer dimension variations internally
        self.weight_processing_conversions = {}

        # Store reference for RoPE patching
        self._original_rope_compute = None
        self._rope_class = None

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.token_embeddings"),
            "blocks": BlockBridge(
                name="transformer.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="attn_norm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="ffn_norm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "qkv": LinearBridge(name="qkv_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                        maintain_native_attention=True,
                        requires_attention_mask=True,
                    ),
                    "mlp": MLPBridge(
                        name="ffn",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="proj_1"),
                            "out": LinearBridge(name="proj_2"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="transformer.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch OpenELM for compatibility with transformers v5.

        Two patches are needed:
        1. RotaryEmbedding: Custom _compute_sin_cos_embeddings fails on meta device
           because it calls .cos() on meta tensors. We wrap it to catch NotImplementedError.
        2. Weight re-initialization: OpenELM's _init_weights re-randomizes ALL weights
           after they've been loaded from safetensors because transformers v5's
           _finalize_load_state_dict calls initialize_weights() on modules lacking the
           _is_hf_initialized flag. We patch _init_weights to skip real (non-meta) tensors.

        Args:
            model_name: The HuggingFace model name/path
            model_kwargs: The kwargs dict for from_pretrained()
        """
        # Force-import the modeling module so we can patch it
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            get_class_from_dynamic_module(
                "modeling_openelm.OpenELMForCausalLM",
                model_name,
            )
        except Exception:
            return

        # Find ALL imported OpenELM modules and apply patches.
        # Each model variant (e.g., OpenELM-1_1B vs OpenELM-1_1B-Instruct) gets its own
        # module in sys.modules with a different cache path, so we patch all of them.
        for key in list(sys.modules.keys()):
            if "openelm" in key.lower() and "modeling" in key.lower():
                module = sys.modules[key]
                if hasattr(module, "OpenELMRotaryEmbedding"):
                    rope_class = module.OpenELMRotaryEmbedding
                    # Skip if already patched (avoid wrapping safe_compute in safe_compute)
                    if getattr(rope_class, "_tl_patched", False):
                        continue
                    # Patch 1: RoPE meta device fix
                    original_compute = rope_class._compute_sin_cos_embeddings

                    def safe_compute(
                        self,
                        key_len,
                        key_device="cpu",
                        key_dtype=torch.float32,
                        _original=original_compute,
                    ):
                        try:
                            _original(self, key_len, key_device, key_dtype)
                        except NotImplementedError:
                            pass  # Deferred: re-initialized in prepare_model()

                    rope_class._compute_sin_cos_embeddings = safe_compute
                    rope_class._tl_patched = True
                    self._original_rope_compute = original_compute
                    self._rope_class = rope_class

                if hasattr(module, "OpenELMPreTrainedModel"):
                    pretrained_class = module.OpenELMPreTrainedModel
                    if getattr(pretrained_class, "_tl_patched", False):
                        continue
                    # Patch 2: Prevent _init_weights from re-randomizing loaded weights.
                    # transformers v5 calls _init_weights on all modules after weight
                    # materialization. For modules with real (non-meta) tensors, we must
                    # skip re-initialization to preserve the loaded checkpoint values.
                    original_init_weights = pretrained_class._init_weights

                    def safe_init_weights(
                        self,
                        mod,
                        _original=original_init_weights,
                    ):
                        # Only initialize modules still on meta device (pre-loading)
                        first_param = next(mod.parameters(), None)
                        if first_param is not None and first_param.device.type != "meta":
                            return  # Already loaded from checkpoint — don't re-randomize
                        _original(self, mod)

                    pretrained_class._init_weights = safe_init_weights
                    pretrained_class._tl_patched = True

    def prepare_model(self, hf_model: Any) -> None:
        """Post-load fixes for non-persistent buffers zeroed during meta materialization.

        Transformers v5 creates models on meta device then materializes weights from
        checkpoint. Non-persistent buffers (registered with persistent=False) are NOT
        in the checkpoint, so they materialize as zeros. OpenELM has two critical
        non-persistent buffers that must be recomputed:

        1. RoPE inv_freq — zeroed inv_freq produces cos=1, sin=0 for all positions,
           destroying positional information entirely.
        2. causal_mask — zeroed mask means no causal masking, allowing all positions
           to attend to future tokens. Single forward passes appear correct (no future
           tokens to leak) but autoregressive generation degenerates immediately.

        We also create a synthetic lm_head for weight-tied models.

        Note: We intentionally do NOT restore the original _compute_sin_cos_embeddings.
        The safe_compute wrapper is functionally equivalent for real (non-meta) tensors,
        and keeping it avoids issues when multiple models are loaded in the same process
        (e.g., benchmark suite loading both HF reference and bridge models).

        Args:
            hf_model: The loaded HuggingFace OpenELM model
        """
        # Ensure use_cache is set on config (transformers v5 raises AttributeError
        # for missing config attributes, and OpenELM's custom config omits use_cache)
        if not hasattr(hf_model.config, "use_cache") or "use_cache" not in hf_model.config.__dict__:
            hf_model.config.use_cache = False

        # Fix 1: Always recompute causal_mask (non-persistent buffer).
        # After meta→real materialization, the buffer may contain garbage values
        # (not all zeros) depending on the materializer's memory state. The old
        # check `not cm.any()` only recomputed when all zeros, missing cases where
        # garbage values are non-zero. Always recompute to guarantee correctness.
        if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "causal_mask"):
            cm = hf_model.transformer.causal_mask
            if cm is not None:
                seq_len = cm.shape[-1]
                correct_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=cm.dtype, device=cm.device),
                    diagonal=1,
                )
                hf_model.transformer.causal_mask = correct_mask

        # Fix 2: Recompute RoPE inv_freq on all layers (non-persistent buffer zeroed
        # during materialization), then force-recompute sin/cos embeddings.
        if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "layers"):
            rope_max = getattr(hf_model.config, "rope_max_length", 4096)
            for layer in hf_model.transformer.layers:
                if hasattr(layer, "attn") and hasattr(layer.attn, "pos_embedding"):
                    rope = layer.attn.pos_embedding
                    # Always recompute inv_freq (non-persistent buffer).
                    # Like causal_mask, inv_freq may contain garbage after meta
                    # materialization rather than clean zeros.
                    correct_inv_freq = 1.0 / (
                        rope.freq_constant
                        ** (
                            torch.arange(0, rope.model_dim, 2, dtype=torch.float32)
                            / rope.model_dim
                        )
                    )
                    rope.inv_freq = correct_inv_freq.to(rope.inv_freq.device)
                    # Force-recompute sin/cos (may have been computed with zero inv_freq)
                    rope._cached_cos = None
                    rope._cached_sin = None
                    rope._compute_sin_cos_embeddings(rope_max)

        # Create synthetic lm_head when embeddings are shared
        if getattr(hf_model, "lm_head", None) is None and hasattr(hf_model, "transformer"):
            embed = hf_model.transformer.token_embeddings
            lm_head = torch.nn.Linear(embed.embedding_dim, embed.num_embeddings, bias=False)
            lm_head.weight = embed.weight
            hf_model.lm_head = lm_head

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up references for OpenELM component testing.

        Args:
            hf_model: The HuggingFace OpenELM model instance
            bridge_model: The TransformerBridge model (if available)
        """
        pass
