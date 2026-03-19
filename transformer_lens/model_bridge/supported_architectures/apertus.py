"""Apertus architecture adapter."""

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
    LinearBridge,
    MLPBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

logger = logging.getLogger(__name__)


class ApertusArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Apertus models.

    Apertus uses a pre-norm architecture with RMSNorm, Q/K normalization in attention,
    rotary position embeddings (RoPE with LLaMA-3 scaling), grouped query attention (GQA),
    non-gated MLP (XiELU activation), and no biases on any projections.

    Similar to Qwen3 (pre-norm RMSNorm, QK-norm, GQA, RoPE) but uses a non-gated MLP
    (up_proj -> XiELU -> down_proj) instead of gated MLP.

    Note: Apertus uses different layer norm names than most Llama-family models:
    - attention_layernorm (instead of input_layernorm)
    - feedforward_layernorm (instead of post_attention_layernorm)
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Apertus architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True

        # Use eager attention to support output_attentions for hook_attn_scores and hook_pattern
        # SDPA doesn't support output_attentions, which is required for HookedTransformer compatibility
        self.cfg.attn_implementation = "eager"

        self.weight_processing_conversions = {
            # Q/K/V weight conversions - handle GQA (Grouped Query Attention)
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(self.cfg, "n_key_value_heads", None) or self.cfg.n_heads,
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(self.cfg, "n_key_value_heads", None) or self.cfg.n_heads,
                ),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
        }

        # Set up component mapping
        # Apertus uses attention_layernorm / feedforward_layernorm instead of the
        # typical input_layernorm / post_attention_layernorm names.
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="attention_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(
                        name="feedforward_layernorm", config=self.cfg
                    ),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch XIELUActivation to handle meta tensor initialization.

        Transformers v5 uses meta tensors during from_pretrained, but
        XIELUActivation.__init__ calls .item() on buffer tensors which fails
        on meta device. We defer the scalar computation to forward() time.
        """
        try:
            from transformers.activations import XIELUActivation
        except ImportError:
            return

        if getattr(XIELUActivation, "_apertus_patched", False):
            return

        import torch

        _orig_init = XIELUActivation.__init__

        def _patched_init(self, *args, **kwargs):
            """XIELUActivation init that defers .item() calls for meta tensor compat."""
            torch.nn.Module.__init__(self)
            alpha_p_init = kwargs.get("alpha_p_init", 0.8)
            alpha_n_init = kwargs.get("alpha_n_init", 0.8)
            beta = kwargs.get("beta", 0.5)
            eps = kwargs.get("eps", -1e-6)
            dtype = kwargs.get("dtype", torch.bfloat16)
            with_vector_loads = kwargs.get("with_vector_loads", False)

            self.alpha_p = torch.nn.Parameter(
                torch.log(torch.expm1(torch.tensor(alpha_p_init, dtype=dtype))).unsqueeze(0)
            )
            self.alpha_n = torch.nn.Parameter(
                torch.log(torch.expm1(torch.tensor(alpha_n_init - beta, dtype=dtype))).unsqueeze(0)
            )
            self.register_buffer("beta", torch.tensor(beta, dtype=dtype))
            self.register_buffer("eps", torch.tensor(eps, dtype=dtype))
            self.with_vector_loads = with_vector_loads
            # Defer scalar computation — will be computed lazily in forward()
            self._beta_scalar = None
            self._eps_scalar = None
            self._xielu_cuda_obj = None

        _orig_forward = XIELUActivation.forward

        def _patched_forward(self, x):
            """Forward that lazily computes scalars on first call."""
            if self._beta_scalar is None:
                self._beta_scalar = float(self.beta.detach().cpu().float().item())
                self._eps_scalar = float(self.eps.detach().cpu().float().item())
            return _orig_forward(self, x)

        XIELUActivation.__init__ = _patched_init
        XIELUActivation.forward = _patched_forward
        XIELUActivation._apertus_patched = True
        logger.debug("Patched XIELUActivation for meta tensor compatibility")

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Apertus component testing.

        Apertus uses RoPE (Rotary Position Embeddings). We set the rotary_emb on
        all attention bridge instances for component testing.

        We also force the HF model to use "eager" attention to match the bridge's
        implementation. The bridge uses "eager" to support output_attentions for hooks.

        Args:
            hf_model: The HuggingFace Apertus model instance
            bridge_model: The TransformerBridge model (if available, set rotary_emb on actual instances)
        """
        # Get rotary embedding instance from the model
        rotary_emb = hf_model.model.rotary_emb

        # Force HF model to use "eager" attention to match bridge implementation
        # Bridge uses "eager" to support output_attentions for hook compatibility
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        # Also set on all attention layers
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Set rotary_emb on actual bridge instances in bridge_model if available
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            # Set on each layer's actual attention bridge instance
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
