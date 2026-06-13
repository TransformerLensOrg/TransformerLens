"""Gemma4 architecture adapter."""


from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    ArithmeticTensorConversion,
    RearrangeTensorConversion,
    TransposeTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.arithmetic_tensor_conversion import (
    OperationTypes,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)


class Gemma4ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma4 models.

    Optional Parameters (may not exist in state_dict):
    ------------------------------------------------
    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP input
    - blocks.{i}.mlp.b_gate - No bias on MLP gate projection
    - blocks.{i}.mlp.b_out - No bias on MLP output
    - blocks.{i}.ln1.b - RMSNorm has no bias
    - blocks.{i}.ln2.b - RMSNorm has no bias
    - blocks.{i}.attn.k_proj.weight - Absent on KV-sharing layers
    - blocks.{i}.attn.v_proj.weight - Absent on KV-sharing layers
    - blocks.{i}.attn.k_norm.weight - Absent on KV-sharing layers
    - blocks.{i}.attn.v_proj.weight - Absent when attention_k_eq_v=True (full attention layers)
    - blocks.{i}.experts.* - Absent on dense layers (only present when enable_moe_block=True)
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma4 architecture adapter."""
        super().__init__(cfg)

        # Detect model type to set correct HF module paths
        # benchmark uses AutoModelForCausalLM which returns:
        #   Gemma4ForConditionalGeneration:  model.language_model.embed_tokens, model.language_model.layers
        #   Gemma4ForCausalLM (text-only):   model.embed_tokens, model.layers
        # Check cfg.architecture (set by boot) or cfg.architectures (HF config list)
        arch = getattr(cfg, "architecture", None) or ""
        if "Gemma4ForConditionalGeneration" in arch:
            self.text_prefix = "model.language_model"
            self.cfg.is_multimodal = True
            # Extract vision config for Phase 7 multimodal testing
            if hasattr(cfg, "vision_config"):
                vcfg = cfg.vision_config
                self.cfg.vision_hidden_size = getattr(vcfg, "hidden_size", None)
                self.cfg.vision_num_layers = getattr(vcfg, "num_hidden_layers", None)
                self.cfg.vision_num_heads = getattr(vcfg, "num_attention_heads", None)
                self.cfg.mm_tokens_per_image = getattr(cfg, "vision_soft_tokens_per_image", 256)
        else:
            self.text_prefix = "model"
        self._dot = f"{self.text_prefix}."

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        self.cfg.final_rms = True
        # Gemma models use (1.0 + weight) in RMSNorm instead of just weight
        # See: https://github.com/huggingface/transformers/pull/29402
        self.cfg.rmsnorm_uses_offset = True

        # Gemma4 uses rotary positional embeddings
        self.cfg.positional_embedding_type = "rotary"

        # Use eager attention to support output_attentions for hook_attn_scores and hook_pattern
        # SDPA doesn't support output_attentions, which is required for HookedTransformer compatibility
        self.cfg.attn_implementation = "eager"
        setattr(self.cfg, "use_native_generate", True)

        # Unwrap text config for multimodal models
        # Gemma4ForConditionalGeneration nests text settings in text_config
        # Gemma4ForCausalLM has them flat on the root config
        text_cfg = getattr(cfg, "text_config", cfg)

        # Gemma4 uses logit softcapping and attention softcapping
        if (
            hasattr(text_cfg, "final_logit_softcapping")
            and text_cfg.final_logit_softcapping is not None
        ):
            self.cfg.output_logits_soft_cap = text_cfg.final_logit_softcapping
        if (
            hasattr(text_cfg, "attn_logit_softcapping")
            and text_cfg.attn_logit_softcapping is not None
        ):
            self.cfg.attn_scores_soft_cap = text_cfg.attn_logit_softcapping

        # Gemma4 E-series has Per-Layer Embeddings (PLE)
        if (
            hasattr(text_cfg, "hidden_size_per_layer_input")
            and text_cfg.hidden_size_per_layer_input > 0
        ):
            setattr(self.cfg, "hidden_size_per_layer_input", text_cfg.hidden_size_per_layer_input)

        # Gemma4 E-series has KV sharing (later layers reuse KV from earlier layers)
        if hasattr(text_cfg, "num_kv_shared_layers") and text_cfg.num_kv_shared_layers > 0:
            setattr(self.cfg, "num_kv_shared_layers", text_cfg.num_kv_shared_layers)

        # Gemma4 has mixed attention: sliding window alternates with full attention
        if hasattr(text_cfg, "layer_types"):
            setattr(self.cfg, "layer_types", text_cfg.layer_types)

        # Gemma4 26B-A4B has MoE: dense MLP + parallel router/experts path
        self.enable_moe_block = getattr(text_cfg, "enable_moe_block", False)

        self.weight_processing_conversions = {
            # Note: Gemma4 uses Gemma4TextScaledWordEmbedding which scales
            # embeddings by sqrt(d_model) INSIDE its forward(). We do NOT
            # scale the stored weights here.
            #
            # Q/K/V weight conversions
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(
                        self.cfg,
                        "n_key_value_heads",
                        self.cfg.n_heads,
                    ),
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(
                        self.cfg,
                        "n_key_value_heads",
                        self.cfg.n_heads,
                    ),
                ),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
            # RMSNorm weight conversions - Gemma adds 1.0 to weights before applying
            # See: https://github.com/huggingface/transformers/pull/29402
            "blocks.{i}.ln1.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.ln1_post.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.ln2.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.ln2_post.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "ln_final.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            # Gemma4 has q_norm, k_norm, and v_norm (per-head RMSNorm) in attention
            "blocks.{i}.attn.q_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.attn.k_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            # MLP weight conversions - transpose from [out, in] to [in, out]
            "blocks.{i}.mlp.gate.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "blocks.{i}.mlp.in.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "blocks.{i}.mlp.out.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            # Unembed weight conversion - transpose from [vocab, d_model] to [d_model, vocab]
            "unembed.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            # Note: Gemma-4 does NOT have biases on attention projections (q/k/v/o_proj.bias are all None)
            # No bias conversions needed
        }

        # Set up component mapping with actual bridge instances
        # Build attention submodules - k/v projections/norms are optional on KV-sharing layers
        _k_norm = RMSNormalizationBridge(name="k_norm", config=self.cfg)
        _k_norm.optional = True
        _v_norm = RMSNormalizationBridge(name="v_norm", config=self.cfg)
        _v_norm.optional = True
        attn_submodules: dict[str, Any] = {
            "q": LinearBridge(name="q_proj"),
            "k": LinearBridge(name="k_proj", optional=True),
            "v": LinearBridge(name="v_proj", optional=True),
            "o": LinearBridge(name="o_proj"),
            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
            "k_norm": _k_norm,
            "v_norm": _v_norm,
        }

        block_submodules: dict[str, Any] = {
            "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
            "ln1_post": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
            "ln2": RMSNormalizationBridge(name="pre_feedforward_layernorm", config=self.cfg),
            "ln2_post": RMSNormalizationBridge(name="post_feedforward_layernorm", config=self.cfg),
            "attn": PositionEmbeddingsAttentionBridge(
                name="self_attn",
                config=self.cfg,
                submodules=attn_submodules,
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
        }

        # MoE: 26B-A4B has a parallel router/experts path after the standard MLP
        if self.enable_moe_block:
            _ln2_post_moe_1 = RMSNormalizationBridge(
                name="post_feedforward_layernorm_1", config=self.cfg
            )
            _ln2_post_moe_1.optional = True
            _ln2_pre_moe_2 = RMSNormalizationBridge(
                name="pre_feedforward_layernorm_2", config=self.cfg
            )
            _ln2_pre_moe_2.optional = True
            _ln2_post_moe_2 = RMSNormalizationBridge(
                name="post_feedforward_layernorm_2", config=self.cfg
            )
            _ln2_post_moe_2.optional = True
            _router = LinearBridge(name="router", config=self.cfg, optional=True)
            _experts = MoEBridge(name="experts", config=self.cfg)
            _experts.optional = True

            block_submodules["ln2_post_moe_1"] = _ln2_post_moe_1
            block_submodules["ln2_pre_moe_2"] = _ln2_pre_moe_2
            block_submodules["ln2_post_moe_2"] = _ln2_post_moe_2
            block_submodules["router"] = _router
            block_submodules["experts"] = _experts

        self.component_mapping: dict[str, Any] = {
            "embed": EmbeddingBridge(name=f"{self._dot}embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name=f"{self._dot}rotary_emb"),
            "blocks": BlockBridge(name=f"{self._dot}layers", submodules=block_submodules),
            "ln_final": RMSNormalizationBridge(name=f"{self._dot}norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

        # For multimodal models, prepend vision components
        if self.cfg.is_multimodal:
            self.component_mapping = {
                "vision_encoder": GeneralizedComponent(name="model.vision_tower"),
                "vision_projector": VisionProjectionBridge(name="model.embed_vision"),
                **self.component_mapping,
            }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Load bridge HF model with sdpa so hf_generate() avoids eager KV-cache bug."""
        if getattr(self.cfg, "use_native_generate", False):
            model_kwargs["attn_implementation"] = "sdpa"

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Setup hook compatibility for Gemma4 models.

        Gemma4 uses Gemma4TextScaledWordEmbedding which scales embeddings
        by sqrt(d_model) INSIDE the embedding layer's forward().
        Therefore we do NOT need a hook_conversion — the embed.hook_out already
        captures the scaled output. Adding a conversion would double-scale.
        """
        pass

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references and native autograd for Gemma4 component testing.

        Gemma4 uses per-layer-type RoPE with different frequencies for sliding vs
        full attention layers. We set the sliding attention RoPE on all bridge
        instances as the default for component testing.

        We also enable use_native_layernorm_autograd on all normalization bridges
        (q_norm, k_norm, v_norm) to delegate to HuggingFace's exact implementation.

        Additionally, we force the HF model to use "eager" attention to match the bridge's
        implementation. The bridge uses "eager" to support output_attentions for hooks, while
        HF defaults to "sdpa". These produce mathematically equivalent results.

        Args:
            hf_model: The HuggingFace Gemma4 model instance
            bridge_model: The TransformerBridge model (if available)
        """
        rotary_emb = self.get_remote_component(hf_model, f"{self._dot}rotary_emb")

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        # Get the layers module using the same prefix
        text_model = self.get_remote_component(hf_model, self.text_prefix)
        if hasattr(text_model, "layers"):
            for layer in text_model.layers:  # type: ignore[union-attr]
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

                    if hasattr(block.attn, "original_component"):
                        hf_attn = block.attn.original_component
                        if hasattr(hf_attn, "q_norm"):
                            hf_attn.q_norm.use_native_layernorm_autograd = True
                        if hasattr(hf_attn, "k_norm"):
                            hf_attn.k_norm.use_native_layernorm_autograd = True
                        if hasattr(hf_attn, "v_norm"):
                            hf_attn.v_norm.use_native_layernorm_autograd = True

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
