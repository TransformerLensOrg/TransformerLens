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
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
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
    - blocks.{i}.attn.v_norm.weight - Absent on KV-sharing layers
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma4 architecture adapter."""
        super().__init__(cfg)

        # Detect model type to set correct HF module paths
        # Gemma4ForCausalLM (text-only):   model.embed_tokens, model.layers
        # Gemma4ForConditionalGeneration:  model.language_model.embed_tokens, model.language_model.layers
        architectures = getattr(cfg, "architectures", [])
        if "Gemma4ForConditionalGeneration" in architectures:
            self.text_prefix = "model.language_model"
        else:
            self.text_prefix = "model"
        text = self.text_prefix

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        self.cfg.final_rms = True
        self.cfg.eps_attr = "rms_norm_eps"
        # Gemma models use (1.0 + weight) in RMSNorm instead of just weight
        # See: https://github.com/huggingface/transformers/pull/29402
        self.cfg.rmsnorm_uses_offset = True

        # Gemma4 uses rotary positional embeddings
        self.cfg.positional_embedding_type = "rotary"

        # Use eager attention to support output_attentions for hook_attn_scores and hook_pattern
        # SDPA doesn't support output_attentions, which is required for HookedTransformer compatibility
        self.cfg.attn_implementation = "eager"

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

        # MoE guard: 26B-A4B variant is not yet supported
        if getattr(text_cfg, "enable_moe_block", False):
            raise NotImplementedError(
                "MoE variants of Gemma 4 (e.g. 26B-A4B) are not yet supported by this adapter."
            )

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
            # Gemma4 adds v_norm (RMSNorm without scale parameter, i.e. with_scale=False)
            "blocks.{i}.attn.v_norm.weight": ParamProcessingConversion(
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
        self.component_mapping = {
            "embed": EmbeddingBridge(name=f"{text}.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name=f"{text}.rotary_emb"),
            "blocks": BlockBridge(
                name=f"{text}.layers",
                submodules={
                    # All Gemma-4 normalizations use simple RMSNorm pass-through
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(
                        name="post_attention_layernorm", config=self.cfg
                    ),
                    "ln2": RMSNormalizationBridge(
                        name="pre_feedforward_layernorm", config=self.cfg
                    ),
                    "ln2_post": RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg
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
                            "v_norm": RMSNormalizationBridge(name="v_norm", config=self.cfg),
                        },
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
            "ln_final": RMSNormalizationBridge(name=f"{text}.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

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
        text = self.text_prefix
        rotary_emb = self.get_remote_component(hf_model, f"{text}.rotary_emb")

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        # Get the layers module using the same prefix
        text_model = self.get_remote_component(hf_model, text)
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
