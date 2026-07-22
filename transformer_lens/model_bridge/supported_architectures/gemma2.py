"""Gemma2 architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    ArithmeticTensorConversion,
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
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Gemma2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma2 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        # Gemma models were not trained with BOS tokens
        # self.cfg.default_prepend_bos = False
        # Gemma models use (1.0 + weight) in RMSNorm instead of just weight
        # See: https://github.com/huggingface/transformers/pull/29402
        self.cfg.rmsnorm_uses_offset = True

        # Gemma2 uses logit softcapping
        if hasattr(self.cfg, "final_logit_softcapping"):
            self.cfg.output_logits_soft_cap = self.cfg.final_logit_softcapping
        if hasattr(self.cfg, "attn_logit_softcapping"):
            self.cfg.attn_scores_soft_cap = self.cfg.attn_logit_softcapping

        # Note: n_key_value_heads is now automatically mapped from num_key_value_heads
        # by map_default_transformer_lens_config() in sources/transformers.py

        self.weight_processing_conversions = {
            # NOTE: Gemma2 scales embeddings by sqrt(d_model) at RUNTIME inside
            # Gemma2TextScaledWordEmbedding.forward() (HF transformers >= 5.0).
            # That layer is what bridge.embed wraps, so embed.hook_out already
            # captures the scaled value — matching HookedTransformer's hook_embed
            # (which uses pre-scaled W_E). We must NOT pre-scale weights here and
            # we must NOT install a runtime hook_conversion that re-scales.
            **self._qkvo_weight_conversions(),
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
            # # Unembed weight conversion - transpose from [vocab, d_model] to [d_model, vocab]
            "unembed.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    # Gemma 2 uses RMSNorm for all normalization layers
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
                    # Gemma 2 uses PositionEmbeddingsAttentionBridge like Gemma 3
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
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model)
