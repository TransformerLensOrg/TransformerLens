"""Gemma1 architecture adapter."""

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


class Gemma1ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma1 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma1 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        # Gemma models use BOS tokens (tokenizer prepends BOS by default)
        # Matches HookedTransformer behavior (default_prepend_bos = True)
        self.cfg.default_prepend_bos = True
        # Gemma models use (1.0 + weight) in RMSNorm instead of just weight
        # See: https://github.com/huggingface/transformers/pull/29402
        self.cfg.rmsnorm_uses_offset = True

        self.weight_processing_conversions = {
            # NOTE: Gemma1 scales embeddings by sqrt(d_model) at RUNTIME inside
            # GemmaTextScaledWordEmbedding.forward() (HF transformers >= 5.0).
            # That layer is what bridge.embed wraps, so embed.hook_out already
            # captures the scaled value — matching HookedTransformer's hook_embed
            # (which uses pre-scaled W_E). We must NOT pre-scale weights here and
            # we must NOT install a runtime hook_conversion that re-scales.
            #
            # Attention weight conversions
            **self._qkvo_weight_conversions(),
            # RMSNorm weight conversions - Gemma adds 1.0 to weights before applying
            # See: https://github.com/huggingface/transformers/pull/29402
            "blocks.{i}.ln1.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.ln2.weight": ParamProcessingConversion(
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
            # Unembed weight conversion - transpose from [vocab, d_model] to [d_model, vocab]
            "unembed.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
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
            "unembed": UnembeddingBridge(name="lm_head"),
        }
