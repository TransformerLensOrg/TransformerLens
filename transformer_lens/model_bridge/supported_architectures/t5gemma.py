"""T5Gemma architecture adapter.

T5GemmaForConditionalGeneration is an encoder-decoder model combining:
- Gemma-style RoPE, GQA, gated MLP, and RMSNorm with offset (+1.0)
- Encoder-decoder cross-attention in the decoder stack
- Nested config: encoder/decoder dims live in cfg.encoder / cfg.decoder

Key differences from plain T5:
- Uses model.encoder.layers / model.decoder.layers (not .block)
- No relative position bias; uses RoPE instead
- All norms are Gemma-style (weight + 1.0)
- lm_head is T5GemmaLMHead wrapping out_proj (no .weight at the top level)
"""

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
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.t5gemma_decoder_block import (
    T5GemmaDecoderBlockBridge,
)


class T5GemmaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for T5GemmaForConditionalGeneration.

    Encoder: BlockBridge over model.encoder.layers (Gemma-style, no cross-attn)
    Decoder: T5GemmaDecoderBlockBridge over model.decoder.layers (adds cross-attn hooks)
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.supports_fold_ln = False

        # Config flags used by bridge weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        # Gemma-family GELU; the nested enc/dec config defeats the auto-mapper,
        # which would otherwise leave act_fn at the "relu" default.
        self.cfg.act_fn = "gelu_pytorch_tanh"
        self.cfg.uses_rms_norm = True
        # T5Gemma uses Gemma-style (1.0 + weight) RMSNorm offset
        self.cfg.rmsnorm_uses_offset = True

        n_heads = self.cfg.n_heads
        n_kv = getattr(self.cfg, "n_key_value_heads", None) or n_heads

        self.weight_processing_conversions = {
            # Encoder self-attention
            "encoder_blocks.{i}.self_attn.q_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_heads),
            ),
            "encoder_blocks.{i}.self_attn.k_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv),
            ),
            "encoder_blocks.{i}.self_attn.v_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv),
            ),
            "encoder_blocks.{i}.self_attn.o_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=n_heads),
            ),
            # Encoder RMSNorm offset - HF stores raw weight; Gemma applies weight+1
            "encoder_blocks.{i}.pre_self_attn_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "encoder_blocks.{i}.post_self_attn_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "encoder_blocks.{i}.pre_feedforward_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "encoder_blocks.{i}.post_feedforward_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            # Encoder MLP (gated)
            "encoder_blocks.{i}.mlp.gate_proj.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "encoder_blocks.{i}.mlp.up_proj.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "encoder_blocks.{i}.mlp.down_proj.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            # Decoder self-attention
            "decoder_blocks.{i}.self_attn.q_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_heads),
            ),
            "decoder_blocks.{i}.self_attn.k_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv),
            ),
            "decoder_blocks.{i}.self_attn.v_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv),
            ),
            "decoder_blocks.{i}.self_attn.o_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=n_heads),
            ),
            # Decoder cross-attention
            "decoder_blocks.{i}.cross_attn.q_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_heads),
            ),
            "decoder_blocks.{i}.cross_attn.k_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv),
            ),
            "decoder_blocks.{i}.cross_attn.v_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv),
            ),
            "decoder_blocks.{i}.cross_attn.o_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=n_heads),
            ),
            # Decoder RMSNorm offset
            "decoder_blocks.{i}.pre_self_attn_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "decoder_blocks.{i}.post_self_attn_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "decoder_blocks.{i}.pre_cross_attn_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "decoder_blocks.{i}.post_cross_attn_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "decoder_blocks.{i}.pre_feedforward_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "decoder_blocks.{i}.post_feedforward_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            # Decoder MLP (gated)
            "decoder_blocks.{i}.mlp.gate_proj.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "decoder_blocks.{i}.mlp.up_proj.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "decoder_blocks.{i}.mlp.down_proj.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            # Final layer norms
            "encoder_ln_final.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "decoder_ln_final.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            # Unembed
            "unembed.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
        }

        self.component_mapping = {
            # Encoder embedding and positional
            "encoder_embed": EmbeddingBridge(name="model.encoder.embed_tokens"),
            "encoder_rotary_emb": RotaryEmbeddingBridge(name="model.encoder.rotary_emb"),
            # Encoder layers - Gemma-style BlockBridge (pre/post norms, RoPE attention, gated MLP)
            "encoder_blocks": BlockBridge(
                name="model.encoder.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="pre_self_attn_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(
                        name="post_self_attn_layernorm", config=self.cfg
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
                        is_causal=False,  # T5Gemma encoder is bidirectional
                    ),
                    "ln2": RMSNormalizationBridge(
                        name="pre_feedforward_layernorm", config=self.cfg
                    ),
                    "ln2_post": RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg
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
            # Encoder final norm
            "encoder_ln_final": RMSNormalizationBridge(name="model.encoder.norm", config=self.cfg),
            # Decoder embedding and positional
            "decoder_embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "decoder_rotary_emb": RotaryEmbeddingBridge(name="model.decoder.rotary_emb"),
            # Decoder layers — T5GemmaDecoderBlockBridge (adds cross-attn + two mid hooks)
            "decoder_blocks": T5GemmaDecoderBlockBridge(
                name="model.decoder.layers",
                config=self.cfg,
                submodules={
                    # Self-attention norms
                    "ln1": RMSNormalizationBridge(name="pre_self_attn_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(
                        name="post_self_attn_layernorm", config=self.cfg
                    ),
                    "self_attn": PositionEmbeddingsAttentionBridge(
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
                    # Cross-attention norms
                    "ln2": RMSNormalizationBridge(name="pre_cross_attn_layernorm", config=self.cfg),
                    "ln2_post": RMSNormalizationBridge(
                        name="post_cross_attn_layernorm", config=self.cfg
                    ),
                    "cross_attn": AttentionBridge(
                        name="cross_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        is_cross_attention=True,
                    ),
                    # MLP norms
                    "ln3": RMSNormalizationBridge(
                        name="pre_feedforward_layernorm", config=self.cfg
                    ),
                    "ln3_post": RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg
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
            # Decoder final norm
            "decoder_ln_final": RMSNormalizationBridge(name="model.decoder.norm", config=self.cfg),
            # lm_head is T5GemmaLMHead; the weight lives on its inner out_proj Linear
            "unembed": UnembeddingBridge(name="lm_head.out_proj", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for T5Gemma component testing.

        Both the encoder and decoder carry their own rotary_emb. We set the
        reference on all PositionEmbeddingsAttentionBridge instances so that
        component-level forward calls can compute RoPE correctly.
        """
        encoder_rotary = hf_model.model.encoder.rotary_emb
        decoder_rotary = hf_model.model.decoder.rotary_emb

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if bridge_model is not None:
            for block in getattr(bridge_model, "encoder_blocks", []):
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(encoder_rotary)
            for block in getattr(bridge_model, "decoder_blocks", []):
                if hasattr(block, "self_attn"):
                    block.self_attn.set_rotary_emb(decoder_rotary)

        enc_attn = self.get_generalized_component("encoder_blocks.0.attn")
        enc_attn.set_rotary_emb(encoder_rotary)
        dec_self_attn = self.get_generalized_component("decoder_blocks.0.self_attn")
        dec_self_attn.set_rotary_emb(decoder_rotary)
