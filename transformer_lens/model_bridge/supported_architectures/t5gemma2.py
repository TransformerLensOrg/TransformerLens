"""T5Gemma2 architecture adapter (text-only).

T5Gemma2ForConditionalGeneration is a multimodal encoder-decoder model. This
adapter bridges the text path only:
- Encoder text stack under model.encoder.text_model (the SigLIP vision_tower and
  multi_modal_projector are intentionally left unmapped).
- Decoder stack under model.decoder.

Key differences from T5Gemma:
- Encoder text lives at model.encoder.text_model.* (not model.encoder.*).
- The decoder uses a single T5Gemma2MergedAttention that fuses self- and
  cross-attention with shared q/k/v/o projections; there is no separate
  cross-attention module and no cross-attention layernorms.
- Both encoder and decoder attention add Gemma-style QK-norm (q_norm/k_norm).
- Per-layer sliding/full attention with dual RoPE and per-head QK-norm are all
  handled natively by HF — the bridge only routes inputs and fires hooks.
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
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.t5gemma2_decoder_block import (
    T5Gemma2DecoderBlockBridge,
)
from transformer_lens.model_bridge.generalized_components.t5gemma2_merged_attention import (
    T5Gemma2MergedAttentionBridge,
)


class T5Gemma2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for T5Gemma2ForConditionalGeneration (text-only).

    Encoder: BlockBridge over model.encoder.text_model.layers (Gemma-style, QK-norm, no cross-attn)
    Decoder: T5Gemma2DecoderBlockBridge over model.decoder.layers (merged self+cross attention)
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
        # T5Gemma2 uses Gemma-style (1.0 + weight) RMSNorm offset
        self.cfg.rmsnorm_uses_offset = True

        # n_heads/n_kv are decoder-effective; the builder surfaces the encoder
        # text stack's own counts for unbalanced pairs.
        n_heads = self.cfg.n_heads
        n_kv = getattr(self.cfg, "n_key_value_heads", None) or n_heads
        enc_heads = getattr(self.cfg, "encoder_attention_heads", None) or n_heads
        enc_kv = getattr(self.cfg, "encoder_key_value_heads", None) or n_kv

        self.weight_processing_conversions = {
            # Encoder self-attention
            "encoder_blocks.{i}.self_attn.q_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=enc_heads),
            ),
            "encoder_blocks.{i}.self_attn.k_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=enc_kv),
            ),
            "encoder_blocks.{i}.self_attn.v_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=enc_kv),
            ),
            "encoder_blocks.{i}.self_attn.o_proj.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=enc_heads),
            ),
            # Encoder QK-norm (Gemma-style +1 offset)
            "encoder_blocks.{i}.self_attn.q_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "encoder_blocks.{i}.self_attn.k_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
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
            # Decoder merged attention (self + cross share these projections)
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
            # Decoder QK-norm (Gemma-style +1 offset)
            "decoder_blocks.{i}.self_attn.q_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "decoder_blocks.{i}.self_attn.k_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            # Decoder RMSNorm offset
            "decoder_blocks.{i}.pre_self_attn_layernorm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "decoder_blocks.{i}.post_self_attn_layernorm.weight": ParamProcessingConversion(
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
            # Encoder embedding and positional (text stack lives under text_model)
            "encoder_embed": EmbeddingBridge(name="model.encoder.text_model.embed_tokens"),
            "encoder_rotary_emb": RotaryEmbeddingBridge(name="model.encoder.text_model.rotary_emb"),
            # Encoder layers - Gemma-style BlockBridge (pre/post norms, QK-norm attention, gated MLP)
            "encoder_blocks": BlockBridge(
                name="model.encoder.text_model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="pre_self_attn_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(
                        name="post_self_attn_layernorm", config=self.cfg
                    ),
                    # Native delegation: the encoder uses per-layer sliding/full
                    # bidirectional windows carried by HF's per-layer mask, which the
                    # manual attention path does not apply (it drifts materially past
                    # the sliding_window length). Delegating keeps sliding correct.
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        # HF's T5Gemma2SelfAttention unpacks position_embeddings
                        # unconditionally, so component testing must supply it.
                        requires_position_embeddings=True,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
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
            "encoder_ln_final": RMSNormalizationBridge(
                name="model.encoder.text_model.norm", config=self.cfg
            ),
            # Decoder embedding and positional
            "decoder_embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "decoder_rotary_emb": RotaryEmbeddingBridge(name="model.decoder.rotary_emb"),
            # Decoder layers — T5Gemma2DecoderBlockBridge (merged self+cross attention)
            "decoder_blocks": T5Gemma2DecoderBlockBridge(
                name="model.decoder.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="pre_self_attn_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(
                        name="post_self_attn_layernorm", config=self.cfg
                    ),
                    # Delegates to the native T5Gemma2MergedAttention (self+cross with
                    # shared q/k/v/o); the merged/cross logic, QK-norm, RoPE, and scaling
                    # cannot be reimplemented by the manual attention path. Exposes the
                    # self pattern (hook_pattern) and cross pattern (hook_cross_pattern).
                    "self_attn": T5Gemma2MergedAttentionBridge(
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
            # Decoder final norm
            "decoder_ln_final": RMSNormalizationBridge(name="model.decoder.norm", config=self.cfg),
            # lm_head is T5Gemma2LMHead; the weight lives on its inner out_proj Linear
            "unembed": UnembeddingBridge(name="lm_head.out_proj", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for T5Gemma2 component testing.

        Both the encoder text stack and the decoder carry their own rotary_emb. We
        set the reference on all PositionEmbeddingsAttentionBridge instances so that
        component-level forward calls can compute RoPE correctly, force eager
        attention (so patterns are hookable), and enable native layernorm autograd
        on QK-norm so the manual encoder path matches HF exactly.
        """
        encoder_rotary = hf_model.model.encoder.text_model.rotary_emb

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        # QK-norm must delegate to HF's exact RMSNorm autograd to avoid manual drift.
        def _enable_qk_native_autograd(layers: Any) -> None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                if hasattr(attn, "q_norm"):
                    attn.q_norm.use_native_layernorm_autograd = True
                if hasattr(attn, "k_norm"):
                    attn.k_norm.use_native_layernorm_autograd = True

        _enable_qk_native_autograd(hf_model.model.encoder.text_model.layers)
        _enable_qk_native_autograd(hf_model.model.decoder.layers)

        if bridge_model is not None:
            for block in getattr(bridge_model, "encoder_blocks", []):
                if hasattr(block, "attn") and hasattr(block.attn, "set_rotary_emb"):
                    block.attn.set_rotary_emb(encoder_rotary)
            # Decoder self_attn delegates to native (which owns its RoPE), so it
            # has no set_rotary_emb; nothing to wire.

        enc_attn = self.get_generalized_component("encoder_blocks.0.attn")
        if hasattr(enc_attn, "set_rotary_emb"):
            enc_attn.set_rotary_emb(encoder_rotary)
