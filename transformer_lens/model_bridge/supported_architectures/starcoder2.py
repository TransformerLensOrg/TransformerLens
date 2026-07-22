"""Starcoder2 architecture adapter."""

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
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Starcoder2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Starcoder2 models (``Starcoder2ForCausalLM``).

    Starcoder2 is a Llama-shaped decoder — sequential pre-norm blocks, GQA, and
    rotary position embeddings — but differs from Llama in three ways it shares
    with its GPTBigCode (StarCoder) predecessor:

    - **LayerNorm (with bias)** rather than RMSNorm (``normalization_type="LN"``).
    - **Non-gated GELU MLP** (``c_fc`` -> ``c_proj``) rather than a gated MLP.
    - **Biases on every attention and MLP projection** (``use_bias=True``).

    Unlike GPTBigCode it uses separate ``q_proj``/``k_proj``/``v_proj`` projections
    and rotary embeddings instead of a combined ``c_attn`` and learned positions.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Starcoder2 architecture adapter."""
        super().__init__(cfg)

        # Config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        self.default_config = {
            "d_model": cfg.d_model,
            "d_head": cfg.d_model // cfg.n_heads,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_vocab": cfg.d_vocab,
        }

        # GQA: Starcoder2 uses num_key_value_heads (< n_heads) for K/V projections.
        n_kv_heads = None
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.default_config["n_key_value_heads"] = cfg.n_key_value_heads
            self.cfg.n_key_value_heads = cfg.n_key_value_heads
            n_kv_heads = cfg.n_key_value_heads

        # Standard Q/K/V/O weight rearrangement, plus per-head bias rearrangement
        # (Starcoder2 has biases on the attention projections, which Llama does not).
        # K/V use the GQA head count; O has no per-head bias (its bias stays [d_model]).
        n_kv = n_kv_heads if n_kv_heads is not None else self.cfg.n_heads
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(n_kv_heads=n_kv_heads),
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=n_kv),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=n_kv),
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": NormalizationBridge(name="post_attention_layernorm", config=self.cfg),
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
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="c_fc"),
                            "out": LinearBridge(name="c_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Starcoder2 component testing.

        Starcoder2 uses RoPE. Mirror the Llama-family setup: set the shared
        ``model.rotary_emb`` reference on each attention bridge instance.

        Args:
            hf_model: The HuggingFace Starcoder2 model instance.
            bridge_model: The TransformerBridge model, if available.
        """
        rotary_emb = hf_model.model.rotary_emb

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
