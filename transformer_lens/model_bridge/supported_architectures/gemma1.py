"""Gemma1 architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
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
    UnembeddingBridge,
)


class Gemma1ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma1 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma1 architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False

        # Gemma models were not trained with BOS tokens
        self.cfg.default_prepend_bos = False
        self.cfg.uses_rms_norm = True

        self.weight_processing_conversions = {
            # Gemma1 scales embeddings by sqrt(d_model)
            "embed.e": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_vocab d_model -> d_vocab d_model",
                    scale=self.cfg.d_model**0.5,
                ),
                source_key="model.embed_tokens.weight",
            ),
            "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
            "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.q_proj.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.k_proj.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.v_proj.weight",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.o_proj.weight",
            ),
            "blocks.{i}.mlp.in": "model.layers.{i}.mlp.up_proj.weight.T",
            "blocks.{i}.mlp.gate": "model.layers.{i}.mlp.gate_proj.weight.T",
            "blocks.{i}.mlp.out": "model.layers.{i}.mlp.down_proj.weight.T",
            "ln_final.w": "model.norm.weight",
            "unembed.u": "lm_head.weight.T",  # Not shared with embedding
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": EmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
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
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
