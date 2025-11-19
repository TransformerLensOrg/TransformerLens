"""MinGPT architecture adapter."""

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
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class MingptArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for MinGPT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the MinGPT architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        self.weight_processing_conversions = {
            "pos_embed.pos": "transformer.wpe.weight",
            "embed.e": "transformer.wte.weight",
            "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
            "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
            "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
            "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.b_Q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.b_K": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.b_V": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (n_head d_head) -> n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_proj.weight",
            ),
            "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
            "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.c_fc.weight",
            "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
            "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight",
            "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
            "unembed.u": "lm_head.weight",
            "unembed.b_U": "lm_head.bias",
            "ln_final.w": "transformer.ln_f.weight",
            "ln_final.b": "transformer.ln_f.bias",
        }

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),  # Word token embeddings
            "pos_embed": PosEmbedBridge(name="transformer.wpe"),  # Positional embeddings
            "blocks": BlockBridge(
                name="transformer.h",  # Base path for blocks
                submodules={
                    "ln1": NormalizationBridge(
                        name="ln_1", config=self.cfg
                    ),  # Pre-attention layer norm
                    "ln2": NormalizationBridge(name="ln_2", config=self.cfg),  # Pre-MLP layer norm
                    "attn": AttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "c_attn": AttentionBridge(
                                name="c_attn", config=self.cfg
                            ),  # QKV projection
                        },
                    ),  # Full attention module
                    "mlp": MLPBridge(name="mlp"),  # Full MLP module
                },
            ),
            "ln_final": NormalizationBridge(
                name="transformer.ln_f", config=self.cfg
            ),  # Final layer norm
            "unembed": UnembeddingBridge(name="lm_head"),  # Language model head
        }
