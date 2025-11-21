"""MinGPT architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointQKVAttentionBridge,
    LinearBridge,
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
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(3 n_head d_head) -> 3 n_head d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (n_head d_head) -> n_head d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_proj.weight",
            ),
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
                    "attn": JointQKVAttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "qkv": LinearBridge(name="c_attn"),  # Combined QKV projection
                            "o": LinearBridge(name="c_proj"),  # Output projection
                        },
                    ),  # Full attention module
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="c_fc"),
                            "out": LinearBridge(name="c_proj"),
                        },
                    ),  # Full MLP module
                },
            ),
            "ln_final": NormalizationBridge(
                name="transformer.ln_f", config=self.cfg
            ),  # Final layer norm
            "unembed": UnembeddingBridge(name="lm_head"),  # Language model head
        }
