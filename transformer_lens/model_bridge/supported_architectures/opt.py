"""OPT architecture adapter."""

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
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class OptArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OPT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the OPT architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # OPT models were trained with BOS tokens (inherits default_prepend_bos = True)

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.decoder.layers.{i}.self_attn.q_proj.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.decoder.layers.{i}.self_attn.k_proj.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.decoder.layers.{i}.self_attn.v_proj.weight",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                source_key="model.decoder.layers.{i}.self_attn.out_proj.weight",
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.decoder.embed_tokens"),
            "pos_embed": PosEmbedBridge(name="model.decoder.embed_positions"),
            "blocks": BlockBridge(
                name="model.decoder.layers",
                submodules={
                    "ln1": NormalizationBridge(name="self_attn_layer_norm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                        },
                    ),
                    "ln2": NormalizationBridge(name="final_layer_norm", config=self.cfg),
                    "mlp": MLPBridge(
                        name=None,  # No MLP container; fc1/fc2 are on block
                        config=self.cfg,  # Pass config for activation function
                        submodules={
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(name="model.decoder.final_layer_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
