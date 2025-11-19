"""GPT-2 LM Head Custom architecture adapter."""

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


class Gpt2LmHeadCustomArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT-2 LM Head Custom models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT-2 LM Head Custom architecture adapter."""
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
                    "d_model (n d_head) -> n d_model d_head"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (n d_head) -> n d_model d_head"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (n d_head) -> n d_model d_head"
                ),
                source_key="transformer.h.{i}.attn.c_attn.weight",
            ),
            "blocks.{i}.attn.b_Q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n d_head) -> n d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.b_K": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n d_head) -> n d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.b_V": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n d_head) -> n d_head"),
                source_key="transformer.h.{i}.attn.c_attn.bias",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n d_head) d_model -> n d_head d_model"
                ),
                source_key="transformer.h.{i}.attn.c_proj.weight",
            ),
            "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
            "blocks.{i}.mlp.in": "transformer.h.{i}.mlp.c_fc.weight",
            "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
            "blocks.{i}.mlp.out": "transformer.h.{i}.mlp.c_proj.weight",
            "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
            "ln_final.w": "transformer.ln_f.weight",
            "ln_final.b": "transformer.ln_f.bias",
            "unembed.u": "lm_head.weight",
            # "unembed.b_U": "lm_head.bias", # gpt2 has no unembed bias
        }

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": PosEmbedBridge(name="transformer.wpe"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="ln_1", config=self.cfg),
                    "attn": AttentionBridge(name="attn", config=self.cfg),
                    "ln2": NormalizationBridge(name="ln_2", config=self.cfg),
                    "mlp": MLPBridge(name="mlp"),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
