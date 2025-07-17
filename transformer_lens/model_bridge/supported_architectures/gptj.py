"""GPTJ architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)


class GptjArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPTJ models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPTJ architecture adapter."""
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.k_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.v_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.out_proj.weight",
                    RearrangeWeightConversion("m (n h) -> n h m", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.fc_in.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.fc_in.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.fc_out.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.fc_out.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": NormalizationBridge(name="ln_1"),
                    "attn": AttentionBridge(name="attn"),
                    "mlp": MLPBridge(name="mlp"),
                },
            ),
            "ln_final": NormalizationBridge(name="transformer.ln_f"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
