"""Qwen architecture adapter."""

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
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)


class QwenArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen architecture adapter."""
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeWeightConversion("m (n h) -> n h m", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.w2.weight.T",
                "blocks.{i}.mlp.W_gate": "transformer.h.{i}.mlp.w1.weight.T",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight.T",
                "ln_final.w": "transformer.ln_f.weight",
                "unembed.W_U": "lm_head.weight.T",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "blocks": BlockBridge(
                name="transformer.h",
                submodules={
                    "ln1": LayerNormBridge(name="ln_1"),
                    "attn": AttentionBridge(name="attn"),
                    "ln2": LayerNormBridge(name="ln_2"),
                    "mlp": MLPBridge(name="mlp"),
                },
            ),
            "ln_final": LayerNormBridge(name="transformer.ln_f"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
