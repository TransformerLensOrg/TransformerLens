"""GPT2 architecture adapter."""

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


class GPT2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT2 architecture adapter."""
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "pos_embed.W_pos": "transformer.wpe.weight",
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.num_attention_heads,
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.num_attention_heads,
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "m (three n h) -> three n m h",
                        three=3,
                        n=self.cfg.num_attention_heads,
                    ),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n h m", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.b_Q": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_K": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_V": "transformer.h.{i}.attn.c_attn.bias",
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight",
            }
        )

        self.component_mapping = {
            "embed": EmbeddingBridge(name="transformer.wte"),
            "pos_embed": EmbeddingBridge(name="transformer.wpe"),
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
