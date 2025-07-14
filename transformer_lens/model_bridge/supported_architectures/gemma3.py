"""Gemma3 architecture adapter."""


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
    LinearBridge,
    MLPBridge,
    UnembeddingBridge,
)


class Gemma3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma3 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma3 architecture adapter."""
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                # Gemma3 scales embeddings by sqrt(d_model)
                "embed.W_E": (
                    "model.embed_tokens.weight",
                    RearrangeWeightConversion(
                        "d_vocab d_model -> d_vocab d_model",
                        scale=self.cfg.hidden_size**0.5,
                    ),
                ),
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion(
                        "(n h) m -> n m h",
                        n=getattr(self.cfg, "num_key_value_heads", self.cfg.num_attention_heads),
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion(
                        "(n h) m -> n m h",
                        n=getattr(self.cfg, "num_key_value_heads", self.cfg.num_attention_heads),
                    ),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion("m (n h) -> n h m", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.W_gate": "model.layers.{i}.mlp.gate_proj.weight.T",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight.T",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "lm_head.weight.T",  # Not shared with embedding
            }
        )

        # Set up component mapping with actual bridge instances
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": LayerNormBridge(name="input_layernorm"),
                    "ln2": LayerNormBridge(name="post_attention_layernorm"),
                    "attn": AttentionBridge(
                        name="self_attn",
                        submodules={
                            "W_Q": LinearBridge(name="q_proj"),
                            "W_K": LinearBridge(name="k_proj"),
                            "W_V": LinearBridge(name="v_proj"),
                            "W_O": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "W_gate": LinearBridge(name="gate_proj"),
                            "W_in": LinearBridge(name="up_proj"),
                            "W_out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": LayerNormBridge(name="model.norm"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
