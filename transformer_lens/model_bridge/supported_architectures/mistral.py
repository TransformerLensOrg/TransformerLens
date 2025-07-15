"""Mistral architecture adapter."""

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


class MistralArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Mistral models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Mistral architecture adapter."""
        super().__init__(cfg)
        self.default_config = {
            "d_model": cfg.hidden_size,
            "d_head": cfg.hidden_size // cfg.num_attention_heads,
            "n_heads": cfg.num_attention_heads,
            "n_layers": cfg.num_hidden_layers,
            "d_vocab": cfg.vocab_size,
            "n_key_value_heads": cfg.num_key_value_heads,
        }

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_attention_heads),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_key_value_heads),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(n h) m -> n m h", n=self.cfg.num_key_value_heads),
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

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": LayerNormBridge(name="input_layernorm"),
                    "ln2": LayerNormBridge(name="post_attention_layernorm"),
                    "attn": AttentionBridge(name="self_attn"),
                    "mlp": MLPBridge(name="mlp"),
                },
            ),
            "ln_final": LayerNormBridge(name="model.norm"),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
