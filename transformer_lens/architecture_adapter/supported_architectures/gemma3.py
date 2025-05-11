"""Gemma3 architecture adapter."""

from transformers.modeling_utils import PreTrainedModel

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class Gemma3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma3 models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Gemma3 architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        # Set up weight conversion rules
        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.num_attention_heads),
                ),
                "blocks.{i}.attn._W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)),
                ),
                "blocks.{i}.attn._W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion("m (n h)->n h m", n=cfg.num_attention_heads),
                ),
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.W_gate": "model.layers.{i}.mlp.gate_proj.weight.T",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight.T",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "embed_tokens.weight.T",  # Shared with embedding
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": "model.embed_tokens",  # Word token embeddings
            "blocks": (
                "model.layers",  # Base path for blocks
                {
                    "ln1": "input_layernorm",  # Pre-attention layer norm
                    "ln2": "post_attention_layernorm",  # Post-attention layer norm
                    "attn": "self_attn",  # Full attention module
                    "mlp": "mlp",  # Full MLP module
                },
            ),
            "ln_final": "model.norm",  # Final layer norm
            "unembed": "model.embed_tokens",  # Language model head (shared with embed)
        } 

    def get_component(self, model: PreTrainedModel, name: str):
        """Get a component from the model by its name.
        Args:
            model: The HuggingFace model
            name: The name of the component to get
        Returns:
            The requested component
        """
        if name == "embed":
            return model.model.embed_tokens
        elif name == "ln_final":
            return model.model.norm
        elif name == "unembed":
            return model.model.embed_tokens
        elif name.startswith("blocks."):
            # Parse block index and component name
            parts = name.split(".")
            if len(parts) != 3:
                raise ValueError(f"Invalid block component name: {name}")
            block_idx = int(parts[1])
            block_component = parts[2]
            block = model.model.layers[block_idx]
            component_map = {
                "ln1": "input_layernorm",
                "ln2": "post_attention_layernorm",
                "attn": "self_attn",
                "mlp": "mlp"
            }
            if block_component not in component_map:
                raise ValueError(f"Unknown block component: {block_component}")
            return getattr(block, component_map[block_component])
        else:
            raise ValueError(f"Unknown component: {name}") 