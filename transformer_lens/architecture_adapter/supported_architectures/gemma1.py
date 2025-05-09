"""Gemma1 architecture adapter."""

from typing import Any

from transformers import PreTrainedModel

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class Gemma1ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma1 models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Gemma1 architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        # Set up weight conversion rules
        self.conversion_rules = WeightConversionSet(
            {
                # Gemma1 scales embeddings by sqrt(d_model)
                "embed.W_E": (
                    "model.embed_tokens.weight",
                    RearrangeWeightConversion(
                        "d_vocab d_model -> d_vocab d_model",
                        scale=cfg.d_model**0.5,
                    ),
                ),
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_heads),
                ),
                "blocks.{i}.attn._W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_key_value_heads),
                ),
                "blocks.{i}.attn._W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_key_value_heads),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion("m (n h)->n h m", n=cfg.n_heads),
                ),
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.W_gate": "model.layers.{i}.mlp.gate_proj.weight.T",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight.T",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "lm_head.weight.T",  # Not shared with embedding
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
            "unembed": "lm_head",  # Language model head (not shared with embed)
        }
        
    def get_component(self, model: PreTrainedModel, name: str) -> Any:
        """Get a component from the model.
        
        Args:
            model: The model to get the component from
            name: The name of the component to get
            
        Returns:
            The requested component
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component")
            
        def resolve_path(path_parts: list[str], mapping: dict[str, Any] | tuple[str, dict[str, str]]) -> str:
            """Recursively resolve a component path to its underlying model path.
            
            Args:
                path_parts: List of path components
                mapping: Current level of component mapping
                
            Returns:
                The resolved path in the underlying model
            """
            if not path_parts:
                raise ValueError("Empty path")
                
            # Handle tuple case (base_path, sub_mapping)
            if isinstance(mapping, tuple):
                base_path, sub_mapping = mapping
                # If we're at a leaf node (just the block index)
                if len(path_parts) == 1:
                    if not path_parts[0].isdigit():
                        raise ValueError(f"Expected layer index, got {path_parts[0]}")
                    return f"{base_path}.{path_parts[0]}"
                # Otherwise, continue with the sub_mapping
                if not path_parts[0].isdigit():
                    raise ValueError(f"Expected layer index, got {path_parts[0]}")
                layer_idx = path_parts[0]
                return f"{base_path}.{layer_idx}.{resolve_path(path_parts[1:], sub_mapping)}"
                
            # Handle dictionary case
            current = path_parts[0]
            if current not in mapping:
                raise ValueError(f"Unknown component: {current}")
                
            value = mapping[current]
            # If this is a leaf node (string path)
            if isinstance(value, str):
                if len(path_parts) == 1:
                    return value
                # If there are more parts, append them to the path
                return f"{value}.{'.'.join(path_parts[1:])}"
            # If this is a nested structure, recurse
            return resolve_path(path_parts[1:], value)
            
        # Parse the component path and resolve it
        parts = name.split(".")
        component_path = resolve_path(parts, self.component_mapping)
        
        # Navigate through the model to get the component
        current = model
        for part in component_path.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
                
        # Wrap with appropriate bridge component based on name
        if name.endswith(".attn"):
            from transformer_lens.architecture_adapter.generalized_components import (
                AttentionBridge,
            )
            return AttentionBridge(current, name)
        elif name.endswith(".mlp"):
            from transformer_lens.architecture_adapter.generalized_components import (
                MLPBridge,
            )
            return MLPBridge(current, name)
            
        # Return original component for other cases
        return current 