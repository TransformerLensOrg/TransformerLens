"""Bridge between HuggingFace and HookedTransformer models."""

from dataclasses import dataclass
from typing import Any

from transformers.modeling_utils import PreTrainedModel

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)
from transformer_lens.architecture_adapter.generalized_components.block import (
    BlockBridge,
)


@dataclass
class Block:
    """A transformer block in the bridge."""

    ln1: LayerNormBridge
    attn: AttentionBridge
    ln2: LayerNormBridge
    mlp: MLPBridge


class TransformerBridge:
    """Bridge between HuggingFace and HookedTransformer models.
    
    This class provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the HookedTransformer and HuggingFace model structures.
    """

    def __init__(self, model: PreTrainedModel, adapter: ArchitectureAdapter):
        """Initialize the bridge.
        
        Args:
            model: The HuggingFace model to bridge
            adapter: The architecture adapter to use
        """
        self.model = model
        self.adapter = adapter
        self.cfg = adapter.cfg
        
        if not hasattr(adapter, 'component_mapping') or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")
        
        # Get and replace components in the model
        embed = adapter.get_component(model, "embed")
        if not isinstance(embed, EmbeddingBridge):
            embed = EmbeddingBridge(embed, "embed")
            # Replace in model using component mapping
            path = adapter.get_component_path("embed")
            self._set_by_path(model, path, embed)
        self.embed = embed
        
        self.blocks = []
        block_bridges = []
        # Use num_hidden_layers for Hugging Face configs, fallback to n_layers
        n_layers = getattr(self.cfg, 'num_hidden_layers', getattr(self.cfg, 'n_layers', None))
        if n_layers is None:
            raise AttributeError('Config has neither num_hidden_layers nor n_layers')
        # Build blocks
        for i in range(n_layers):
            # Get block components
            ln1 = adapter.get_component(model, f"blocks.{i}.ln1")
            ln2 = adapter.get_component(model, f"blocks.{i}.ln2")
            # Wrap layer norms with bridge
            if not isinstance(ln1, LayerNormBridge):
                ln1 = LayerNormBridge(ln1, f"blocks.{i}.ln1")
                path = adapter.get_block_component_path(i, "ln1")
                self._set_by_path(model, path, ln1)
            if not isinstance(ln2, LayerNormBridge):
                ln2 = LayerNormBridge(ln2, f"blocks.{i}.ln2")
                path = adapter.get_block_component_path(i, "ln2")
                self._set_by_path(model, path, ln2)
            attn = adapter.get_component(model, f"blocks.{i}.attn")
            if not isinstance(attn, AttentionBridge):
                attn = AttentionBridge(attn, f"blocks.{i}.attn")
                path = adapter.get_block_component_path(i, "attn")
                self._set_by_path(model, path, attn)
            mlp = adapter.get_component(model, f"blocks.{i}.mlp")
            if not isinstance(mlp, MLPBridge):
                mlp = MLPBridge(mlp, f"blocks.{i}.mlp")
                path = adapter.get_block_component_path(i, "mlp")
                self._set_by_path(model, path, mlp)
            # Get the original block for reference
            original_block = None
            try:
                original_block = model.model.layers[i]
            except Exception:
                pass
            # Create block bridge
            block_bridge = BlockBridge(ln1, attn, ln2, mlp, original_component=original_block)
            block_bridges.append(block_bridge)
        # Replace model layers with block bridges in-place
        try:
            for i, block_bridge in enumerate(block_bridges):
                model.model.layers[i] = block_bridge
        except Exception:
            # Fallback: try to replace the whole layers attribute
            try:
                model.model.layers = type(model.model.layers)(block_bridges)
            except Exception:
                model.model.layers = block_bridges
        self.blocks = block_bridges
        
        # Get final components
        ln_final = adapter.get_component(model, "ln_final")
        if not isinstance(ln_final, LayerNormBridge):
            ln_final = LayerNormBridge(ln_final, "ln_final")
            # Replace in model using component mapping
            path = adapter.get_component_path("ln_final")
            self._set_by_path(model, path, ln_final)
        self.ln_final = ln_final
        
        unembed = adapter.get_component(model, "unembed")
        if not isinstance(unembed, UnembeddingBridge):
            unembed = UnembeddingBridge(unembed, "unembed")
            # Replace in model using component mapping
            path = adapter.get_component_path("unembed")
            self._set_by_path(model, path, unembed)
        self.unembed = unembed
        
    def _set_by_path(self, obj: Any, path: str, value: Any) -> None:
        """Set a value in an object by its path.
        
        Args:
            obj: The object to modify
            path: The dot-separated path to the attribute
            value: The value to set
        """
        parts = path.split(".")
        for part in parts[:-1]:
            if part == "model":
                obj = obj.model
            elif part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        
    def __str__(self) -> str:
        """Get a string representation of the bridge.
        
        Returns:
            A string describing the bridge's components
        """
        lines = []
        lines.append("TransformerBridge:")
        lines.append(f"  embed: {type(self.embed).__name__}({type(self.embed.original_component).__name__})")
        lines.append(f"  ln_final: {type(self.ln_final).__name__}({type(self.ln_final.original_component).__name__})")
        lines.append(f"  unembed: {type(self.unembed).__name__}({type(self.unembed.original_component).__name__})")
        if self.blocks:
            block = self.blocks[0]
            lines.append(f"  blocks: {type(block).__name__}({type(getattr(block, 'original_component', type(block))).__name__})")
            lines.append(f"    ln1: {type(block.ln1).__name__}({type(block.ln1.original_component).__name__})")
            lines.append(f"    attn: {type(block.attn).__name__}({type(block.attn.original_component).__name__})")
            lines.append(f"    ln2: {type(block.ln2).__name__}({type(block.ln2.original_component).__name__})")
            lines.append(f"    mlp: {type(block.mlp).__name__}({type(block.mlp.original_component).__name__})")
            # Show any additional components in the block
            for attr_name, attr_value in vars(block).items():
                if attr_name not in ['ln1', 'attn', 'ln2', 'mlp'] and hasattr(attr_value, 'original_component'):
                    lines.append(f"    {attr_name}: {type(attr_value).__name__}({type(attr_value.original_component).__name__})")
        else:
            lines.append("  blocks: None")
        return "\n".join(lines)
        
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate text using the underlying model.
        
        Args:
            *args: Positional arguments to pass to the model's generate method
            **kwargs: Keyword arguments to pass to the model's generate method
            
        Returns:
            The generated output from the model
        """
        # Get hooks from kwargs if provided
        hooks = kwargs.pop('hooks', {})
        
        # Register hooks if provided
        registered_hooks = []
        if hooks:
            for block_idx, block_hooks in hooks.items():
                if isinstance(block_idx, int) and 0 <= block_idx < len(self.blocks):
                    block = self.blocks[block_idx]
                    for component_name, component_hooks in block_hooks.items():
                        component = getattr(block, component_name, None)
                        if component is not None:
                            for hook_point, hook_fn in component_hooks.items():
                                hook = getattr(component, hook_point, None)
                                if hook is not None:
                                    hook.add_hook(hook_fn)
                                    registered_hooks.append((hook, hook_fn))
        
        try:
            # Generate text
            return self.model.generate(*args, **kwargs)
        finally:
            # Clean up hooks
            for hook, hook_fn in registered_hooks:
                hook.remove_hooks()

    def forward(self, *args, **kwargs):
        """Forward pass through the underlying HuggingFace model, with all arguments passed through."""
        # Pre-processing (if needed)
        output = self.model(*args, **kwargs)
        # Post-processing (if needed)
        return output

    def run_with_cache(self, *args, **kwargs):
        """Run the model and cache all activations at HookPoint objects.
        Returns (output, cache_dict)."""
        import torch.nn as nn

        from transformer_lens.hook_points import HookPoint

        cache = {}
        hooks = []
        visited = set()

        def make_cache_hook(name):
            def cache_hook(tensor, hook):
                cache[name] = tensor.detach().cpu()
                return tensor
            return cache_hook

        # Recursively collect all HookPoint objects and their names
        def collect_hookpoints(module, prefix=""):
            obj_id = id(module)
            if obj_id in visited:
                return
            visited.add(obj_id)
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(module, attr_name)
                except Exception:
                    continue
                name = f"{prefix}.{attr_name}" if prefix else attr_name
                if isinstance(attr, HookPoint):
                    hooks.append((attr, name))
                elif isinstance(attr, nn.Module):
                    collect_hookpoints(attr, name)
                elif isinstance(attr, (list, tuple)):
                    for i, item in enumerate(attr):
                        if isinstance(item, nn.Module):
                            collect_hookpoints(item, f"{name}[{i}]")
        collect_hookpoints(self)

        # Register hooks
        for hp, name in hooks:
            hp.add_hook(make_cache_hook(name))

        try:
            # Run the underlying model's forward method
            output = self.model(*args, **kwargs)
        finally:
            # Remove hooks
            for hp, _ in hooks:
                hp.remove_hooks()
        return output, cache 