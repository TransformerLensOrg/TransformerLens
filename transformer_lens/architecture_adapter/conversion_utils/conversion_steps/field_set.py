"""Field set module for handling component mapping."""

from typing import Any

from transformers import PreTrainedModel


class FieldSet:
    """Field set for handling component mapping.
    
    This class provides functionality for mapping between HookedTransformer
    component names and the underlying model's structure.
    """

    def __init__(self, component_mapping: dict[str, str | tuple[str, dict[str, str]]]):
        """Initialize the field set.
        
        Args:
            component_mapping: The component mapping dictionary.
        """
        self.component_mapping = component_mapping

    def get_component(self, model: PreTrainedModel, name: str) -> Any:
        """Get a component from the model using the component mapping.

        This method maps HookedTransformer component names to the underlying model's structure
        using the component mapping dictionary. It handles nested structures recursively.

        Args:
            model: The model to get the component from.
            name: The name of the component to get.

        Returns:
            The requested component.

        Raises:
            ValueError: If the component path is invalid.
        """
        def resolve_path(path_parts: list[str], mapping: dict[str, Any] | tuple[str, dict[str, str]]) -> str:
            """Recursively resolve a component path to its underlying model path.
            
            Args:
                path_parts: List of path components.
                mapping: Current level of component mapping.
            
            Returns:
                The resolved path in the underlying model.
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

        return current 