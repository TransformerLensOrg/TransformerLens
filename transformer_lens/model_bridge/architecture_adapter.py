"""Architecture adapter base class.

This module contains the base class for architecture adapters that map between different model architectures.
"""

from typing import Any, cast

import torch
from torch import nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import HookConversionSet
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
    JointQKVAttentionBridge,
)
from transformer_lens.model_bridge.types import (
    ComponentMapping,
    RemoteComponent,
    RemoteModel,
    RemotePath,
    TransformerLensPath,
)


class ArchitectureAdapter:
    """Base class for architecture adapters.

    This class provides the interface for adapting between different model architectures.
    It handles both component mapping (for accessing model parts) and weight conversion
    (for initializing weights from one format to another).
    """

    default_cfg: dict[str, Any] = {}

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        """Initialize the architecture adapter.

        Args:
            cfg: The configuration object.
        """
        self.cfg = cfg

        self.component_mapping: ComponentMapping | None = None
        self.conversion_rules: HookConversionSet | None = None

        # Configuration for attention weight handling
        self.uses_split_attention: bool = getattr(cfg, "uses_split_attention", False)

        # Merge default_cfg into cfg for missing variables
        self._merge_default_config()

    def _merge_default_config(self) -> None:
        """Merge default_cfg into cfg for variables that don't exist in cfg."""
        for key, value in self.default_cfg.items():
            if not hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

    def get_component_mapping(self) -> ComponentMapping:
        """Get the full component mapping.

        Returns:
            The component mapping dictionary

        Raises:
            ValueError: If the component mapping is not set
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component_mapping")
        return self.component_mapping

    def get_remote_component(self, model: RemoteModel, path: RemotePath) -> RemoteComponent:
        """Get a component from a remote model by its path.

        This method should be overridden by subclasses to provide the logic for
        accessing components in a specific model architecture.

        Args:
            model: The remote model
            path: The path to the component in the remote model's format

        Returns:
            The component (e.g., a PyTorch module)

        Raises:
            AttributeError: If a component in the path doesn't exist
            IndexError: If an invalid index is accessed
            ValueError: If the path is empty or invalid

        Examples:
            Get an embedding component:

            >>> # adapter.get_remote_component(model, "model.embed_tokens")
            >>> # <Embedding>

            Get a transformer block:

            >>> # adapter.get_remote_component(model, "model.layers.0")
            >>> # <TransformerBlock>

            Get a layer norm component:

            >>> # adapter.get_remote_component(model, "model.layers.0.ln1")
            >>> # <LayerNorm>
        """

        current = model
        for part in path.split("."):
            if part.isdigit():
                current = current[int(part)]  # type: ignore[index]
            else:
                current = getattr(current, part)
        return current

    def get_component_from_list_module(
        self,
        list_module: RemoteComponent,
        bridge_component: GeneralizedComponent,
        parts: list[str],
    ) -> RemoteComponent:
        """Get a component from a list module using the bridge component and the transformer lens path.
        Args:
            list_module: The remote list module to get the component from
            bridge_component: The bridge component
            parts: The parts of the transformer lens path to navigate
        Returns:
            The requested component from the list module described by the path
        """

        # Handle list item indexing (like blocks)
        item_index = parts[1]
        if not item_index.isdigit():
            raise ValueError(f"Expected item index, got {item_index}")

        if not hasattr(list_module, "__getitem__"):
            raise TypeError(f"Component {bridge_component.name} is not indexable")

        # Cast to indicate to mypy that list_module is indexable after the check
        indexable_container = cast(Any, list_module)
        item = indexable_container[int(item_index)]

        if len(parts) == 2:
            # Just return the item
            return item
        else:
            # Get subcomponent from the item using bridge mapping
            subcomponent_name = parts[2]

            # Check the submodules attribute for bridge submodules
            if subcomponent_name in bridge_component.submodules:
                subcomponent_bridge = bridge_component.submodules[subcomponent_name]

                # If there are more parts (like blocks.0.attn.W_Q), navigate deeper
                if len(parts) > 3:
                    # Navigate through the deeper subcomponents
                    current_bridge = subcomponent_bridge
                    if subcomponent_bridge.name is None:
                        current = item
                    else:
                        current = getattr(item, subcomponent_bridge.name)

                    for i in range(3, len(parts)):
                        deeper_component_name = parts[i]

                        if deeper_component_name.isdigit() and current_bridge.is_list_item:
                            # We are dealing with a nested BlockBridge, call the function recursively
                            # and pass the path (parts) starting from the nested BlockBridge
                            return self.get_component_from_list_module(
                                current, current_bridge, parts[i - 1 :]
                            )

                        # Check submodules for deeper components
                        if deeper_component_name in current_bridge.submodules:
                            current_bridge = current_bridge.submodules[deeper_component_name]
                            if current_bridge.name is None:
                                # No container, stay at current level
                                pass
                            else:
                                current = getattr(current, current_bridge.name)
                        else:
                            raise ValueError(
                                f"Component {deeper_component_name} not found in {'.'.join(parts[:i])} components"
                            )

                    return current
                else:
                    # Just the 3-level path
                    if subcomponent_bridge.name is None:
                        return item
                    else:
                        return getattr(item, subcomponent_bridge.name)
            else:
                raise ValueError(
                    f"Component {subcomponent_name} not found in {parts[0]} components"
                )

    def get_generalized_component(self, path: TransformerLensPath) -> GeneralizedComponent:
        """Get the generalized component (bridge component) for a given TransformerLens path.

        Args:
            path: The TransformerLens path to get the component for

        Returns:
            The generalized component that handles this path

        Raises:
            ValueError: If component_mapping is not set or if the component is not found

        Examples:
            Get the embedding bridge component:

            >>> # adapter.get_generalized_component("embed")
            >>> # <EmbeddingBridge>

            Get the attention bridge component:

            >>> # adapter.get_generalized_component("blocks.0.attn")
            >>> # <AttentionBridge>
        """
        if self.component_mapping is None:
            raise ValueError(
                "component_mapping must be set before calling get_generalized_component"
            )

        # Strip parameter suffixes to get the component path
        component_path, _ = self._preprocess_parameter_path(path)
        parts = component_path.split(".")
        if not parts:
            raise ValueError("Empty path")

        # Get the top-level component from the mapping
        if parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        bridge_component = self.component_mapping[parts[0]]

        if len(parts) == 1:
            # Simple case: just return the top-level component
            return bridge_component

        # For nested paths, navigate through the component hierarchy
        current_component = bridge_component
        for i in range(1, len(parts)):
            part = parts[i]

            # Handle list item indexing (like blocks.0)
            if part.isdigit():
                # For list items, we return the bridge component itself
                # since the indexing is handled at the model level
                continue

            # Navigate to subcomponent
            if hasattr(current_component, "submodules") and part in current_component.submodules:
                current_component = current_component.submodules[part]
            else:
                # Check if this is an attention parameter (q, k, v, o) that should map to the attention component
                # This handles cases like "blocks.0.attn.W_Q" -> "blocks.0.attn.q" -> return attention component
                if (
                    hasattr(current_component, "__class__")
                    and "AttentionBridge" in current_component.__class__.__name__
                    and part in ["q", "k", "v", "o"]
                ):
                    # Check if this is a JointQKVAttentionBridge (like GPT-2) or regular AttentionBridge (like Gemma3)
                    if "JointQKV" in current_component.__class__.__name__:
                        # For joint QKV attention, return the attention component itself
                        # since the individual q, k, v, o are handled as attributes, not submodules
                        continue
                    else:
                        # For separate Q, K, V attention (like Gemma3), navigate to the subcomponent
                        if (
                            hasattr(current_component, "submodules")
                            and part in current_component.submodules
                        ):
                            current_component = current_component.submodules[part]
                            continue
                # Check if this is an MLP parameter (in, out, gate) that should map to the MLP component
                # This handles cases like "blocks.0.mlp.W_in" -> "blocks.0.mlp.in" -> return MLP component
                elif (
                    hasattr(current_component, "__class__")
                    and "MLPBridge" in current_component.__class__.__name__
                    and part in ["in", "out", "gate"]
                ):
                    # Check if this MLP has separate subcomponents (like Gemma3) or property aliases (like GPT-2)
                    if (
                        hasattr(current_component, "submodules")
                        and part in current_component.submodules
                    ):
                        # For separate MLP components (like Gemma3), navigate to the subcomponent
                        current_component = current_component.submodules[part]
                        continue
                    else:
                        # For property alias MLP (like GPT-2), return the MLP component itself
                        continue
                else:
                    raise ValueError(
                        f"Component {part} not found in {'.'.join(parts[:i])} components"
                    )

        return current_component

    def get_component(self, model: RemoteModel, path: TransformerLensPath) -> RemoteComponent:
        """Get a component from the model using the component_mapping.

        Args:
            model: The model to extract components from
            path: The path of the component to get, as defined in component_mapping

        Returns:
            The requested component from the model

        Raises:
            ValueError: If component_mapping is not set or if the component is not found
            AttributeError: If a component in the path doesn't exist
            IndexError: If an invalid index is accessed

        Examples:
            Get an embedding component:

            >>> # adapter.get_component(model, "embed")
            >>> # <Embedding>

            Get a transformer block:

            >>> # adapter.get_component(model, "blocks.0")
            >>> # <TransformerBlock>

            Get a layer norm component:

            >>> # adapter.get_component(model, "blocks.0.ln1")
            >>> # <LayerNorm>
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component")

        # In the new system, we get the bridge component from the mapping
        # and use its name attribute to get the remote component
        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")

        # Get the top-level component from the mapping
        if self.component_mapping is None or parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        bridge_component = self.component_mapping[parts[0]]

        if len(parts) == 1:
            # Simple case: just return the component at the bridge's remote path
            if bridge_component.name is None:
                return model
            return self.get_remote_component(model, bridge_component.name)

        # For nested paths like "blocks.0.attn", we need to handle the indexing
        if bridge_component.is_list_item and len(parts) >= 2:
            # Get the remote ModuleList for the indexed item
            if bridge_component.name is None:
                raise ValueError(f"List component {parts[0]} must have a name")
            list_module = self.get_remote_component(model, bridge_component.name)
            return self.get_component_from_list_module(list_module, bridge_component, parts)

        # For other nested paths, navigate through the remote model
        remote_path = bridge_component.name
        if remote_path is None:
            raise ValueError(f"Component {parts[0]} must have a name for nested paths")
        if len(parts) > 1:
            remote_path = f"{remote_path}.{'.'.join(parts[1:])}"

        return self.get_remote_component(model, remote_path)

    def translate_transformer_lens_path(
        self, path: TransformerLensPath, last_component_only: bool = False
    ) -> RemotePath:
        """Translate a TransformerLens path to a remote model path.

        Args:
            path: The TransformerLens path to translate
            last_component_only: If True, return only the last component of the path

        Returns:
            The corresponding remote model path

        Raises:
            ValueError: If the path is not found in the component mapping
        """
        if self.component_mapping is None:
            raise ValueError(
                "component_mapping must be set before calling translate_transformer_lens_path"
            )

        # Preprocess the path to handle parameter name mapping
        path, param_suffix = self._preprocess_parameter_path(path)

        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")

        # Get the top-level component from the mapping
        if parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        bridge_component = self.component_mapping[parts[0]]

        if len(parts) == 1:
            # Simple case: just return the bridge's remote path
            remote_path = bridge_component.name
            if remote_path is None:
                raise ValueError(f"Component {parts[0]} must have a name for path translation")
            # Add parameter suffix from preprocessing
            if param_suffix:
                remote_path = remote_path + param_suffix
            if last_component_only:
                return remote_path.split(".")[-1]
            return remote_path

        # For nested paths like "blocks.0.attn", we need to handle the indexing
        if bridge_component.is_list_item and len(parts) >= 2:
            # Handle list item indexing (like blocks)
            item_index = parts[1]
            if not item_index.isdigit():
                raise ValueError(f"Expected item index, got {item_index}")

            # Get the base items path
            items_path = bridge_component.name
            if items_path is None:
                raise ValueError(f"List component {parts[0]} must have a name for path translation")

            if len(parts) == 2:
                # Just return the indexed item path
                remote_path = f"{items_path}.{item_index}"
                # Add parameter suffix from preprocessing
                if param_suffix:
                    remote_path = remote_path + param_suffix
                if last_component_only:
                    return remote_path.split(".")[-1]
                return remote_path
            else:
                # Get subcomponent from the item bridge
                subcomponent_name = parts[2]

                # Check the submodules attribute for bridge submodules
                if subcomponent_name in bridge_component.submodules:
                    subcomponent_bridge = bridge_component.submodules[subcomponent_name]

                    # If there are more parts (like blocks.0.attn.q_proj), navigate deeper
                    if len(parts) > 3:
                        # Navigate through the deeper subcomponents
                        current_bridge = subcomponent_bridge
                        subcomponent_name_str = subcomponent_bridge.name
                        if subcomponent_name_str is None:
                            raise ValueError(
                                f"Subcomponent {subcomponent_name} must have a name for path translation"
                            )
                        remote_path_parts = [items_path, item_index, subcomponent_name_str]

                        for i in range(3, len(parts)):
                            deeper_component_name = parts[i]

                            # Check submodules for deeper components
                            if deeper_component_name in current_bridge.submodules:
                                current_bridge = current_bridge.submodules[deeper_component_name]
                                deeper_name = current_bridge.name
                                if deeper_name is None:
                                    raise ValueError(
                                        f"Component {deeper_component_name} must have a name for path translation"
                                    )
                                remote_path_parts.append(deeper_name)
                            else:
                                raise ValueError(
                                    f"Component {deeper_component_name} not found in {'.'.join(parts[:i])} components"
                                )

                        remote_path = ".".join(remote_path_parts)
                        # Add parameter suffix from preprocessing
                        if param_suffix:
                            remote_path = remote_path + param_suffix
                        if last_component_only:
                            return remote_path.split(".")[-1]
                        return remote_path
                    else:
                        # Just the 3-level path
                        subcomponent_name_str = subcomponent_bridge.name
                        if subcomponent_name_str is None:
                            raise ValueError(
                                f"Subcomponent {subcomponent_name} must have a name for path translation"
                            )
                        remote_path = f"{items_path}.{item_index}.{subcomponent_name_str}"
                        # Add parameter suffix from preprocessing
                        if param_suffix:
                            remote_path = remote_path + param_suffix
                        if last_component_only:
                            return remote_path.split(".")[-1]
                        return remote_path
                else:
                    raise ValueError(
                        f"Component {subcomponent_name} not found in {parts[0]} components"
                    )

        # For other nested paths, navigate through the bridge components
        remote_path = bridge_component.name
        if remote_path is None:
            raise ValueError(f"Component {parts[0]} must have a name for path translation")
        if len(parts) > 1:
            remote_path = f"{remote_path}.{'.'.join(parts[1:])}"

        # Add parameter suffix from preprocessing
        if param_suffix:
            remote_path = remote_path + param_suffix

        if last_component_only:
            return remote_path.split(".")[-1]
        return remote_path

    def _preprocess_parameter_path(self, path: str) -> tuple[str, str]:
        """Preprocess TransformerLens path to map parameter names to component names.

        Args:
            path: The original TransformerLens path

        Returns:
            Tuple of (preprocessed_path, parameter_suffix)
        """
        # Determine parameter suffix from the original path
        param_suffix = ""  # Initialize to handle all code paths
        if path.endswith(
            (
                ".W_Q",
                ".W_K",
                ".W_V",
                ".W_O",
                ".W_in",
                ".W_out",
                ".W_gate",
                ".W_E",
                ".W_U",
                ".W_pos",
                ".w",
                "._W_K",
                "._W_V",
            )
        ):
            param_suffix = ".weight"
        elif path.endswith(
            (
                ".b_Q",
                ".b_K",
                ".b_V",
                ".b_O",
                ".b_in",
                ".b_out",
                ".b_gate",
                ".b_E",
                ".b_U",
                ".b_pos",
                ".b",
                "._b_K",
                "._b_V",
            )
        ):
            param_suffix = ".bias"

        # Handle attention weights based on actual architecture
        # Check if this is an attention weight that needs architecture-specific mapping
        if any(
            path.endswith(suffix)
            for suffix in [
                ".W_Q",
                ".W_K",
                ".W_V",
                ".b_Q",
                ".b_K",
                ".b_V",
                "._W_K",
                "._W_V",
                "._b_K",
                "._b_V",
            ]
        ):
            # Extract the attention component path (e.g., "blocks.0.attn")
            attn_path_parts = path.split(".")
            if len(attn_path_parts) >= 3 and attn_path_parts[-2] == "attn":
                attn_component_path = ".".join(attn_path_parts[:-1])  # e.g., "blocks.0.attn"

                # Check what attention components are actually available
                try:
                    if self.component_mapping:
                        # Navigate to the attention component to see what submodules it has
                        current_mapping = self.component_mapping
                        for part in attn_component_path.split("."):
                            if (
                                hasattr(current_mapping, "submodules")
                                and part in current_mapping.submodules
                            ):
                                current_mapping = current_mapping.submodules[part]  # type: ignore
                            elif hasattr(current_mapping, "__getitem__"):
                                current_mapping = current_mapping[part]  # type: ignore

                        # Check available attention subcomponents
                        if hasattr(current_mapping, "submodules"):
                            attn_components = list(current_mapping.submodules.keys())

                            # If we have a combined qkv component, map all Q/K/V to it
                            if "qkv" in attn_components:
                                path = path.replace(".W_Q", ".qkv")
                                path = path.replace(".W_K", ".qkv")
                                path = path.replace(".W_V", ".qkv")
                                path = path.replace(".b_Q", ".qkv")
                                path = path.replace(".b_K", ".qkv")
                                path = path.replace(".b_V", ".qkv")
                                # Handle GQA-specific paths
                                path = path.replace("._W_K", ".qkv")
                                path = path.replace("._W_V", ".qkv")
                                path = path.replace("._b_K", ".qkv")
                                path = path.replace("._b_V", ".qkv")
                            # If we have separate q, k, v components, map individually
                            elif all(comp in attn_components for comp in ["q", "k", "v"]):
                                path = path.replace(".W_Q", ".q")
                                path = path.replace(".W_K", ".k")
                                path = path.replace(".W_V", ".v")
                                path = path.replace(".b_Q", ".q")
                                path = path.replace(".b_K", ".k")
                                path = path.replace(".b_V", ".v")
                                # Handle GQA-specific paths - map to regular k/v components
                                path = path.replace("._W_K", ".k")
                                path = path.replace("._W_V", ".v")
                                path = path.replace("._b_K", ".k")
                                path = path.replace("._b_V", ".v")
                            # If we have qkv_proj (like some other architectures), use that
                            elif "qkv_proj" in attn_components:
                                path = path.replace(".W_Q", ".qkv_proj")
                                path = path.replace(".W_K", ".qkv_proj")
                                path = path.replace(".W_V", ".qkv_proj")
                                path = path.replace(".b_Q", ".qkv_proj")
                                path = path.replace(".b_K", ".qkv_proj")
                                path = path.replace(".b_V", ".qkv_proj")
                except Exception:
                    # Fallback to default behavior if component mapping inspection fails
                    pass

        # If no architecture-specific mapping was applied, use default fallback
        if any(
            path.endswith(suffix) for suffix in [".W_Q", ".W_K", ".W_V", ".b_Q", ".b_K", ".b_V"]
        ):
            # Default fallback - assume separate components
            path = path.replace(".W_Q", ".q")
            path = path.replace(".W_K", ".k")
            path = path.replace(".W_V", ".v")
            path = path.replace(".b_Q", ".q")
            path = path.replace(".b_K", ".k")
            path = path.replace(".b_V", ".v")

        # Handle other attention weights
        path = path.replace(".W_O", ".o")
        path = path.replace(".b_O", ".o")

        # Handle MLP weights based on actual architecture
        # Check if this is an MLP weight that needs architecture-specific mapping
        if any(
            path.endswith(suffix)
            for suffix in [".W_in", ".W_out", ".b_in", ".b_out", ".ln.w", ".ln.b"]
        ):
            # Extract the MLP component path (e.g., "blocks.0.mlp")
            mlp_path_parts = path.split(".")
            if len(mlp_path_parts) >= 3 and mlp_path_parts[-2] == "mlp":
                mlp_component_path = ".".join(mlp_path_parts[:-1])  # e.g., "blocks.0.mlp"

                # Check what MLP components are actually available
                try:
                    if self.component_mapping:
                        # Navigate to the MLP component to see what submodules it has
                        current_mapping = self.component_mapping
                        for part in mlp_component_path.split("."):
                            if (
                                hasattr(current_mapping, "submodules")
                                and part in current_mapping.submodules
                            ):
                                current_mapping = current_mapping.submodules[part]  # type: ignore
                            elif hasattr(current_mapping, "__getitem__"):
                                current_mapping = current_mapping[part]  # type: ignore

                        # Check available MLP subcomponents
                        if hasattr(current_mapping, "submodules"):
                            mlp_components = list(current_mapping.submodules.keys())

                            # Map based on available components
                            if "input" in mlp_components and "out" in mlp_components:
                                # GPT-2 style: input/out
                                path = path.replace(".W_in", ".input")
                                path = path.replace(".b_in", ".input")
                                path = path.replace(".W_out", ".out")
                                path = path.replace(".b_out", ".out")
                            elif "in" in mlp_components and "out" in mlp_components:
                                # Standard style: in/out
                                path = path.replace(".W_in", ".in")
                                path = path.replace(".b_in", ".in")
                                path = path.replace(".W_out", ".out")
                                path = path.replace(".b_out", ".out")
                            elif "fc_in" in mlp_components and "fc_out" in mlp_components:
                                # Some other style: fc_in/fc_out
                                path = path.replace(".W_in", ".fc_in")
                                path = path.replace(".b_in", ".fc_in")
                                path = path.replace(".W_out", ".fc_out")
                                path = path.replace(".b_out", ".fc_out")

                            # Handle SoLU MLP layer norm paths
                            if "ln" in mlp_components:
                                path = path.replace(".ln.w", ".ln")
                                path = path.replace(".ln.b", ".ln")
                except Exception:
                    # Fallback to default behavior if component mapping inspection fails
                    pass

        # If no architecture-specific mapping was applied, use default fallback for MLP
        if any(path.endswith(suffix) for suffix in [".W_in", ".W_out", ".b_in", ".b_out"]):
            # Default fallback - assume standard in/out components
            path = path.replace(".W_in", ".in")
            path = path.replace(".b_in", ".in")
            path = path.replace(".W_out", ".out")
            path = path.replace(".b_out", ".out")
        path = path.replace(".W_gate", ".gate")
        path = path.replace(".b_gate", ".gate")

        # Handle embedding/unembedding weights (these keep their suffix)
        if not (path.endswith(".weight") or path.endswith(".bias")):
            path = path.replace(".W_E", "")
            path = path.replace(".b_E", "")
            path = path.replace(".W_U", "")
            path = path.replace(".b_U", "")
            path = path.replace(".W_pos", "")
            path = path.replace(".b_pos", "")
            path = path.replace(".w", "")
            path = path.replace(".b", "")

        return path, param_suffix

    def _translate_parameter_name(self, remote_path: str, original_path: str) -> str:
        """Translate parameter names from TransformerLens format to target format.

        Since preprocessing handles most parameter mapping, this method just
        handles any remaining cases.

        Args:
            remote_path: The translated component path
            original_path: The original TransformerLens path

        Returns:
            The path with parameter names translated
        """
        # Most parameter translation is handled by preprocessing,
        # so this method is now much simpler
        return remote_path

    def convert_weights(self, hf_model: nn.Module) -> dict[str, torch.Tensor]:
        """Convert the weights from the HuggingFace format to the HookedTransformer format.

        Args:
            hf_model: The HuggingFace model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        if self.conversion_rules is None:
            raise ValueError("conversion_rules must be set before calling convert_weights")
        state_dict = self.conversion_rules.convert(input_value=hf_model)

        # Flatten state dictionary such that PyTorch can load it properly
        flattened_state_dict = self.flatten_nested_dict(state_dict)
        return flattened_state_dict

    def flatten_nested_dict(
        self,
        input: dict[str, torch.Tensor] | list[Any] | torch.Tensor,
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, torch.Tensor]:
        """
        Flattens a nested dictionary/list structure into a flat dictionary with dot notation.

        Args:
            input: The input structure (can be dict, list, or a value)
            parent_key: The parent key for the current item (used in recursion)
            sep: Separator to use between nested keys (default '.')

        Returns:
            dict: Flattened dictionary with dot notation keys
        """
        items = {}

        if isinstance(input, dict):
            for k, v in input.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.update(self.flatten_nested_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v

        elif isinstance(input, list):
            for i, v in enumerate(input):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.update(self.flatten_nested_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
        else:
            items[parent_key] = input

        return items

    def extract_weights_using_components(self, model) -> dict[str, torch.Tensor]:
        """Extract weights in TransformerLens format using component-based weight processing.

        This method uses the architecture adapter's component mapping to process weights
        through each component's process_weights() method, ensuring consistency with
        the component-based approach.

        Args:
            model: The original model to extract weights from

        Returns:
            dict[str, torch.Tensor]: Weights in TransformerLens format
        """
        if self.component_mapping is None:
            raise ValueError("Architecture adapter component mapping not initialized")

        tl_state_dict = {}

        # Process top-level components (embed, pos_embed, ln_final, unembed)
        for comp_name, component in self.component_mapping.items():
            if comp_name != "blocks":
                try:
                    # Get the original component from the model
                    original_component = self.get_component(model, comp_name)

                    # Create a fresh instance to avoid any state issues
                    component_class = type(component)
                    if comp_name in ["ln_final"]:
                        # Components that need config
                        fresh_component = component_class(name=component.name, config=self.cfg)
                    else:
                        # Components that don't need config
                        fresh_component = component_class(name=component.name)

                    fresh_component.set_original_component(original_component)
                    fresh_component.process_weights()
                    component_weights = fresh_component.get_processed_state_dict()

                    # Add weights with component prefix
                    for key, value in component_weights.items():
                        tl_state_dict[f"{comp_name}.{key}"] = value.clone()
                except Exception as e:
                    print(f"Warning: Failed to process component {comp_name}: {e}")

        # Process transformer blocks using the configured component mapping
        blocks_component = self.component_mapping["blocks"]
        for layer_idx in range(self.cfg.n_layers):
            try:
                # Process each subcomponent for this layer
                for subcomp_name, subcomponent in blocks_component.submodules.items():
                    try:
                        # Get the original subcomponent for this layer
                        original_subcomponent = self.get_component(
                            model, f"blocks.{layer_idx}.{subcomp_name}"
                        )

                        # Create a fresh instance using the configured component
                        component_class = type(subcomponent)
                        if subcomp_name in ["ln1", "ln2"]:
                            # Normalization components need config
                            fresh_component = component_class(
                                name=subcomponent.name, config=self.cfg
                            )
                        elif subcomp_name == "attn":
                            # Attention component needs config and split function (if it's a JointQKVAttentionBridge)
                            if issubclass(component_class, JointQKVAttentionBridge):
                                attn_name = subcomponent.name
                                if attn_name is None:
                                    raise ValueError("Attention component must have a name")
                                if hasattr(self, "split_qkv_matrix"):
                                    fresh_component = component_class(
                                        name=attn_name,
                                        config=self.cfg,
                                        split_qkv_matrix=self.split_qkv_matrix,
                                    )
                                else:
                                    # Fallback for non-GPT2 architectures
                                    def dummy_split_qkv_matrix(attn_layer):
                                        return None, None, None

                                    fresh_component = component_class(
                                        name=attn_name,
                                        config=self.cfg,
                                        split_qkv_matrix=dummy_split_qkv_matrix,
                                    )
                            else:
                                # Regular attention bridge - no split_qkv_matrix parameter
                                fresh_component = component_class(
                                    name=subcomponent.name,
                                    config=self.cfg,
                                )
                        elif subcomp_name == "mlp":
                            # MLP component - process its subcomponents
                            if hasattr(subcomponent, "submodules"):
                                for (
                                    mlp_subcomp_name,
                                    mlp_subcomponent,
                                ) in subcomponent.submodules.items():
                                    try:
                                        # Get the original MLP subcomponent
                                        original_mlp_subcomp = self.get_component(
                                            model, f"blocks.{layer_idx}.mlp.{mlp_subcomp_name}"
                                        )

                                        # Create specialized linear component with correct key naming
                                        from typing import Union

                                        from transformer_lens.model_bridge.generalized_components.linear import (
                                            LinearBridge,
                                        )

                                        mlp_fresh_component: Union[
                                            "LinearBridge",
                                            "MLPInputLinearBridge",
                                            "MLPOutputLinearBridge",
                                        ]

                                        if mlp_subcomp_name == "input":

                                            class MLPInputLinearBridge(LinearBridge):
                                                def process_weights(
                                                    self,
                                                    fold_ln: bool = False,
                                                    center_writing_weights: bool = False,
                                                    center_unembed: bool = False,
                                                    fold_value_biases: bool = False,
                                                    refactor_factored_attn_matrices: bool = False,
                                                    **kwargs,
                                                ) -> None:
                                                    if self.original_component is None:
                                                        return
                                                    weight_tensor = getattr(
                                                        self.original_component, "weight", None
                                                    )
                                                    bias_tensor = getattr(
                                                        self.original_component, "bias", None
                                                    )
                                                    processed_weights = {}
                                                    if weight_tensor is not None:
                                                        processed_weights[
                                                            "W_in"
                                                        ] = weight_tensor.clone()
                                                    if bias_tensor is not None:
                                                        processed_weights[
                                                            "b_in"
                                                        ] = bias_tensor.clone()
                                                    self._processed_weights = processed_weights

                                            mlp_input_name = mlp_subcomponent.name
                                            if mlp_input_name is None:
                                                raise ValueError(
                                                    "MLP input component must have a name"
                                                )
                                            mlp_fresh_component = MLPInputLinearBridge(
                                                name=mlp_input_name
                                            )
                                        elif mlp_subcomp_name == "out":

                                            class MLPOutputLinearBridge(LinearBridge):
                                                def process_weights(
                                                    self,
                                                    fold_ln: bool = False,
                                                    center_writing_weights: bool = False,
                                                    center_unembed: bool = False,
                                                    fold_value_biases: bool = False,
                                                    refactor_factored_attn_matrices: bool = False,
                                                    **kwargs,
                                                ) -> None:
                                                    if self.original_component is None:
                                                        return
                                                    weight_tensor = getattr(
                                                        self.original_component, "weight", None
                                                    )
                                                    bias_tensor = getattr(
                                                        self.original_component, "bias", None
                                                    )
                                                    processed_weights = {}
                                                    if weight_tensor is not None:
                                                        processed_weights[
                                                            "W_out"
                                                        ] = weight_tensor.clone()
                                                    if bias_tensor is not None:
                                                        processed_weights[
                                                            "b_out"
                                                        ] = bias_tensor.clone()
                                                    self._processed_weights = processed_weights

                                            mlp_output_name = mlp_subcomponent.name
                                            if mlp_output_name is None:
                                                raise ValueError(
                                                    "MLP output component must have a name"
                                                )
                                            mlp_fresh_component = MLPOutputLinearBridge(
                                                name=mlp_output_name
                                            )
                                        else:
                                            mlp_generic_name = mlp_subcomponent.name
                                            if mlp_generic_name is None:
                                                raise ValueError(
                                                    f"MLP component {mlp_subcomp_name} must have a name"
                                                )
                                            mlp_fresh_component = LinearBridge(
                                                name=mlp_generic_name
                                            )

                                        mlp_fresh_component.set_original_component(
                                            original_mlp_subcomp
                                        )
                                        mlp_fresh_component.process_weights()
                                        mlp_weights = mlp_fresh_component.get_processed_state_dict()

                                        # Add MLP weights with proper prefixes
                                        for key, value in mlp_weights.items():
                                            tl_state_dict[
                                                f"blocks.{layer_idx}.mlp.{key}"
                                            ] = value.clone()
                                    except Exception as e:
                                        print(
                                            f"Warning: Failed to process MLP subcomponent {mlp_subcomp_name} for layer {layer_idx}: {e}"
                                        )
                            continue  # Skip the rest of the MLP processing
                        else:
                            # Unknown component type, use generic
                            fresh_component = component_class(name=subcomponent.name)

                        # Process the component
                        fresh_component.set_original_component(original_subcomponent)
                        fresh_component.process_weights()
                        comp_weights = fresh_component.get_processed_state_dict()

                        # Add weights with proper prefixes
                        for key, value in comp_weights.items():
                            tl_state_dict[
                                f"blocks.{layer_idx}.{subcomp_name}.{key}"
                            ] = value.clone()

                    except Exception as e:
                        print(
                            f"Warning: Failed to process subcomponent {subcomp_name} for layer {layer_idx}: {e}"
                        )

            except Exception as e:
                print(f"Warning: Failed to process layer {layer_idx}: {e}")

        return tl_state_dict

    def convert_hf_key_to_bridge_key(self, hf_key: str) -> str:
        """Convert a HuggingFace-style key to a bridge key with _original_component references.

        Args:
            hf_key: The HuggingFace-style key (e.g., "transformer.h.0.attn.c_attn.weight")

        Returns:
            The bridge key with _original_component references (e.g., "transformer.h.0._original_component.attn._original_component.c_attn._original_component.weight")
        """
        # Handle different key patterns
        if "transformer.h." in hf_key:
            parts = hf_key.split(".")
            if len(parts) >= 4 and parts[2].isdigit():
                layer = parts[2]

                # Pattern: transformer.h.X.attn.c_attn.weight -> transformer.h.X._original_component.attn._original_component.c_attn._original_component.weight
                if "attn.c_attn" in hf_key:
                    return f"transformer.h.{layer}._original_component.attn._original_component.c_attn._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.attn.c_proj.weight -> transformer.h.X._original_component.attn._original_component.c_proj._original_component.weight
                elif "attn.c_proj" in hf_key:
                    return f"transformer.h.{layer}._original_component.attn._original_component.c_proj._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.mlp.c_fc.weight -> transformer.h.X._original_component.mlp._original_component.c_fc._original_component.weight
                elif "mlp.c_fc" in hf_key:
                    return f"transformer.h.{layer}._original_component.mlp._original_component.c_fc._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.mlp.c_proj.weight -> transformer.h.X._original_component.mlp._original_component.c_proj._original_component.weight
                elif "mlp.c_proj" in hf_key:
                    return f"transformer.h.{layer}._original_component.mlp._original_component.c_proj._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.attn.qkv.weight -> transformer.h.X._original_component.attn.qkv._original_component.weight
                elif "attn.qkv" in hf_key:
                    return f"transformer.h.{layer}._original_component.attn.qkv._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.attn.o.weight -> transformer.h.X._original_component.attn.o._original_component.weight
                elif "attn.o" in hf_key:
                    return f"transformer.h.{layer}._original_component.attn.o._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.mlp.input.weight -> transformer.h.X._original_component.mlp.input._original_component.weight
                elif "mlp.input" in hf_key:
                    return f"transformer.h.{layer}._original_component.mlp.input._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.mlp.out.weight -> transformer.h.X._original_component.mlp.out._original_component.weight
                elif "mlp.out" in hf_key:
                    return f"transformer.h.{layer}._original_component.mlp.out._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.ln_1.weight -> transformer.h.X._original_component.ln_1._original_component.weight
                elif "ln_1" in hf_key:
                    return f"transformer.h.{layer}._original_component.ln_1._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.ln_2.weight -> transformer.h.X._original_component.ln_2._original_component.weight
                elif "ln_2" in hf_key:
                    return f"transformer.h.{layer}._original_component.ln_2._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.ln1.weight -> transformer.h.X._original_component.ln_1._original_component.weight (map ln1 to ln_1)
                elif "ln1" in hf_key:
                    return f"transformer.h.{layer}._original_component.ln_1._original_component.{parts[-1]}"

                # Pattern: transformer.h.X.ln2.weight -> transformer.h.X._original_component.ln_2._original_component.weight (map ln2 to ln_2)
                elif "ln2" in hf_key:
                    return f"transformer.h.{layer}._original_component.ln_2._original_component.{parts[-1]}"

        # Pattern: transformer.wte.weight -> transformer.wte._original_component.weight
        elif hf_key == "transformer.wte.weight":
            return "transformer.wte._original_component.weight"

        # Pattern: transformer.wpe.weight -> transformer.wpe._original_component.weight
        elif hf_key == "transformer.wpe.weight":
            return "transformer.wpe._original_component.weight"

        # Pattern: lm_head.weight -> lm_head._original_component.weight
        elif hf_key == "lm_head.weight":
            return "lm_head._original_component.weight"

        # Pattern: transformer.ln_f.bias -> transformer.ln_f._original_component.bias
        elif "transformer.ln_f" in hf_key:
            if "weight" in hf_key:
                return "transformer.ln_f._original_component.weight"
            elif "bias" in hf_key:
                return "transformer.ln_f._original_component.bias"

        # If no pattern matches, return the original key
        return hf_key
