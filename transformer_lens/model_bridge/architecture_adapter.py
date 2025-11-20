"""Architecture adapter base class.

This module contains the base class for architecture adapters that map between different model architectures.
"""
from typing import Any, Dict, cast

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
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
        self.weight_processing_conversions: Dict[str, ParamProcessingConversion | str] | None = None
        self.uses_split_attention: bool = getattr(cfg, "uses_split_attention", False)
        self._merge_default_config()

    def _merge_default_config(self) -> None:
        """Merge default_cfg into cfg for variables that don't exist in cfg."""
        for key, value in self.default_cfg.items():
            if not hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply architecture-specific weight transformations before ProcessWeights.

        This method allows architectures to apply custom transformations to weights
        before standard weight processing (fold_layer_norm, center_writing_weights, etc.).
        For example, Gemma models scale embeddings by sqrt(d_model).

        Args:
            state_dict: The state dictionary with HuggingFace format keys

        Returns:
            The modified state dictionary (default implementation returns unchanged)
        """
        return state_dict

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
            >>> # <TransformerBlock> # type: ignore[index]

            Get a layer norm component:

            >>> # adapter.get_remote_component(model, "model.layers.0.ln1")
            >>> # <LayerNorm>
        """
        current = model
        for part in path.split("."):
            # If current is a GeneralizedComponent bridge, unwrap to get the original HF component
            if (
                isinstance(current, GeneralizedComponent)
                and hasattr(current, "original_component")
                and current.original_component is not None
            ):
                current = current.original_component

            if part.isdigit():
                current = current[int(part)]  # type: ignore[index]
            else:
                current = getattr(current, part)
        return current

    def get_component_from_list_module(
        self, list_module: RemoteComponent, bridge_component: GeneralizedComponent, parts: list[str]
    ) -> RemoteComponent:
        """Get a component from a list module using the bridge component and the transformer lens path.
        Args:
            list_module: The remote list module to get the component from
            bridge_component: The bridge component
            parts: The parts of the transformer lens path to navigate
        Returns:
            The requested component from the list module described by the path
        """
        item_index = parts[1]
        if not item_index.isdigit():
            raise ValueError(f"Expected item index, got {item_index}")
        if not hasattr(list_module, "__getitem__"):
            raise TypeError(f"Component {bridge_component.name} is not indexable")
        indexable_container = cast(Any, list_module)
        item = indexable_container[int(item_index)]
        if len(parts) == 2:
            return item
        else:
            subcomponent_name = parts[2]
            if subcomponent_name in bridge_component.submodules:
                subcomponent_bridge = bridge_component.submodules[subcomponent_name]
                if len(parts) > 3:
                    current_bridge = subcomponent_bridge
                    if subcomponent_bridge.name is None:
                        current = item
                    else:
                        current = self.get_remote_component(item, subcomponent_bridge.name)
                    for i in range(3, len(parts)):
                        deeper_component_name = parts[i]
                        if deeper_component_name.isdigit() and current_bridge.is_list_item:
                            return self.get_component_from_list_module(
                                current, current_bridge, parts[i - 1 :]
                            )
                        if deeper_component_name in current_bridge.submodules:
                            current_bridge = current_bridge.submodules[deeper_component_name]
                            if current_bridge.name is None:
                                pass
                            else:
                                current = self.get_remote_component(current, current_bridge.name)
                        else:
                            raise ValueError(
                                f"Component {deeper_component_name} not found in {'.'.join(parts[:i])} components"
                            )
                    return current
                elif subcomponent_bridge.name is None:
                    return item
                else:
                    return self.get_remote_component(item, subcomponent_bridge.name)
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
        component_path, _ = self._preprocess_parameter_path(path)
        parts = component_path.split(".")
        if not parts:
            raise ValueError("Empty path")
        if parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")
        bridge_component = self.component_mapping[parts[0]]
        if len(parts) == 1:
            return bridge_component
        current_component = bridge_component
        for i in range(1, len(parts)):
            part = parts[i]
            if part.isdigit():
                continue
            if hasattr(current_component, "submodules") and part in current_component.submodules:
                current_component = current_component.submodules[part]
            elif (
                hasattr(current_component, "__class__")
                and "AttentionBridge" in current_component.__class__.__name__
                and (part in ["q", "k", "v", "o"])
            ):
                if "JointQKV" in current_component.__class__.__name__:
                    continue
                elif (
                    hasattr(current_component, "submodules")
                    and part in current_component.submodules
                ):
                    current_component = current_component.submodules[part]
                    continue
            elif (
                hasattr(current_component, "__class__")
                and "MLPBridge" in current_component.__class__.__name__
                and (part in ["in", "out", "gate"])
            ):
                if (
                    hasattr(current_component, "submodules")
                    and part in current_component.submodules
                ):
                    current_component = current_component.submodules[part]
                    continue
                else:
                    continue
            else:
                raise ValueError(f"Component {part} not found in {'.'.join(parts[:i])} components")
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
        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")
        if self.component_mapping is None or parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")
        bridge_component = self.component_mapping[parts[0]]
        if len(parts) == 1:
            if bridge_component.name is None:
                return model
            return self.get_remote_component(model, bridge_component.name)
        if bridge_component.is_list_item and len(parts) >= 2:
            if bridge_component.name is None:
                raise ValueError(f"List component {parts[0]} must have a name")
            list_module = self.get_remote_component(model, bridge_component.name)
            return self.get_component_from_list_module(list_module, bridge_component, parts)
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
        path, param_suffix = self._preprocess_parameter_path(path)
        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")
        if parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")
        bridge_component = self.component_mapping[parts[0]]
        if len(parts) == 1:
            remote_path = bridge_component.name
            if remote_path is None:
                raise ValueError(f"Component {parts[0]} must have a name for path translation")
            if param_suffix:
                remote_path = remote_path + param_suffix
            if last_component_only:
                return remote_path.split(".")[-1]
            return remote_path
        if bridge_component.is_list_item and len(parts) >= 2:
            item_index = parts[1]
            if not item_index.isdigit():
                raise ValueError(f"Expected item index, got {item_index}")
            items_path = bridge_component.name
            if items_path is None:
                raise ValueError(f"List component {parts[0]} must have a name for path translation")
            if len(parts) == 2:
                remote_path = f"{items_path}.{item_index}"
                if param_suffix:
                    remote_path = remote_path + param_suffix
                if last_component_only:
                    return remote_path.split(".")[-1]
                return remote_path
            else:
                subcomponent_name = parts[2]
                if subcomponent_name in bridge_component.submodules:
                    subcomponent_bridge = bridge_component.submodules[subcomponent_name]
                    if len(parts) > 3:
                        current_bridge = subcomponent_bridge
                        subcomponent_name_str = subcomponent_bridge.name
                        if subcomponent_name_str is None:
                            raise ValueError(
                                f"Subcomponent {subcomponent_name} must have a name for path translation"
                            )
                        remote_path_parts = [items_path, item_index, subcomponent_name_str]
                        for i in range(3, len(parts)):
                            deeper_component_name = parts[i]
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
                        if param_suffix:
                            remote_path = remote_path + param_suffix
                        if last_component_only:
                            return remote_path.split(".")[-1]
                        return remote_path
                    else:
                        subcomponent_name_str = subcomponent_bridge.name
                        if subcomponent_name_str is None:
                            raise ValueError(
                                f"Subcomponent {subcomponent_name} must have a name for path translation"  # type: ignore[assignment]
                            )
                        remote_path = f"{items_path}.{item_index}.{subcomponent_name_str}"
                        if param_suffix:
                            remote_path = remote_path + param_suffix
                        if last_component_only:
                            return remote_path.split(".")[-1]
                        return remote_path
                else:
                    raise ValueError(
                        f"Component {subcomponent_name} not found in {parts[0]} components"
                    )
        remote_path = bridge_component.name
        if remote_path is None:
            raise ValueError(f"Component {parts[0]} must have a name for path translation")
        if len(parts) > 1:
            remote_path = f"{remote_path}.{'.'.join(parts[1:])}"
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
        param_suffix = ""
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
                ".b_K",  # type: ignore[assignment]
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
        if any(
            (
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
            )
        ):
            attn_path_parts = path.split(".")
            if len(attn_path_parts) >= 3 and attn_path_parts[-2] == "attn":
                attn_component_path = ".".join(attn_path_parts[:-1])
                try:
                    if self.component_mapping:
                        current_mapping = self.component_mapping
                        for part in attn_component_path.split("."):
                            if (
                                hasattr(current_mapping, "submodules")
                                and part in current_mapping.submodules
                            ):
                                current_mapping = current_mapping.submodules[part]
                            elif hasattr(current_mapping, "__getitem__"):
                                current_mapping = current_mapping[part]  # type: ignore[assignment]
                        if hasattr(current_mapping, "submodules"):
                            attn_components = list(current_mapping.submodules.keys())
                            path = path.replace(".W_Q", ".q")
                            path = path.replace(".W_K", ".k")
                            path = path.replace(".W_V", ".v")
                            path = path.replace(".b_Q", ".q")
                            path = path.replace(".b_K", ".k")
                            path = path.replace(".b_V", ".v")
                            path = path.replace("._W_K", ".k")
                            path = path.replace("._W_V", ".v")
                            path = path.replace("._b_K", ".k")
                            path = path.replace("._b_V", ".v")
                except Exception:
                    pass
        if any(
            (path.endswith(suffix) for suffix in [".W_Q", ".W_K", ".W_V", ".b_Q", ".b_K", ".b_V"])
        ):
            path = path.replace(".W_Q", ".q")
            path = path.replace(".W_K", ".k")
            path = path.replace(".W_V", ".v")
            path = path.replace(".b_Q", ".q")
            path = path.replace(".b_K", ".k")
            path = path.replace(".b_V", ".v")
        path = path.replace(".W_O", ".o")
        path = path.replace(".b_O", ".o")
        if any(
            (
                path.endswith(suffix)
                for suffix in [".W_in", ".W_out", ".b_in", ".b_out", ".ln.w", ".ln.b"]
            )
        ):
            mlp_path_parts = path.split(".")
            if len(mlp_path_parts) >= 3 and mlp_path_parts[-2] == "mlp":
                mlp_component_path = ".".join(mlp_path_parts[:-1])
                try:
                    if self.component_mapping:
                        current_mapping = self.component_mapping
                        for part in mlp_component_path.split("."):
                            if (
                                hasattr(current_mapping, "submodules")
                                and part in current_mapping.submodules
                            ):
                                current_mapping = current_mapping.submodules[part]
                            elif hasattr(current_mapping, "__getitem__"):
                                current_mapping = current_mapping[part]  # type: ignore[assignment]
                        if hasattr(current_mapping, "submodules"):
                            mlp_components = list(current_mapping.submodules.keys())
                            if "input" in mlp_components and "out" in mlp_components:
                                path = path.replace(".W_in", ".input")
                                path = path.replace(".b_in", ".input")
                                path = path.replace(".W_out", ".out")
                                path = path.replace(".b_out", ".out")
                            elif "in" in mlp_components and "out" in mlp_components:
                                path = path.replace(".W_in", ".in")
                                path = path.replace(".b_in", ".in")
                                path = path.replace(".W_out", ".out")
                                path = path.replace(".b_out", ".out")
                            elif "fc_in" in mlp_components and "fc_out" in mlp_components:
                                path = path.replace(".W_in", ".fc_in")
                                path = path.replace(".b_in", ".fc_in")
                                path = path.replace(".W_out", ".fc_out")
                                path = path.replace(".b_out", ".fc_out")
                            if "ln" in mlp_components:
                                path = path.replace(".ln.w", ".ln")
                                path = path.replace(".ln.b", ".ln")
                except Exception:
                    pass
        if any((path.endswith(suffix) for suffix in [".W_in", ".W_out", ".b_in", ".b_out"])):
            path = path.replace(".W_in", ".in")
            path = path.replace(".b_in", ".in")
            path = path.replace(".W_out", ".out")
            path = path.replace(".b_out", ".out")
        path = path.replace(".W_gate", ".gate")
        path = path.replace(".b_gate", ".gate")
        if not (path.endswith(".weight") or path.endswith(".bias")):
            path = path.replace(".W_E", "")
            path = path.replace(".b_E", "")
            path = path.replace(".W_U", "")
            path = path.replace(".b_U", "")
            path = path.replace(".W_pos", "")
            path = path.replace(".b_pos", "")
            path = path.replace(".w", "")
            path = path.replace(".b", "")
        return (path, param_suffix)

    def convert_hf_key_to_tl_key(self, hf_key: str) -> str:
        """Convert a HuggingFace-style key to TransformerLens format key using component mapping.

        The component mapping keys ARE the TL format names (e.g., "embed", "pos_embed", "blocks").
        The component.name is the HF path (e.g., "transformer.wte").

        Args:
            hf_key: The HuggingFace-style key (e.g., "transformer.wte.weight")

        Returns:
            The TransformerLens format key (e.g., "embed.weight")
        """
        if self.component_mapping is None:
            return hf_key
        for tl_name, component in self.component_mapping.items():
            if tl_name == "blocks":
                continue
            hf_path = component.name
            if hf_path is not None and hf_key.startswith(hf_path + "."):
                param = hf_key[len(hf_path) + 1 :]
                return f"{tl_name}.{param}"
        blocks_component = self.component_mapping.get("blocks")
        if blocks_component:
            hf_blocks_prefix = blocks_component.name
            if hf_blocks_prefix is not None and hf_key.startswith(hf_blocks_prefix + "."):
                rest = hf_key[len(hf_blocks_prefix) + 1 :]
                parts = rest.split(".", 1)
                if len(parts) >= 2 and parts[0].isdigit():
                    layer_idx = parts[0]
                    subkey = parts[1]
                    if hasattr(blocks_component, "submodules"):
                        for tl_subname, subcomponent in blocks_component.submodules.items():
                            hf_subpath = subcomponent.name
                            if hf_subpath is not None and subkey.startswith(hf_subpath + "."):
                                param = subkey[len(hf_subpath) + 1 :]
                                return f"blocks.{layer_idx}.{tl_subname}.{param}"
                            if hasattr(subcomponent, "submodules"):
                                for tl_nested_name, nested_comp in subcomponent.submodules.items():
                                    hf_nested_path = f"{hf_subpath}.{nested_comp.name}"
                                    if subkey.startswith(hf_nested_path + "."):
                                        param = subkey[len(hf_nested_path) + 1 :]
                                        return f"blocks.{layer_idx}.{tl_subname}.{tl_nested_name}.{param}"
        return hf_key

    def setup_component_testing(self, hf_model: RemoteModel, bridge_model: Any = None) -> None:
        """Set up model-specific references needed for component testing.

        This hook is called after the adapter is created and has access to the HF model.
        Subclasses can override this to configure bridges with model-specific components
        (e.g., rotary embeddings, normalization parameters) needed for get_random_inputs().

        Args:
            hf_model: The HuggingFace model instance
            bridge_model: Optional TransformerBridge model instance (for configuring actual bridges)

        Note:
            This is a no-op in the base class. Override in subclasses as needed.
        """
        pass

    def _enable_ht_attention(self, attn_bridge, hf_attn):
        """Enable HT computation for attention (architecture-agnostic).

        Detects the architecture by checking which weight attributes exist.
        """
        n_heads = getattr(
            self.cfg,
            "n_heads",
            getattr(self.cfg, "n_head", getattr(self.cfg, "num_attention_heads", None)),
        )
        d_model = getattr(
            self.cfg, "d_model", getattr(self.cfg, "n_embd", getattr(self.cfg, "hidden_size", None))
        )
        if n_heads is None or d_model is None:
            raise RuntimeError(f"Could not determine n_heads or d_model from config: {self.cfg}")
        d_head = d_model // n_heads
        if hasattr(hf_attn, "c_attn"):
            W_Q, W_K, W_V, b_Q, b_K, b_V = self._extract_qkv_gpt2_style(
                hf_attn.c_attn, n_heads, d_model, d_head
            )
            W_O, b_O = self._extract_output_proj(hf_attn.c_proj, n_heads, d_head, d_model)
        elif (
            hasattr(hf_attn, "q_proj") and hasattr(hf_attn, "k_proj") and hasattr(hf_attn, "v_proj")
        ):
            W_Q, b_Q = self._extract_linear_ht_format(hf_attn.q_proj, n_heads, d_head, d_model)  # type: ignore[attr-defined]
            W_K, b_K = self._extract_linear_ht_format(hf_attn.k_proj, n_heads, d_head, d_model)  # type: ignore[attr-defined]
            W_V, b_V = self._extract_linear_ht_format(hf_attn.v_proj, n_heads, d_head, d_model)  # type: ignore[attr-defined]
            out_proj = hf_attn.out_proj if hasattr(hf_attn, "out_proj") else hf_attn.o_proj
            W_O, b_O = self._extract_output_proj(out_proj, n_heads, d_head, d_model)
        elif hasattr(hf_attn, "query_key_value"):
            W_Q, W_K, W_V, b_Q, b_K, b_V = self._extract_qkv_neox_style(  # type: ignore[attr-defined]
                hf_attn.query_key_value, n_heads, d_model, d_head
            )
            W_O, b_O = self._extract_output_proj(hf_attn.dense, n_heads, d_head, d_model)
        else:
            raise ValueError(
                f"Unsupported attention architecture. Module has attributes: {dir(hf_attn)}"
            )
        attn_bridge.set_processed_weights(
            {
                "W_Q": W_Q,
                "W_K": W_K,
                "W_V": W_V,
                "W_O": W_O,
                "b_Q": b_Q,
                "b_K": b_K,
                "b_V": b_V,
                "b_O": b_O,
            }
        )
        self._disable_hook_conversions(attn_bridge)

    def _extract_qkv_gpt2_style(self, c_attn, n_heads, d_model, d_head):
        """Extract Q, K, V weights from GPT-2 style combined c_attn.

        GPT-2 uses Conv1D which stores weights as [in_features, out_features] = [d_model, 3*d_model].
        We need to split and reshape to [n_heads, d_model, d_head] format for HookedTransformer.
        """
        import einops

        W = c_attn.weight.data
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=n_heads)
        qkv_bias = c_attn.bias.data
        qkv_bias = einops.rearrange(
            qkv_bias, "(qkv index head)->qkv index head", qkv=3, index=n_heads, head=d_head
        )
        b_Q = qkv_bias[0]
        b_K = qkv_bias[1]
        b_V = qkv_bias[2]
        return (W_Q, W_K, W_V, b_Q, b_K, b_V)

    def _extract_output_proj(self, out_proj, n_heads, d_head, d_model):
        """Extract output projection weights in HT format.

        Returns W_O in [n_heads, d_head, d_model] format for HookedTransformer compatibility.

        For Conv1D (GPT-2), weight is stored as [d_model, d_model] = [nx, nf].
        For Linear, weight is stored as [d_model, d_model] = [out_features, in_features].
        """
        weight = out_proj.weight.data
        bias = out_proj.bias.data if hasattr(out_proj, "bias") else None
        W_O = weight.view(n_heads, d_head, d_model).contiguous()
        b_O = bias.contiguous() if bias is not None else None
        return (W_O, b_O)

    def _disable_hook_conversions(self, attn_bridge):
        """Disable hook conversions for attention submodules.

        Note: In no_processing mode, we DON'T disable conversions because Q/K/V hooks need
        to convert from 3D [batch, seq, d_model] to 4D [batch, seq, n_heads, d_head].
        We also preserve o.hook_in.hook_conversion (hook_z).

        This method is kept for potential future use but currently does nothing in no_processing mode.
        """
        pass
