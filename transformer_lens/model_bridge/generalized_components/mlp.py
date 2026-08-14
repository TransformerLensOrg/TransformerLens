"""MLP bridge component.

This module contains the bridge component for MLP layers.
"""
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class MLPBridge(GeneralizedComponent):
    """Bridge component for MLP layers.

    This component wraps an MLP layer from a remote model and provides a consistent interface
    for accessing its weights and performing MLP operations.
    """

    hook_aliases = {"hook_pre": "in.hook_out", "hook_post": "out.hook_in"}
    property_aliases = {
        "b_gate": "gate.bias",
        "b_in": "in.bias",
        "b_out": "out.bias",
    }

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
        optional: bool = False,
    ):
        """Initialize the MLP bridge.

        Args:
            name: The name of the component in the model (None if no container exists)
            config: Optional configuration (unused for MLPBridge)
            submodules: Dictionary of submodules to register (e.g., gate_proj, up_proj, down_proj)
            optional: If True, setup skips this bridge when absent (hybrid architectures).
        """
        super().__init__(name, config, submodules=submodules, optional=optional)

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the MLP bridge.

        Returns a tensor, or the component's own (hidden, ...) tuple re-packed
        with hooked hidden states for recurrent MLPs.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output hidden states
        """
        hidden_states = args[0]
        hidden_states = self.hook_in(hidden_states)
        in_module = getattr(self, "in", None) or getattr(self, "input", None)
        if in_module is not None and hasattr(in_module, "hook_in"):
            hidden_states = in_module.hook_in(hidden_states)  # type: ignore[misc]
        new_args = (hidden_states,) + args[1:]
        original_component = self.original_component
        if original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        output = original_component(*new_args, **kwargs)
        # Recurrent MLPs (RWKV's channel-mix) return (hidden, state). Hook the
        # hidden states and re-pack, or hook_out would hand users a tuple and
        # interventions on it would be silently dropped.
        if isinstance(output, tuple):
            # Only the block-level hook: the wrapped `out` projection already
            # fired its own hook_out inside the forward, and for gated recurrent
            # MLPs (RWKV channel-mix) output[0] is the post-gate product — a
            # different tensor, so re-firing would double-apply interventions.
            return (self.hook_out(output[0]),) + output[1:]
        output = self.hook_out(output)
        if hasattr(self, "out") and hasattr(self.out, "hook_out"):
            output = self.out.hook_out(output)
        return output

    def _weight_layout_in_out(self, proj: Any) -> Optional[bool]:
        """Whether proj's wrapped module stores its weight as [in, out].

        Conv1D (GPT-2 style) stores [in_features, out_features]; nn.Linear stores
        [out_features, in_features]. Returns None when the wrapped module is
        neither, so callers can fall back to in_features/out_features or shape heuristic.
        """
        from transformers.pytorch_utils import Conv1D

        component = getattr(proj, "original_component", None)
        if isinstance(component, Conv1D):
            return True
        if isinstance(component, torch.nn.Linear):
            return False
        return None

    def _normalize_mlp_weight(
        self, weight: torch.Tensor, layout: Optional[bool], proj: Any, pattern: str = "in"
    ) -> torch.Tensor:
        """Normalize MLP weight to TL orientation.

        Args:
            weight: 2D weight tensor from the projection
            layout: True if [in, out] (Conv1D), False if [out, in] (nn.Linear),
                None falls back to in_features/out_features or shape heuristic.
            proj: The projection module (LinearBridge) for metadata fallback.
            pattern: "in" for W_in/W_gate [d_model, d_mlp], "out" for W_out [d_mlp, d_model]

        Note: Shape heuristic assumes d_model < d_mlp, which fails for architectures
        like GIDD's ScaledLinear where d_mlp < d_model. When layout is None, we first
        check in_features/out_features; shape heuristic is last resort.
        """
        if layout is None:
            # Try in_features/out_features from the wrapped module (works for bare nn.Module)
            component = getattr(proj, "original_component", None)
            in_f = getattr(component, "in_features", None)
            out_f = getattr(component, "out_features", None)
            if in_f is not None and out_f is not None:
                # Module declares its orientation; weight[0] == out_f means [out, in] layout
                layout = weight.shape[0] == in_f  # True if [in, out], False if [out, in]
            else:
                # Last resort: shape heuristic. WARNING: assumes d_model < d_mlp.
                # This will mis-orient architectures where d_mlp < d_model.
                if pattern == "in":
                    layout = weight.shape[0] < weight.shape[1]
                else:
                    layout = weight.shape[0] > weight.shape[1]
        if layout:
            return weight  # Conv1D-style: already in TL orientation
        return weight.T  # nn.Linear-style: transpose to TL orientation

    @property
    def W_in(self) -> torch.Tensor:
        """MLP input weight in TL orientation [d_model, d_mlp]."""
        in_module = getattr(self, "in", None)
        if in_module is None:
            raise AttributeError("No 'in' submodule on this MLP bridge")
        weight = in_module.weight
        layout = self._weight_layout_in_out(in_module)
        return self._normalize_mlp_weight(weight, layout, in_module, pattern="in")

    @property
    def W_gate(self) -> Optional[torch.Tensor]:
        """MLP gate weight in TL orientation [d_model, d_mlp], or None if ungated."""
        gate_module = getattr(self, "gate", None)
        if gate_module is None:
            return None
        weight = gate_module.weight
        layout = self._weight_layout_in_out(gate_module)
        return self._normalize_mlp_weight(weight, layout, gate_module, pattern="in")

    @property
    def W_out(self) -> torch.Tensor:
        """MLP output weight in TL orientation [d_mlp, d_model]."""
        out_module = getattr(self, "out", None)
        if out_module is None:
            raise AttributeError("No 'out' submodule on this MLP bridge")
        weight = out_module.weight
        layout = self._weight_layout_in_out(out_module)
        return self._normalize_mlp_weight(weight, layout, out_module, pattern="out")
