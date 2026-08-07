"""MLP bridge component.

This module contains the bridge component for MLP layers.
"""
from typing import Any, Dict, Optional

import torch
from transformers.pytorch_utils import Conv1D

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class MLPBridge(GeneralizedComponent):
    """Bridge component for MLP layers.

    This component wraps an MLP layer from a remote model and provides a consistent interface
    for accessing its weights and performing MLP operations.
    """

    hook_aliases = {"hook_pre": "in.hook_out", "hook_post": "out.hook_in"}
    # W_* are real properties below (layout-aware); only the 1-D biases are
    # orientation-free enough for raw passthrough aliases.
    property_aliases = {
        "b_gate": "gate.bias",
        "b_in": "in.bias",
        "b_out": "out.bias",
    }

    def _tl_oriented_weight(self, proj_name: str) -> Any:
        """Wrapped projection weight in the TL orientation ([d_model, d_mlp] for
        the input side; [d_mlp, d_model] for the output side).

        nn.Linear stores [out_features, in_features] (transpose of TL); Conv1D
        (GPT-2 style) stores [in_features, out_features] (TL as-is). Unknown
        wrapper types pass the raw weight through unchanged.
        """
        proj = getattr(self, proj_name, None)
        if proj is None:
            raise AttributeError(f"{type(self).__name__} has no '{proj_name}' projection")
        weight = proj.weight
        if weight.ndim != 2:
            return weight
        component = getattr(proj, "original_component", None)
        if isinstance(component, Conv1D):
            return weight
        if isinstance(component, torch.nn.Linear):
            return weight.T
        return weight

    @property
    def W_in(self) -> Any:
        """W_in in TL orientation [d_model, d_mlp]."""
        return self._tl_oriented_weight("in")

    @property
    def W_out(self) -> Any:
        """W_out in TL orientation [d_mlp, d_model]."""
        return self._tl_oriented_weight("out")

    @property
    def W_gate(self) -> Any:
        """W_gate in TL orientation [d_model, d_mlp]."""
        return self._tl_oriented_weight("gate")

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
