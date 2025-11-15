"""MLP bridge component.

This module contains the bridge component for MLP layers.
"""
from typing import Any, Dict, Mapping, Optional

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
        "W_gate": "gate.weight",
        "b_gate": "gate.bias",
        "W_in": "in.weight",
        "b_in": "in.bias",
        "W_out": "out.weight",
        "b_out": "out.bias",
    }

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the MLP bridge.

        Args:
            name: The name of the component in the model (None if no container exists)
            config: Optional configuration (unused for MLPBridge)
            submodules: Dictionary of submodules to register (e.g., gate_proj, up_proj, down_proj)
        """
        super().__init__(name, config, submodules=submodules)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the MLP bridge.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output hidden states
        """
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            from transformer_lens.utilities.addmm import batch_addmm

            hidden_states = args[0]
            hidden_states = self.hook_in(hidden_states)
            in_module = getattr(self, "in", None) or getattr(self, "input", None)
            if in_module is not None and hasattr(in_module, "hook_in"):
                hidden_states = in_module.hook_in(hidden_states)  # type: ignore[misc]
            if hasattr(self, "_processed_W_in") and hasattr(self, "_processed_W_out"):
                b_in = (
                    self._processed_b_in
                    if self._processed_b_in is not None
                    else torch.zeros(
                        self._processed_W_in.shape[-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
                hidden = batch_addmm(b_in, self._processed_W_in, hidden_states)
                in_module = getattr(self, "in", None) or getattr(self, "input", None)
                if in_module and hasattr(in_module, "hook_out"):
                    hidden = in_module.hook_out(hidden)
                hidden = torch.nn.functional.gelu(hidden)
                if hasattr(self, "out") and hasattr(self.out, "hook_in"):
                    hidden = self.out.hook_in(hidden)
                b_out = (
                    self._processed_b_out
                    if self._processed_b_out is not None
                    else torch.zeros(
                        self._processed_W_out.shape[-1], device=hidden.device, dtype=hidden.dtype
                    )
                )
                output = batch_addmm(b_out, self._processed_W_out, hidden)
            else:
                new_args = (hidden_states,) + args[1:]
                output = self.original_component(*new_args, **kwargs)  # type: ignore[misc]
            output = self.hook_out(output)
            if hasattr(self, "out") and hasattr(self.out, "hook_out"):
                output = self.out.hook_out(output)
            return output
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
        output = self.hook_out(output)
        if hasattr(self, "out") and hasattr(self.out, "hook_out"):
            output = self.out.hook_out(output)
        return output

    def set_processed_weights(self, weights: Mapping[str, torch.Tensor | None]) -> None:
        """Set the processed weights for use in compatibility mode.

        This stores the processed weights as attributes on the MLP component so they can be
        used directly in the forward pass without modifying the original component.

        Args:
            W_in: The processed MLP input weight tensor [d_model, d_mlp]
            W_out: The processed MLP output weight tensor [d_mlp, d_model]
            b_in: The processed MLP input bias tensor (optional)
            b_out: The processed MLP output bias tensor (optional)
            W_gate: The processed MLP gate weight tensor [d_model, d_mlp] (for gated MLPs)
            b_gate: The processed MLP gate bias tensor (optional, for gated MLPs)
        """
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        W_in = weights.get("W_in")
        W_out = weights.get("W_out")
        if W_in is None or W_out is None:
            raise ValueError("Processed MLP weights must include 'W_in' and 'W_out' tensors.")
        b_in = weights.get("b_in")
        b_out = weights.get("b_out")
        self._use_processed_weights = True
        self._processed_W_in = W_in
        self._processed_b_in = b_in
        self._processed_W_out = W_out
        self._processed_b_out = b_out
        in_module = getattr(self, "in", None)
        out_module = getattr(self, "out", None)
        if in_module and hasattr(in_module, "set_processed_weights"):
            in_module.set_processed_weights({"weight": W_in, "bias": b_in})
        if out_module and hasattr(out_module, "set_processed_weights"):
            out_module.set_processed_weights({"weight": W_out, "bias": b_out})
