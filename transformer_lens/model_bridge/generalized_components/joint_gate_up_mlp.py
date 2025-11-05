"""MLP bridge component.

This module contains the bridge component for MLP layers with joint gating and up-projection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge


class JointGateUpMLPBridge(MLPBridge):
    """Bridge component for MLP layers with joint gating and up-projections.

    This component wraps an MLP layer with fused gate and up projections such that both the activations
    from the joint projection and the seperate gate and up projections are hooked and accessible.
    """

    # Override parent's hook_aliases to use gate.hook_out instead of in.hook_out/input.hook_out
    # Note: hook_post is not defined for JointGateUpMLPBridge as submodule structure varies
    hook_aliases = {
        "hook_pre": "gate.hook_out",
    }

    def __init__(
        self,
        name: str,
        model_config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
        gate_up_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the JointGateUpMLP bridge.

        Args:
            name: The name of the component in the model
            model_config: Optional configuration (unused for MLPBridge)
            submodules: Dictionary of submodules to register (e.g., gate_proj, up_proj, down_proj)
            gate_up_config: Gate_Up-specific configuration which holds function to split the joint projection into two
        """
        super().__init__(name, model_config, submodules=submodules)
        self.gate_up_config = gate_up_config or {}
        self.gate = LinearBridge(name="gate", config=model_config)
        self.up = LinearBridge(name="up", config=model_config)

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original MLP component and initialize LinearBridges for gate and up projections.

        Args:
            original_component: The original MLP component to wrap
        """
        super().set_original_component(original_component)

        Gate_projection, Up_projection = self.gate_up_config["split_gate_up_matrix"](
            original_component
        )

        # Initialize the LinearBridges for the seperated gate and up projections
        self.gate.set_original_component(Gate_projection)
        self.up.set_original_component(Up_projection)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the JointGateUpMLP bridge.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output hidden states
        """
        output = super().forward(*args, **kwargs)

        # Extract input tensor to run through gate and up projections
        # in order to hook their outputs
        input_tensor = (
            args[0] if len(args) > 0 else kwargs.get("input", kwargs.get("hidden_states"))
        )
        if input_tensor is not None:
            gated_output = self.gate(input_tensor)
            self.up(gated_output)

        return output
