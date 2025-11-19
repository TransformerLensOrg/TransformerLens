"""Gated MLP bridge component.

This module contains the bridge component for gated MLP layers (e.g., LLaMA, Gemma).
"""
from typing import Any, Dict, Mapping, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge


class GatedMLPBridge(MLPBridge):
    """Bridge component for gated MLP layers.

    This component wraps a gated MLP layer from a remote model (e.g., LLaMA, Gemma)
    and provides a consistent interface for accessing its weights and performing MLP operations.

    Gated MLPs have the structure:
    output = down_proj(act_fn(gate_proj(x)) * up_proj(x))

    Where:
    - gate_proj: The gating projection (produces the activation to be gated)
    - up_proj (in): The input projection (produces the linear component)
    - down_proj (out): The output projection
    """

    hook_aliases = {
        "hook_pre": "gate.hook_out",
        "hook_pre_linear": "in.hook_out",
        "hook_post": "out.hook_in",
    }
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
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the gated MLP bridge.

        Args:
            name: The name of the component in the model (None if no container exists)
            config: Optional configuration (unused for GatedMLPBridge)
            submodules: Dictionary of submodules to register (e.g., gate_proj, up_proj, down_proj)
        """
        super().__init__(name, config, submodules=submodules or {})

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the gated MLP bridge.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output hidden states
        """
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            hidden_states = args[0]
            hidden_states = self.hook_in(hidden_states)
            if hasattr(self, "_processed_W_gate") and hasattr(self, "_processed_W_in"):
                gate_output = torch.nn.functional.linear(
                    hidden_states, self._processed_W_gate, self._processed_b_gate
                )
                if hasattr(self, "gate") and hasattr(self.gate, "hook_out"):
                    gate_output = self.gate.hook_out(gate_output)
                linear_output = torch.nn.functional.linear(
                    hidden_states, self._processed_W_in, self._processed_b_in
                )
                in_module = getattr(self, "in", None)
                if in_module is not None and hasattr(in_module, "hook_out"):
                    linear_output = in_module.hook_out(linear_output)  # type: ignore[misc]
                act_fn_name = None
                if self.config:
                    act_fn_name = getattr(self.config, "activation_function", None)
                    if act_fn_name is None:
                        act_fn_name = getattr(self.config, "hidden_activation", None)
                    if act_fn_name is None:
                        act_fn_name = getattr(self.config, "hidden_act", None)
                    if act_fn_name is None:
                        act_fn_name = getattr(self.config, "act_fn", None)
                if act_fn_name is None:
                    act_fn_name = "silu"
                if act_fn_name in ("silu", "swish"):
                    activated = torch.nn.functional.silu(gate_output)
                elif act_fn_name == "gelu":
                    activated = torch.nn.functional.gelu(gate_output)
                elif act_fn_name == "gelu_new" or act_fn_name == "gelu_pytorch_tanh":
                    activated = torch.nn.functional.gelu(gate_output, approximate="tanh")
                elif act_fn_name == "relu":
                    activated = torch.nn.functional.relu(gate_output)
                else:
                    activated = torch.nn.functional.silu(gate_output)
                hidden = activated * linear_output
                if hasattr(self, "out") and hasattr(self.out, "hook_in"):
                    hidden = self.out.hook_in(hidden)
                output = torch.nn.functional.linear(
                    hidden, self._processed_W_out, self._processed_b_out
                )
            else:
                new_args = (hidden_states,) + args[1:]
                output = self.original_component(*new_args, **kwargs)  # type: ignore[misc]
            output = self.hook_out(output)
            return output
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        hidden_states = args[0]
        hidden_states = self.hook_in(hidden_states)
        new_args = (hidden_states,) + args[1:]
        output = self.original_component(*new_args, **kwargs)
        output = self.hook_out(output)
        return output

    def set_processed_weights(
        self, weights: Mapping[str, torch.Tensor | None], verbose: bool = False
    ) -> None:
        """Set the processed weights to use when layer norm is folded.

        Args:
            W_gate: The processed MLP gate weight tensor
            W_in: The processed MLP input weight tensor
            W_out: The processed MLP output weight tensor
            b_gate: The processed MLP gate bias tensor (optional)
            b_in: The processed MLP input bias tensor (optional)
            b_out: The processed MLP output bias tensor (optional)
            verbose: If True, print detailed information about weight setting
        """
        if verbose:
            print(
                f"\n  set_processed_weights: GatedMLPBridge (name={getattr(self, 'name', 'unknown')})"
            )
            print(f"    Received {len(weights)} weight keys")

        super().set_processed_weights(weights, verbose=verbose)  # type: ignore[arg-type]
        W_gate = weights.get("gate.weight")
        if W_gate is None:
            return
        b_gate = weights.get("gate.bias")

        if verbose:
            print(f"    Setting W_gate with shape: {W_gate.shape}")
            if b_gate is not None:
                print(f"    Setting b_gate with shape: {b_gate.shape}")

        gate_module = getattr(self, "gate", None)
        self._use_processed_weights = True
        self._processed_W_gate = W_gate
        self._processed_b_gate = b_gate
        if gate_module and hasattr(gate_module, "set_processed_weights"):
            gate_weights: Dict[str, torch.Tensor] = {"weight": W_gate}
            if b_gate is not None:
                gate_weights["bias"] = b_gate
            gate_module.set_processed_weights(gate_weights, verbose=verbose)
