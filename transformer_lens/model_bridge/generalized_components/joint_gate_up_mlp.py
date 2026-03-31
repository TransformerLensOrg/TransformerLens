"""Bridge component for MLP layers with fused gate+up projections (e.g., Phi-3)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.gated_mlp import (
    GatedMLPBridge,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class JointGateUpMLPBridge(GatedMLPBridge):
    """Bridge for MLPs with fused gate+up projections (e.g., Phi-3's gate_up_proj).

    Splits the fused projection into separate LinearBridges and reconstructs
    the gated MLP forward pass, allowing individual hook access to gate and up
    activations. Follows the same pattern as JointQKVAttentionBridge for fused QKV.

    Hook interface matches GatedMLPBridge: hook_pre (gate), hook_pre_linear (up),
    hook_post (before down_proj).
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        split_gate_up_matrix: Optional[Callable] = None,
    ):
        super().__init__(name, config, submodules=submodules)
        self.split_gate_up_matrix = (
            split_gate_up_matrix
            if split_gate_up_matrix is not None
            else self._default_split_gate_up
        )

        # Up projection registered as "in" to match GatedMLPBridge convention
        # (hook_aliases, property_aliases, and weight keys all use "in").
        self.gate = LinearBridge(name="gate")
        _up_bridge = LinearBridge(name="in")
        setattr(self, "in", _up_bridge)  # "in" is a keyword; use setattr

        self.submodules["gate"] = self.gate
        self.submodules["in"] = _up_bridge

        self.real_components["gate"] = ("gate", self.gate)
        self.real_components["in"] = ("in", _up_bridge)
        if hasattr(self, "out"):
            self.real_components["out"] = ("out", self.out)

        # Typed as Any: HF exposes activation_fn as nn.Module (e.g. nn.SiLU)
        self._activation_fn: Any = None

        self._register_state_dict_hook(JointGateUpMLPBridge._filter_gate_up_state_dict)

    @staticmethod
    def _filter_gate_up_state_dict(
        module: torch.nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
    ) -> None:
        """State dict hook that removes stale combined gate_up entries."""
        gate_up_prefix = prefix + "gate_up."
        keys_to_remove = [k for k in state_dict if k.startswith(gate_up_prefix)]
        for k in keys_to_remove:
            del state_dict[k]

    @staticmethod
    def _default_split_gate_up(
        original_mlp_component: Any,
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Split gate_up_proj [2*d_mlp, d_model] into (gate, up) nn.Linear modules."""
        fused_weight = original_mlp_component.gate_up_proj.weight
        gate_w, up_w = torch.tensor_split(fused_weight, 2, dim=0)
        d_model = fused_weight.shape[1]
        d_mlp = gate_w.shape[0]

        has_bias = (
            hasattr(original_mlp_component.gate_up_proj, "bias")
            and original_mlp_component.gate_up_proj.bias is not None
        )
        gate_b: torch.Tensor | None = None
        up_b: torch.Tensor | None = None
        if has_bias:
            gate_b, up_b = torch.tensor_split(original_mlp_component.gate_up_proj.bias, 2, dim=0)

        gate_proj = torch.nn.Linear(d_model, d_mlp, bias=has_bias)
        gate_proj.weight = torch.nn.Parameter(gate_w)
        if gate_b is not None:
            gate_proj.bias = torch.nn.Parameter(gate_b)

        up_proj = torch.nn.Linear(d_model, d_mlp, bias=has_bias)
        up_proj.weight = torch.nn.Parameter(up_w)
        if up_b is not None:
            up_proj.bias = torch.nn.Parameter(up_b)

        return gate_proj, up_proj

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original MLP component and split fused projections."""
        super().set_original_component(original_component)

        gate_proj, up_proj = self.split_gate_up_matrix(original_component)
        self.gate.set_original_component(gate_proj)
        getattr(self, "in").set_original_component(up_proj)

        # Capture activation function from original component
        if hasattr(original_component, "activation_fn"):
            self._activation_fn = original_component.activation_fn
        elif hasattr(original_component, "act_fn"):
            self._activation_fn = original_component.act_fn

    def _resolve_activation_fn(self) -> Callable:
        """Resolve the activation function for the reconstructed forward pass."""
        if self._activation_fn is not None:
            return self._activation_fn

        # Config-based fallback (same logic as GatedMLPBridge)
        act_fn_name = None
        if self.config:
            act_fn_name = getattr(self.config, "activation_function", None)
            if act_fn_name is None:
                act_fn_name = getattr(self.config, "hidden_activation", None)
            if act_fn_name is None:
                act_fn_name = getattr(self.config, "hidden_act", None)
            if act_fn_name is None:
                act_fn_name = getattr(self.config, "act_fn", None)

        if act_fn_name is None or act_fn_name in ("silu", "swish"):
            return torch.nn.functional.silu
        elif act_fn_name == "gelu":
            return torch.nn.functional.gelu
        elif act_fn_name in ("gelu_new", "gelu_pytorch_tanh"):

            def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.gelu(x, approximate="tanh")

            return gelu_tanh
        elif act_fn_name == "relu":
            return torch.nn.functional.relu
        else:
            return torch.nn.functional.silu

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Reconstructed gated MLP forward with individual hook access."""
        # Delegate to GatedMLPBridge's processed-weights path only when ALL
        # processed weights exist; its fallback bypasses intermediate hooks.
        if (
            hasattr(self, "_use_processed_weights")
            and self._use_processed_weights
            and hasattr(self, "_processed_W_gate")
            and hasattr(self, "_processed_W_in")
        ):
            return super().forward(*args, **kwargs)

        hidden_states = self.hook_in(args[0])

        gate_output = self.gate(hidden_states)
        up_output = getattr(self, "in")(hidden_states)

        act_fn = self._resolve_activation_fn()
        gated = act_fn(gate_output) * up_output

        if hasattr(self, "out") and self.out is not None:
            output = self.out(gated)
        else:
            raise RuntimeError(
                f"No 'out' (down_proj) submodule found in {self.__class__.__name__}. "
                "Ensure 'out' is provided in submodules."
            )

        return self.hook_out(output)
