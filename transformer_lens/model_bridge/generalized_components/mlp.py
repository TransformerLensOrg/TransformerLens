"""MLP bridge component.

This module contains the bridge component for MLP layers.
"""
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.utilities.quantization import require_readable_weight


def weight_layout_in_out(proj: Any) -> Optional[bool]:
    """Whether proj's wrapped module stores its weight as [in, out].

    Conv1D (GPT-2 style) stores [in_features, out_features]; nn.Linear stores
    [out_features, in_features]. Returns None when the wrapped module is
    neither, so callers fall back to in_features/out_features or a shape heuristic.
    """
    from transformers.pytorch_utils import Conv1D

    component = getattr(proj, "original_component", None)
    if isinstance(component, Conv1D):
        return True
    if isinstance(component, torch.nn.Linear):
        return False
    return None


def normalize_mlp_weight(
    weight: torch.Tensor, layout: Optional[bool], proj: Any, pattern: str = "in"
) -> torch.Tensor:
    """Normalize an MLP projection weight to TL orientation ([d_model, d_mlp]
    for "in"/W_gate, [d_mlp, d_model] for "out")."""
    if layout is None:
        component = getattr(proj, "original_component", None)
        in_f = getattr(component, "in_features", None)
        out_f = getattr(component, "out_features", None)
        if in_f is not None and out_f is not None:
            layout = weight.shape[0] == in_f
        else:
            # Last resort. WARNING: assumes d_model < d_mlp (false for GIDD's
            # ScaledLinear, which is why in_features/out_features go first).
            if pattern == "in":
                layout = weight.shape[0] < weight.shape[1]
            else:
                layout = weight.shape[0] > weight.shape[1]
    if layout:
        return weight  # Conv1D-style: already in TL orientation
    return weight.T  # nn.Linear-style: transpose to TL orientation


class MLPBridge(GeneralizedComponent):
    """Bridge component for MLP layers.

    This component wraps an MLP layer from a remote model and provides a consistent interface
    for accessing its weights and performing MLP operations.
    """

    hook_aliases = {"hook_pre": "in.hook_out", "hook_post": "out.hook_in"}
    # Containerless (name=None) instances refuse forward, so hook_in/hook_out
    # must be mirrored from the in/out subcomponents at setup.
    mirror_placeholder_hooks = True
    # W_* are real properties below (layout-aware); only the 1-D biases are
    # orientation-free enough for raw passthrough aliases.
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
        if self.name is None:
            # Containerless (fc1/fc2 sit on the decoder layer): the PARENT
            # LAYER is bound as original_component, so delegating would
            # silently run the whole layer — attention included.
            raise RuntimeError(
                f"{type(self).__name__} is containerless (name=None) — calling it "
                "directly would execute the whole parent layer. Call the block, "
                "or use the in/out submodules and their hooks."
            )
        hidden_states = args[0]
        hidden_states = self.hook_in(hidden_states)
        in_module = getattr(self, "in", None) or getattr(self, "input", None)
        if in_module is not None and not hasattr(in_module, "hook_in"):
            in_module = None
        if in_module is not None:
            hidden_states = in_module.hook_in(hidden_states)
        new_args = (hidden_states,) + args[1:]
        original_component = self.original_component
        if original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        out_module = getattr(self, "out", None)
        if out_module is not None:
            object.__setattr__(out_module, "_fired_hook_out", False)
        # The pre-fire above already hooked the tensor entering the module; tell
        # the replaced `in` projection to skip its own hook_in once so the same
        # tensor is not double-hooked (wrapped forwards that bypass the
        # projection leave the flag set; it is cleared below).
        if in_module is not None:
            object.__setattr__(in_module, "_suppress_next_hook_in", True)
        try:
            output = original_component(*new_args, **kwargs)
        finally:
            if in_module is not None:
                object.__setattr__(in_module, "_suppress_next_hook_in", False)
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
        # Fallback only, for wrapped forwards that bypass the replaced `out`
        # projection: if it did run, re-firing would double-apply interventions
        # and, on residual-inside MLPs (MPT), stamp the residual-added module
        # output over the additive contribution.
        if out_module is not None and not out_module._fired_hook_out:
            output = out_module.hook_out(output)
        return output

    def _weight_layout_in_out(self, proj: Any) -> Optional[bool]:
        """Whether proj's wrapped module stores its weight as [in, out]."""
        return weight_layout_in_out(proj)

    def _normalize_mlp_weight(
        self, weight: torch.Tensor, layout: Optional[bool], proj: Any, pattern: str = "in"
    ) -> torch.Tensor:
        """Normalize MLP weight to TL orientation."""
        return normalize_mlp_weight(weight, layout, proj, pattern=pattern)

    @property
    def W_in(self) -> torch.Tensor:
        """MLP input weight in TL orientation [d_model, d_mlp]."""
        in_module = getattr(self, "in", None)
        if in_module is None:
            raise AttributeError("No 'in' submodule on this MLP bridge")
        weight = require_readable_weight(
            in_module.weight, operation=f"read W_in from {self.name}", owner=in_module
        )
        layout = self._weight_layout_in_out(in_module)
        return self._normalize_mlp_weight(weight, layout, in_module, pattern="in")

    @property
    def W_gate(self) -> Optional[torch.Tensor]:
        """MLP gate weight in TL orientation [d_model, d_mlp], or None if ungated."""
        gate_module = getattr(self, "gate", None)
        if gate_module is None:
            return None
        weight = require_readable_weight(
            gate_module.weight, operation=f"read W_gate from {self.name}", owner=gate_module
        )
        layout = self._weight_layout_in_out(gate_module)
        return self._normalize_mlp_weight(weight, layout, gate_module, pattern="in")

    @property
    def W_out(self) -> torch.Tensor:
        """MLP output weight in TL orientation [d_mlp, d_model]."""
        out_module = getattr(self, "out", None)
        if out_module is None:
            raise AttributeError("No 'out' submodule on this MLP bridge")
        weight = require_readable_weight(
            out_module.weight, operation=f"read W_out from {self.name}", owner=out_module
        )
        layout = self._weight_layout_in_out(out_module)
        return self._normalize_mlp_weight(weight, layout, out_module, pattern="out")
