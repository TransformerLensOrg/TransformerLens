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

    hook_aliases = {
        # hook_pre can be either "in.hook_out" (most models) or "input.hook_out" (GPT-2)
        "hook_pre": ["in.hook_out", "input.hook_out"],
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
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the MLP bridge.

        Args:
            name: The name of the component in the model (None if no container exists)
            config: Optional configuration (unused for MLPBridge)
            submodules: Dictionary of submodules to register (e.g., gate_proj, up_proj, down_proj)
        """
        super().__init__(name, config, submodules=submodules)

        # No extra hooks; use only hook_in and hook_out

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the MLP bridge.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output hidden states
        """

        # Check if we're using processed weights from a reference model (layer norm folding case)
        # This happens when set_processed_weights has been called
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            from transformer_lens.utilities.addmm import batch_addmm

            hidden_states = args[0]
            # Apply input hook
            hidden_states = self.hook_in(hidden_states)

            # Use the processed weights directly with the same computation as reference model
            if hasattr(self, "_processed_W_in") and hasattr(self, "_processed_W_out"):
                # Input projection using TransformerLens format weights [d_model, d_mlp]
                # Use batch_addmm to match HookedTransformer exactly - no transpose needed!
                # HookedTransformer: batch_addmm(self.b_in, self.W_in, x) where W_in is [d_model, d_mlp]
                # Handle None bias by creating a zero tensor with the appropriate shape
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

                # Apply hook_pre (in.hook_out or input.hook_out) - pre-activation hidden state
                # In compatibility mode, this hook is aliased as "blocks.L.mlp.hook_pre"
                # Try "in" first (standard name), then "input" (GPT-2 naming)
                in_module = getattr(self, "in", None) or getattr(self, "input", None)
                if in_module and hasattr(in_module, "hook_out"):
                    hidden = in_module.hook_out(hidden)

                # Apply activation (GELU for GPT-2)
                hidden = torch.nn.functional.gelu(hidden)

                # Apply hook_post (out.hook_in) - post-activation hidden state before output projection
                # In compatibility mode, this hook is aliased as "blocks.L.mlp.hook_post"
                if hasattr(self, "out") and hasattr(self.out, "hook_in"):
                    hidden = self.out.hook_in(hidden)

                # Output projection using TransformerLens format weights [d_mlp, d_model]
                # Use batch_addmm to match HookedTransformer exactly - no transpose needed!
                # HookedTransformer: batch_addmm(self.b_out, self.W_out, post_act) where W_out is [d_mlp, d_model]
                # Handle None bias by creating a zero tensor with the appropriate shape
                b_out = (
                    self._processed_b_out
                    if self._processed_b_out is not None
                    else torch.zeros(
                        self._processed_W_out.shape[-1], device=hidden.device, dtype=hidden.dtype
                    )
                )
                output = batch_addmm(b_out, self._processed_W_out, hidden)
            else:
                # Fallback to original component
                new_args = (hidden_states,) + args[1:]
                output = self.original_component(*new_args, **kwargs)  # type: ignore[misc]

            # Apply output hook
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
        self,
        W_in: torch.Tensor,
        W_out: torch.Tensor,
        b_in: torch.Tensor | None = None,
        b_out: torch.Tensor | None = None,
    ) -> None:
        """Set the processed weights to use when layer norm is folded.

        Args:
            W_in: The processed MLP input weight tensor
            W_out: The processed MLP output weight tensor
            b_in: The processed MLP input bias tensor (optional)
            b_out: The processed MLP output bias tensor (optional)
        """
        self._processed_W_in = W_in
        self._processed_W_out = W_out
        self._processed_b_in = b_in
        self._processed_b_out = b_out
        self._use_processed_weights = True
