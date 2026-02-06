"""Conv1D bridge component for wrapping Conv1D layers with hook points."""
from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class Conv1DBridge(GeneralizedComponent):
    """Bridge component for Conv1D layers.

    This component wraps a Conv1D layer (transformers.pytorch_utils.Conv1D)
    and provides hook points for intercepting the input and output activations.

    Conv1D is used in GPT-2 style models and has shape [in_features, out_features]
    (transpose of nn.Linear which is [out_features, in_features]).
    """

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the Conv1D layer with hooks.

        Args:
            input: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after Conv1D transformation
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        input = self.hook_in(input)
        output = self.original_component(input, *args, **kwargs)
        output = self.hook_out(output)
        return output

    def __repr__(self) -> str:
        """String representation of the Conv1DBridge."""
        if self.original_component is not None:
            try:
                # Conv1D has nf (out) and nx (in) attributes
                in_features = self.original_component.nx
                out_features = self.original_component.nf
                # Conv1D always has bias
                return f"Conv1DBridge({in_features} -> {out_features}, bias=True, original_component={type(self.original_component).__name__})"
            except AttributeError:
                return f"Conv1DBridge(name={self.name}, original_component={type(self.original_component).__name__})"
        else:
            return f"Conv1DBridge(name={self.name}, original_component=None)"
