"""Linear bridge component for wrapping linear layers with hook points."""
from typing import Any, Dict, Mapping

import einops
import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class LinearBridge(GeneralizedComponent):
    """Bridge component for linear layers.

    This component wraps a linear layer (nn.Linear) and provides hook points
    for intercepting the input and output activations.

    Note: For Conv1D layers (used in GPT-2 style models), use Conv1DBridge instead.
    """

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the linear layer with hooks.

        Args:
            input: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after linear transformation
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
        """String representation of the LinearBridge."""
        if self.original_component is not None:
            try:
                in_features = self.original_component.in_features
                out_features = self.original_component.out_features
                bias = self.original_component.bias is not None
                return f"LinearBridge({in_features} -> {out_features}, bias={bias}, original_component={type(self.original_component).__name__})"
            except AttributeError:
                return f"LinearBridge(name={self.name}, original_component={type(self.original_component).__name__})"
        else:
            return f"LinearBridge(name={self.name}, original_component=None)"

    def set_processed_weights(
        self, weights: Mapping[str, torch.Tensor | None], verbose: bool = False
    ) -> None:
        """Set the processed weights by loading them into the original component.

        This loads the processed weights directly into the original_component's parameters,
        so when forward() delegates to original_component, it uses the processed weights.

        Handles Linear layers (shape [out, in]).
        Also handles 3D weights [n_heads, d_model, d_head] by flattening them first.

        Args:
            weights: Dictionary containing:
                - weight: The processed weight tensor. Can be:
                    - 2D [in, out] format (will be transposed to [out, in] for Linear)
                    - 3D [n_heads, d_model, d_head] format (will be flattened to 2D)
                - bias: The processed bias tensor (optional). Can be:
                    - 1D [out] format
                    - 2D [n_heads, d_head] format (will be flattened to 1D)
            verbose: If True, print detailed information about weight setting
        """
        if verbose:
            print(f"\n  set_processed_weights: LinearBridge (name={self.name})")
            print(f"    Received {len(weights)} weight keys")

        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        weight = weights.get("weight")
        if weight is None:
            raise ValueError("Processed weights for LinearBridge must include 'weight'.")
        bias = weights.get("bias")

        if verbose:
            print(f"    Found weight key with shape: {weight.shape}")
            if bias is not None:
                print(f"    Found bias key with shape: {bias.shape}")

        # Handle 3D weights by flattening to 2D
        if weight.ndim == 3:
            n_heads, dim1, dim2 = weight.shape
            if dim1 > dim2:
                # [n_heads, d_model, d_head] -> [n_heads * d_head, d_model] (nn.Linear format)
                weight = einops.rearrange(
                    weight, "n_heads d_model d_head -> (n_heads d_head) d_model"
                )
            else:
                # [n_heads, d_head, d_model] -> [d_model, n_heads * d_head]
                weight = einops.rearrange(
                    weight, "n_heads d_head d_model -> d_model (n_heads d_head)"
                )

        # Handle 2D bias by flattening to 1D
        if bias is not None and bias.ndim == 2:
            bias = einops.rearrange(bias, "n_heads d_head -> (n_heads d_head)")

        processed_weights: Dict[str, torch.Tensor] = {
            "weight": weight,
        }

        if bias is not None:
            processed_weights["bias"] = bias

        super().set_processed_weights(processed_weights, verbose=verbose)
