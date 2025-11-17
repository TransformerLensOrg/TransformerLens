"""Conv1D bridge component for wrapping Conv1D layers with hook points."""
from typing import Any, Mapping

import einops
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

    def set_processed_weights(
        self, weights: Mapping[str, torch.Tensor | None], verbose: bool = False
    ) -> None:
        """Set the processed weights by loading them into the original component.

        This loads the processed weights directly into the original_component's parameters,
        so when forward() delegates to original_component, it uses the processed weights.

        Handles Conv1D (GPT-2 style, shape [in, out]).
        Also handles 3D weights [n_heads, d_model, d_head] by flattening them first.

        Args:
            weights: Dictionary containing:
                - weight: The processed weight tensor. Can be:
                    - 2D [in, out] format (Conv1D format)
                    - 3D [n_heads, d_model, d_head] format (will be flattened to 2D)
                - bias: The processed bias tensor (optional). Can be:
                    - 1D [out] format
                    - 2D [n_heads, d_head] format (will be flattened to 1D)
            verbose: If True, print detailed information about weight setting
        """
        if verbose:
            print(f"\n  set_processed_weights: Conv1DBridge (name={self.name})")
            print(f"    Received {len(weights)} weight keys")

        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        weight = weights.get("weight")
        if weight is None:
            raise ValueError("Processed weights for Conv1DBridge must include 'weight'.")
        bias = weights.get("bias")

        if verbose:
            print(f"    Found weight key with shape: {weight.shape}")
            if bias is not None:
                print(f"    Found bias key with shape: {bias.shape}")

        # Handle 3D weights by flattening to 2D
        if weight.ndim == 3:
            n_heads, dim1, dim2 = weight.shape
            if dim1 > dim2:
                # [n_heads, d_model, d_head] -> [d_model, n_heads * d_head]
                weight = einops.rearrange(
                    weight, "n_heads d_model d_head -> d_model (n_heads d_head)"
                )
            else:
                # [n_heads, d_head, d_model] -> [n_heads * d_head, d_model]
                weight = einops.rearrange(
                    weight, "n_heads d_head d_model -> (n_heads d_head) d_model"
                )

        # Handle 2D bias by flattening to 1D
        if bias is not None and bias.ndim == 2:
            bias = einops.rearrange(bias, "n_heads d_head -> (n_heads d_head)")

        # Load weights into Conv1D layer
        # Conv1D stores weights in [in, out] format (no transpose needed)
        for name, param in self.original_component.named_parameters():
            if "weight" in name.lower():
                if verbose:
                    print(f"    Setting param '{name}' with shape {weight.contiguous().shape}")
                param.data = weight.contiguous()
            elif "bias" in name.lower() and bias is not None:
                if verbose:
                    print(f"    Setting param '{name}' with shape {bias.contiguous().shape}")
                param.data = bias.contiguous()
