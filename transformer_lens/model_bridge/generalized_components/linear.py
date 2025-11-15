"""Linear bridge component for wrapping linear layers with hook points."""
from typing import Any, Dict, Mapping, Optional

import einops
import torch

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class LinearBridge(GeneralizedComponent):
    """Bridge component for linear layers.

    This component wraps a linear layer (nn.Linear) and provides hook points
    for intercepting the input and output activations.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        conversion_rule: Optional[BaseHookConversion] = None,
    ) -> None:
        """Initialize the LinearBridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for LinearBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
            conversion_rule: Optional conversion rule for this component's hooks
        """
        super().__init__(name, config, submodules=submodules, conversion_rule=conversion_rule)

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
                return f"LinearBridge({in_features} -> {out_features}, bias={bias})"
            except AttributeError:
                return f"LinearBridge(name={self.name}, original_component={type(self.original_component).__name__})"
        else:
            return f"LinearBridge(name={self.name}, original_component=None)"

    def set_processed_weights(self, weights: Mapping[str, torch.Tensor | None]) -> None:
        """Set the processed weights by loading them into the original component.

        This loads the processed weights directly into the original_component's parameters,
        so when forward() delegates to original_component, it uses the processed weights.

        Handles both Conv1D (GPT-2 style, shape [in, out]) and Linear (shape [out, in]).
        Also handles 3D weights [n_heads, d_model, d_head] by flattening them first.

        Args:
            weight: The processed weight tensor. Can be:
                - 2D [in, out] format
                - 3D [n_heads, d_model, d_head] format (will be flattened to 2D)
            bias: The processed bias tensor (optional). Can be:
                - 1D [out] format
                - 2D [n_heads, d_head] format (will be flattened to 1D)
        """
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        weight = weights.get("weight")
        if weight is None:
            raise ValueError("Processed weights for LinearBridge must include 'weight'.")
        bias = weights.get("bias")
        if weight.ndim == 3:
            n_heads, dim1, dim2 = weight.shape
            if dim1 > dim2:
                weight = einops.rearrange(
                    weight, "n_heads d_model d_head -> d_model (n_heads d_head)"
                )
            else:
                weight = einops.rearrange(
                    weight, "n_heads d_head d_model -> (n_heads d_head) d_model"
                )
        if bias is not None and bias.ndim == 2:
            bias = einops.rearrange(bias, "n_heads d_head -> (n_heads d_head)")
        for name, param in self.original_component.named_parameters():
            if "weight" in name.lower():
                try:
                    from transformers.pytorch_utils import Conv1D

                    is_conv1d = isinstance(self.original_component, Conv1D)
                except ImportError:
                    is_conv1d = False
                if is_conv1d:
                    param.data = weight.contiguous()
                else:
                    param.data = weight.T.contiguous()
            elif "bias" in name.lower() and bias is not None:
                param.data = bias.contiguous()
