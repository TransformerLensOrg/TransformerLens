"""Linear bridge component for wrapping linear layers with hook points."""

from typing import Any, Dict, Optional

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

        # Apply input hook
        input = self.hook_in(input)

        # Forward through the original linear layer
        output = self.original_component(input, *args, **kwargs)

        # Apply output hook
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

    def process_weights(
        self,
        fold_ln: bool = False,
        center_writing_weights: bool = False,
        center_unembed: bool = False,
        fold_value_biases: bool = False,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process linear weights according to GPT2 pretrained logic.

        For linear layers, this is typically a direct mapping without transformation.
        """
        if self.original_component is None:
            return

        # Determine weight keys based on component name and context
        component_name = self.name or ""
        if "c_fc" in component_name or "input" in component_name:
            weight_key = "W_in"
            bias_key = "b_in"
        elif "c_proj" in component_name and "mlp" in str(type(self)).lower():
            weight_key = "W_out"
            bias_key = "b_out"
        elif "c_proj" in component_name and "attn" in str(type(self)).lower():
            weight_key = "W_O"
            bias_key = "b_O"
        else:
            # Default keys
            weight_key = "weight"
            bias_key = "bias"

        # Store processed weights in TransformerLens format (direct mapping)
        weight_tensor = getattr(self.original_component, "weight", None)
        bias_tensor = getattr(self.original_component, "bias", None)

        processed_weights = {}
        if weight_tensor is not None:
            processed_weights[weight_key] = weight_tensor.clone()

        # Add bias if it exists
        if bias_tensor is not None:
            processed_weights[bias_key] = bias_tensor.clone()

        self._processed_weights = processed_weights

    def get_processed_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the processed weights in TransformerLens format.

        Returns:
            Dictionary mapping TransformerLens parameter names to processed tensors
        """
        if not hasattr(self, "_processed_weights") or self._processed_weights is None:
            # If weights haven't been processed, process them now
            self.process_weights()

        return self._processed_weights.copy()
