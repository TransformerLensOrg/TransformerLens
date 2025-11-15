"""Unembedding bridge component.

This module contains the bridge component for unembedding layers.
"""
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class UnembeddingBridge(GeneralizedComponent):
    """Unembedding bridge that wraps transformer unembedding layers.

    This component provides standardized input/output hooks.
    """

    property_aliases = {"W_U": "u.weight"}

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the unembedding bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for UnembeddingBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)

    @property
    def W_U(self) -> torch.Tensor:
        """Return the unembedding weight matrix."""
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            if "_processed_W_U" in self._parameters:
                processed_W_U = self._parameters["_processed_W_U"]
                if processed_W_U is not None:
                    return processed_W_U
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        assert hasattr(
            self.original_component, "weight"
        ), f"Component {self.name} has no weight attribute"
        weight = self.original_component.weight
        assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
        return weight.T

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the unembedding bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Unembedded output (logits)
        """
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            hidden_states = self.hook_in(hidden_states)
            if "_processed_W_U" in self._parameters:
                processed_W_U = self._parameters["_processed_W_U"]
                if processed_W_U is not None:
                    output = torch.nn.functional.linear(hidden_states, processed_W_U.T, self.b_U)
                else:
                    output = torch.nn.functional.linear(hidden_states, self.W_U.T, self.b_U)
            else:
                output = torch.nn.functional.linear(hidden_states, self.W_U.T, self.b_U)
            output = self.hook_out(output)
            return output
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        target_dtype = None
        try:
            target_dtype = next(self.original_component.parameters()).dtype
        except StopIteration:
            pass
        hidden_states = self.hook_in(hidden_states)
        if (
            target_dtype is not None
            and isinstance(hidden_states, torch.Tensor)
            and hidden_states.is_floating_point()
        ):
            hidden_states = hidden_states.to(dtype=target_dtype)
        output = self.original_component(hidden_states, **kwargs)
        output = self.hook_out(output)
        return output

    @property
    def b_U(self) -> torch.Tensor:
        """Access the unembedding bias vector."""
        if "_b_U" in self._parameters:
            param = self._parameters["_b_U"]
            if param is not None:
                return param
        if hasattr(self, "_processed_b_U") and self._processed_b_U is not None:
            return self._processed_b_U
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        if hasattr(self.original_component, "bias") and self.original_component.bias is not None:
            bias = self.original_component.bias
            assert isinstance(bias, torch.Tensor), f"Bias is not a tensor for {self.name}"
            return bias
        else:
            assert hasattr(
                self.original_component, "weight"
            ), f"Component {self.name} has no weight attribute"
            weight = self.original_component.weight
            assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
            device = weight.device
            dtype = weight.dtype
            vocab_size: int = int(weight.shape[0])
            return torch.zeros(vocab_size, device=device, dtype=dtype)

    def set_processed_weight(self, W_U: torch.Tensor, b_U: torch.Tensor | None = None) -> None:
        """Set the processed weights to use when layer norm is folded.

        Args:
            W_U: The processed unembedding weight tensor
            b_U: The processed unembedding bias tensor (optional)
        """
        self.register_parameter("_processed_W_U", torch.nn.Parameter(W_U))
        if b_U is not None:
            self.register_parameter("_b_U", torch.nn.Parameter(b_U))
        else:
            vocab_size = W_U.shape[1]
            self.register_parameter(
                "_b_U",
                torch.nn.Parameter(torch.zeros(vocab_size, device=W_U.device, dtype=W_U.dtype)),
            )
        self._use_processed_weights = True

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override named_parameters to expose _b_U as b_U.

        This ensures that the parameter shows up as 'unembed.b_U' instead of 'unembed._b_U'
        in the output, matching HookedTransformer's naming convention.
        """
        for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
            if name.endswith("._b_U"):
                yield (name[:-5] + ".b_U", param)
            elif name == "_b_U":
                yield ("b_U", param)
            else:
                yield (name, param)
