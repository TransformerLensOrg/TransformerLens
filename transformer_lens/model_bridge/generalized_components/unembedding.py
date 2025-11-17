"""Unembedding bridge component.

This module contains the bridge component for unembedding layers.
"""
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

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
        """Return the unembedding weight matrix in TL format [d_model, d_vocab]."""
        if "_processed_W_U" in self._parameters:
            processed_W_U = self._parameters["_processed_W_U"]
            if processed_W_U is not None:
                # Processed weights are in HF format [vocab, d_model]
                # Transpose to TL format [d_model, d_vocab]
                return processed_W_U.T
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        assert hasattr(
            self.original_component, "weight"
        ), f"Component {self.name} has no weight attribute"
        weight = self.original_component.weight
        assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
        # HF format is [d_vocab, d_model], transpose to TL format [d_model, d_vocab]
        return weight.T

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the unembedding bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Unembedded output (logits)
        """
        # If using processed weights, use custom forward to handle format
        if "_processed_W_U" in self._parameters:
            processed_W_U = self._parameters["_processed_W_U"]
            if processed_W_U is not None:
                hidden_states = self.hook_in(hidden_states)
                # Note: processed weights are actually in HF format [vocab, d_model]
                # despite being called "processed_tl_weights" in the bridge
                # linear expects weight in [out_features, in_features] = [vocab, d_model]
                # So we use the weight as-is without transpose
                output = torch.nn.functional.linear(hidden_states, processed_W_U, self.b_U)
                output = self.hook_out(output)
                return output

        # Otherwise delegate to original component
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

    def set_processed_weights(
        self, weights: Mapping[str, torch.Tensor | None], verbose: bool = False
    ) -> None:
        """Set the processed weights by loading them into the original component.

        This loads the processed weights directly into the original_component's parameters,
        so when forward() delegates to original_component, it uses the processed weights.

        Note: W_U is expected in TL format [d_model, d_vocab] and will be transposed
        to HF format [d_vocab, d_model] before being stored in the original component.

        Args:
            weights: Dictionary containing:
                - "weight": The processed W_U tensor in TL format [d_model, d_vocab]
                - "bias": The processed b_U tensor (optional) [d_vocab]
            verbose: If True, print detailed information about weight setting
        """
        if verbose:
            print(
                f"\n  set_processed_weights: UnembeddingBridge (name={getattr(self, 'name', 'unknown')})"
            )
            print(f"    Received {len(weights)} weight keys")

        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")

        weight = weights.get("weight")
        if weight is None:
            raise ValueError("Processed weights for UnembeddingBridge must include 'weight'.")

        bias = weights.get("bias")

        if verbose:
            print(f"    Found weight key with shape: {weight.shape}")
            if bias is not None:
                print(f"    Found bias key with shape: {bias.shape}")

        # Register processed weights as parameters (for backward compatibility)
        self.register_parameter("_processed_W_U", torch.nn.Parameter(weight))
        if bias is not None:
            self.register_parameter("_b_U", torch.nn.Parameter(bias))
        else:
            vocab_size = weight.shape[1]
            self.register_parameter(
                "_b_U",
                torch.nn.Parameter(
                    torch.zeros(vocab_size, device=weight.device, dtype=weight.dtype)
                ),
            )

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
