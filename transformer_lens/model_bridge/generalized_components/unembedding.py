"""Unembedding bridge component.

This module contains the bridge component for unembedding layers.
"""
from collections import ChainMap
from typing import Any, Dict, Optional, cast

import torch
from accelerate.hooks import AlignDevicesHook, SequentialHook
from accelerate.utils import align_module_device

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

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component with a fixed zero bias when the checkpoint has none.

        Args:
            original_component: The original transformer component to wrap
        """
        # Keep b_U and state_dict compatibility without adding an optimizer parameter.
        if isinstance(original_component, torch.nn.Linear) and original_component.bias is None:
            # Get the output features (vocab size)
            vocab_size = original_component.weight.shape[0]  # shape is safe on a meta tensor too
            dtype = original_component.weight.dtype  # dtype is also safe on a meta tensor

            device = self._bias_device(original_component)

            del original_component.bias
            original_component.register_buffer(
                "bias", torch.zeros(vocab_size, device=device, dtype=dtype)
            )
            self._update_offloaded_bias(original_component)

        super().set_original_component(original_component)

    @staticmethod
    def _offload_hook(component: torch.nn.Module) -> Optional[AlignDevicesHook]:
        pending = [getattr(component, "_hf_hook", None)]
        while pending:
            hook = pending.pop()
            if isinstance(hook, SequentialHook):
                pending.extend(hook.hooks)
            elif isinstance(hook, AlignDevicesHook) and hook.offload:
                return hook
        return None

    @staticmethod
    def _bias_device(component: torch.nn.Module) -> torch.device:
        with align_module_device(component):
            weight = component.weight
            assert isinstance(weight, torch.Tensor)
            device = weight.device
        if device.type == "meta":
            # align_module_device does not recognize SequentialHook offloading.
            hook = UnembeddingBridge._offload_hook(component)
            if hook is not None and hook.execution_device is not None:
                device = torch.device(hook.execution_device)
        return device

    @staticmethod
    def _update_offloaded_bias(component: torch.nn.Module) -> None:
        hook = UnembeddingBridge._offload_hook(component)
        if hook is not None:
            bias = component.bias
            assert isinstance(bias, torch.Tensor)
            assert hook.weights_map is not None
            # Only the first ChainMap layer is written; the offload map stays lazy.
            hook.weights_map = ChainMap({"bias": bias.detach().cpu()}, cast(Any, hook.weights_map))
            # Hook removal must restore the latest bias alongside its weight.
            hook.original_devices.setdefault("bias", hook.original_devices["weight"])

    def set_processed_weights(
        self, weights: Dict[str, torch.Tensor], verbose: bool = False
    ) -> None:
        """Apply folded output biases while keeping synthetic biases non-trainable."""
        component = self.original_component
        if component is not None and "bias" in component._buffers and "bias" in weights:
            bias = component._buffers["bias"]
            new_bias = weights["bias"]
            if bias is not None and bias.shape != new_bias.shape:
                raise ValueError(
                    f"Shape mismatch when setting weight 'bias' in {type(component).__name__}: "
                    f"existing bias shape {bias.shape} != new tensor shape {new_bias.shape}"
                )
            component.bias = new_bias.detach()
            self._update_offloaded_bias(component)
        super().set_processed_weights(weights, verbose=verbose)

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
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        hidden_states = self.hook_in(hidden_states)
        output = self.original_component(hidden_states, **kwargs)

        output = self.hook_out(output)
        return output

    @property
    def b_U(self) -> torch.Tensor:
        """Access the unembedding bias vector."""
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
            dtype = weight.dtype  # safe on a meta tensor
            vocab_size: int = int(weight.shape[0])  # safe on a meta tensor
            device = self._bias_device(self.original_component)
            return torch.zeros(vocab_size, device=device, dtype=dtype)
