"""Normalization bridge component implementation."""
from typing import Any, Dict, Optional, cast

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class NormalizationBridge(GeneralizedComponent):
    """Normalization bridge that wraps transformer normalization layers but implements the calculation from scratch.

    This component provides standardized input/output hooks.
    """

    property_aliases = {"w": "weight", "b": "bias"}

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
        use_native_layernorm_autograd: bool = False,
    ):
        """Initialize the normalization bridge.

        Args:
            name: The name of this component
            config: Optional configuration
            submodules: Dictionary of GeneralizedComponent submodules to register
            use_native_layernorm_autograd: If True, use HuggingFace's native LayerNorm
                                          autograd for exact gradient matching. If False,
                                          use custom implementation. Defaults to False.
        """
        super().__init__(name, config, submodules=submodules)
        self.hook_normalized = HookPoint()
        self.hook_scale = HookPoint()
        self.use_native_layernorm_autograd = use_native_layernorm_autograd

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the normalization bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Normalized output
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        assert self.config is not None
        hidden_states = self.hook_in(hidden_states)
        self._last_input_before_norm = hidden_states
        if self.use_native_layernorm_autograd:
            result = self._hf_autograd_forward_with_hooks(hidden_states)
        elif hasattr(self.config, "layer_norm_folding") and self.config.layer_norm_folding:
            result = self._hf_autograd_forward_with_hooks(hidden_states)
        else:
            uses_rms_norm = getattr(self.config, "uses_rms_norm", False)
            if not uses_rms_norm:
                hidden_states = hidden_states - hidden_states.mean(-1, keepdim=True)
            scale = self.hook_scale(
                (
                    hidden_states.pow(2).mean(-1, keepdim=True) + getattr(self.config, "eps", 1e-05)
                ).sqrt()
            )
            dtype = getattr(self.config, "dtype", hidden_states.dtype)
            hidden_states = self.hook_normalized(hidden_states / scale).to(dtype)
            if uses_rms_norm:
                hidden_states = hidden_states * self.weight
            else:
                hidden_states = hidden_states * self.weight
                if (
                    hasattr(self.original_component, "bias")
                    and self.original_component.bias is not None
                ):
                    hidden_states = hidden_states + cast(torch.Tensor, self.original_component.bias)
            result = hidden_states
        output = self.hook_out(result)
        return output

    def _hf_autograd_forward_with_hooks(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that preserves HF's autograd while firing intermediate hooks.

        This method calls HF's LayerNorm for the final result (to preserve exact gradients),
        but also computes intermediate values to fire hook_scale and hook_normalized.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor from HF's LayerNorm
        """
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        with torch.no_grad():
            if not getattr(self.config, "uses_rms_norm", False):
                x_centered = x - x.mean(-1, keepdim=True)
            else:
                x_centered = x
            eps_tensor = getattr(self.original_component, "eps", None)
            if eps_tensor is None:
                eps_tensor = getattr(self.original_component, "variance_epsilon", None)
            if eps_tensor is None:
                eps_value: float | torch.Tensor = getattr(self.config, "eps", 1e-05)
            else:
                eps_value = eps_tensor
            if isinstance(eps_value, torch.Tensor):
                scale = (x_centered.pow(2).mean(-1, keepdim=True) + eps_value).sqrt()
            else:
                scale = (x_centered.pow(2).mean(-1, keepdim=True) + float(eps_value)).sqrt()
            x_normalized = x_centered / scale
        _ = self.hook_scale(scale)
        _ = self.hook_normalized(x_normalized)
        input_dtype = x.dtype
        result = self.original_component(x)
        if result.dtype != input_dtype:
            result = result.to(input_dtype)
        return result

    def process_weights(
        self,
        fold_ln: bool = False,
        center_writing_weights: bool = False,
        center_unembed: bool = False,
        fold_value_biases: bool = False,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process normalization weights according to GPT2 pretrained logic.

        For layer norm, this is a direct mapping without transformation.
        """
        if self.original_component is None:
            return
        component_name = self.name or ""
        if "ln_f" in component_name or "final" in component_name:
            weight_key = "w"
            bias_key = "b"
        elif "ln_1" in component_name:
            weight_key = "w"
            bias_key = "b"
        elif "ln_2" in component_name:
            weight_key = "w"
            bias_key = "b"
        else:
            weight_key = "w"
            bias_key = "b"
        weight_tensor = getattr(self.original_component, "weight", None)
        bias_tensor = getattr(self.original_component, "bias", None)
        processed_weights = {}
        if weight_tensor is not None:
            processed_weights[weight_key] = weight_tensor.clone()
        if bias_tensor is not None:
            processed_weights[bias_key] = bias_tensor.clone()
        self._processed_weights = processed_weights
