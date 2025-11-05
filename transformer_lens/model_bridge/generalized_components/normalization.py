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

    property_aliases = {
        "w": "weight",
        "b": "bias",
    }

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

        # Store whether to use native layernorm autograd
        self.use_native_layernorm_autograd = use_native_layernorm_autograd

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
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

        # keep mypy happy
        assert self.config is not None

        hidden_states = self.hook_in(hidden_states)

        # Check if we should use HuggingFace's autograd directly (for exact gradient matching)
        if self.use_native_layernorm_autograd:
            # Use HuggingFace LayerNorm's forward directly to preserve exact computational graph
            result = self._hf_autograd_forward(hidden_states)
        # Check if we should use LayerNormPre behavior (when layer norm folding is enabled)
        elif hasattr(self.config, "layer_norm_folding") and self.config.layer_norm_folding:
            # LayerNormPre mode: center and normalize without learnable parameters
            # This matches LayerNormPre behavior exactly
            result = self._layernorm_pre_forward(hidden_states)
        else:
            # Standard normalization behavior with learnable parameters
            if not getattr(self.config, "uses_rms_norm", False):
                # Only center if not using RMSNorm
                hidden_states = hidden_states - hidden_states.mean(-1, keepdim=True)

            scale = self.hook_scale(
                (
                    hidden_states.pow(2).mean(-1, keepdim=True) + getattr(self.config, "eps", 1e-5)
                ).sqrt()
            )
            # Match HookedTransformer's dtype casting after normalization
            dtype = getattr(self.config, "dtype", hidden_states.dtype)
            hidden_states = self.hook_normalized(hidden_states / scale).to(dtype)

            # Apply learnable parameters if not folding layer norms
            if getattr(self.config, "uses_rms_norm", False):
                # No bias if using RMSNorm
                hidden_states = hidden_states * self.weight
            else:
                # Add bias if using LayerNorm and the original component has a bias
                hidden_states = hidden_states * self.weight
                if (
                    hasattr(self.original_component, "bias")
                    and self.original_component.bias is not None
                ):
                    hidden_states = hidden_states + cast(torch.Tensor, self.original_component.bias)

            result = hidden_states

        output = self.hook_out(result)
        return output

    def _hf_autograd_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass matching HookedTransformer's LayerNorm computation exactly.

        This replicates HookedTransformer's LayerNorm forward method to ensure
        the same computational graph and gradients.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """
        # Get parameters from the original component
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")

        # Handle different eps attribute names based on config
        # Most models use 'eps', but some (like Llama) use 'variance_epsilon'
        eps_attr = getattr(self.config, "eps_attr", "eps")
        eps = getattr(self.original_component, eps_attr, 1e-5)
        weight = self.original_component.weight
        bias = getattr(self.original_component, "bias", None)  # RMSNorm doesn't have bias

        # Match HookedTransformer LayerNorm computation exactly
        # dtype handling: convert to float32 if not float32/float64
        original_dtype = x.dtype
        if (
            self.config is not None
            and hasattr(self.config, "dtype")
            and self.config.dtype not in [torch.float32, torch.float64]
        ):
            x = x.to(torch.float32)

        x = x - x.mean(-1, keepdim=True)
        scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + eps).sqrt())  # type: ignore[operator]
        x = self.hook_normalized(x / scale)

        # Convert back to original dtype or config dtype
        if self.config is not None and hasattr(self.config, "dtype"):
            x = x.to(self.config.dtype)  # type: ignore[union-attr]
        else:
            # If no config dtype, use the weight's dtype to ensure consistency
            x = x.to(weight.dtype)  # type: ignore[arg-type]

        # Apply weight and bias (bias may be None for RMSNorm)
        if bias is not None:
            return x * weight + bias  # type: ignore[operator]
        else:
            return x * weight  # type: ignore[operator]

    def _layernorm_pre_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass matching LayerNormPre behavior exactly.

        This is the 'center and normalise' part of LayerNorm without learnable parameters.
        Centering is equivalent to deleting one direction of residual space.
        Normalising projects the residual stream onto the unit hypersphere.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """
        # Handle dtype conversion like LayerNormPre
        original_dtype = x.dtype
        config_dtype = getattr(self.config, "dtype", torch.float32)
        if config_dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        # Center: subtract mean (equivalent to centering)
        x = x - x.mean(-1, keepdim=True)

        # Normalize: apply scaling with hook
        eps = getattr(self.config, "eps", 1e-5)
        scale = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + eps).sqrt())
        result = self.hook_normalized(x / scale)

        # Convert back to original dtype
        return result.to(original_dtype)

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

        # Determine weight keys based on component name
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

        # Store processed weights in TransformerLens format (direct mapping)
        weight_tensor = getattr(self.original_component, "weight", None)
        bias_tensor = getattr(self.original_component, "bias", None)

        processed_weights = {}
        if weight_tensor is not None:
            processed_weights[weight_key] = weight_tensor.clone()
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

    def get_expected_parameter_names(self, prefix: str = "") -> list[str]:
        """Get the expected TransformerLens parameter names for this normalization component.

        Args:
            prefix: Prefix to add to parameter names (e.g., "blocks.0")

        Returns:
            List of expected parameter names in TransformerLens format
        """
        # Normalization components always have weight 'w' and bias 'b'
        weight_name = f"{prefix}.w" if prefix else "w"
        bias_name = f"{prefix}.b" if prefix else "b"
        return [weight_name, bias_name]

    @classmethod
    def create_normalization_bridge(
        cls,
        name: str,
        config: Any,
        original_component: Any,
    ) -> "NormalizationBridge":
        """Create a normalization bridge that adapts behavior based on runtime config.

        Args:
            name: The name of this component
            config: Configuration object
            original_component: The original layer norm component

        Returns:
            NormalizationBridge that adapts its behavior based on config.layer_norm_folding
        """
        # Create the bridge - behavior is determined at runtime based on config
        bridge = cls(name=name, config=config)

        # Set the original component
        bridge.set_original_component(original_component)

        return bridge
