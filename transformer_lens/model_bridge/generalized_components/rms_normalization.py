"""RMS Normalization bridge component implementation.

RMSNorm (Root Mean Square Layer Normalization) is used in models like T5, LLaMA, Mistral, etc.
Unlike LayerNorm, RMSNorm doesn't center the inputs (no mean subtraction) and has no bias.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)


class RMSNormalizationBridge(NormalizationBridge):
    """RMS Normalization bridge for models that use RMSNorm (T5, LLaMA, etc).

    RMSNorm differs from LayerNorm in two ways:
    1. No mean centering (no subtraction of mean)
    2. No bias term (only weight/scale parameter)

    This bridge does a simple pass-through to the original HuggingFace component
    with hooks on input and output.
    """

    property_aliases = {
        "w": "weight",
        # No bias alias for RMSNorm
    }

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, "GeneralizedComponent"]] = None,  # type: ignore
    ):
        """Initialize the RMS normalization bridge.

        Args:
            name: The name of this component
            config: Configuration object
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules or {})

        # Override config to indicate this is RMSNorm
        # This ensures the parent NormalizationBridge forward method
        # uses the correct normalization formula
        if self.config is not None and not hasattr(self.config, "uses_rms_norm"):
            self.config.uses_rms_norm = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass with simple delegation to the original component.

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

        # Simple pass-through: hook_in → delegate → hook_out
        hidden_states = self.hook_in(hidden_states)
        result = self.original_component(hidden_states, **kwargs)
        output = self.hook_out(result)
        return output
