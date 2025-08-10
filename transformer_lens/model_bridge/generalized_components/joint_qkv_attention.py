"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused QKV matrix.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.conversion_utils.conversion_steps.rearrange_hook_conversion import (
    RearrangeHookConversion,
)
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class JointQKVAttentionBridge(AttentionBridge):
    """Joint QKV attention bridge that wraps a joint QKV linear layer.

    This component wraps attention layers that use a fused QKV matrix such that both
    the activations from the joint QKV matrix and from the individual, separated Q, K, and V matrices
    are hooked and accessible.
    """

    def __init__(
        self,
        name: str,
        model_config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        qkv_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the joint QKV attention bridge.

        Args:
            name: The name of this component
            model_config: Model configuration passed to parent AttentionBridge
            submodules: Dictionary of GeneralizedComponent submodules to register
            qkv_config: QKV-specific configuration including split_qkv_matrix function and conversion patterns
        """
        super().__init__(name, model_config, submodules=submodules)

        self.qkv_config = qkv_config
        if self.qkv_config is None:
            raise RuntimeError(
                f"QKV config not set for {self.name}. QKV config is required for QKV separation."
            )
        if "split_qkv_matrix" not in self.qkv_config:
            raise RuntimeError(f"Config for {self.name} must include 'split_qkv_matrix' function.")

        # Create conversion rules for Q, K, V based on configuration
        qkv_conversion_rule = self._create_qkv_conversion_rule()

        # Create LinearBridge components for Q, K, and V activations with conversion rules
        self.q = LinearBridge(name="q", config=model_config, conversion_rule=qkv_conversion_rule)
        self.k = LinearBridge(name="k", config=model_config, conversion_rule=qkv_conversion_rule)
        self.v = LinearBridge(name="v", config=model_config, conversion_rule=qkv_conversion_rule)

    def _create_qkv_conversion_rule(self) -> RearrangeHookConversion:
        """Create the appropriate conversion rule for joint QKV matrices.

        Returns:
            RearrangeHookConversion for joint QKV reshaping
        """
        # Keep mypy happy - we know qkv_config is not None due to earlier checks
        assert self.qkv_config is not None

        # Get conversion pattern from config, with sensible defaults
        if "qkv_pattern" in self.qkv_config:
            pattern = self.qkv_config["qkv_pattern"]
        else:
            # Default pattern for joint QKV: (d_model, 3*n_heads*d_head) -> (3, n_heads, d_model, d_head)
            pattern = "d_model (three num_attention_heads d_head) -> three num_attention_heads d_model d_head"

        # Get number of heads from model config (passed to parent AttentionBridge)
        model_config = getattr(self, "config", None)
        if model_config is None:
            raise RuntimeError(f"Cannot create QKV conversion rule: model config not available")

        n_heads = getattr(model_config, "n_heads", None) or getattr(
            model_config, "num_attention_heads", None
        )
        if n_heads is None:
            raise RuntimeError(
                f"Cannot create QKV conversion rule: num_attention_heads not found in config"
            )

        return RearrangeHookConversion(
            pattern,
            three=3,
            num_attention_heads=n_heads,
        )

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component that this bridge wraps and initialize LinearBridges for Q, K, and V transformations.

        Args:
            original_component: The original attention layer to wrap
        """

        super().set_original_component(original_component)

        # Keep mypy happy
        assert self.qkv_config is not None

        W_Q_transformation, W_K_transformation, W_V_transformation = self.qkv_config[
            "split_qkv_matrix"
        ](original_component)

        # Initialize LinearBridges for Q, K, and V transformations
        self.q.set_original_component(W_Q_transformation)
        self.k.set_original_component(W_K_transformation)
        self.v.set_original_component(W_V_transformation)

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the QKV linear transformation with hooks.

        Args:
            input: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after QKV linear transformation
        """
        output = super().forward(input, *args, **kwargs)

        # Run the input through the individual Q, K, and V transformations
        # in order to hook their outputs
        self.q(input)
        self.k(input)
        self.v(input)

        return output
