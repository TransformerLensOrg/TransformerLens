"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused QKV matrix.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
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
        pattern_conversion_rule: Optional[BaseHookConversion] = None,
        qkv_conversion_rule: Optional[BaseHookConversion] = None,
    ):
        """Initialize the joint QKV attention bridge.

        Args:
            name: The name of this component
            model_config: Model configuration passed to parent AttentionBridge
            submodules: Dictionary of GeneralizedComponent submodules to register
            qkv_config: QKV-specific configuration including split_qkv_matrix function and conversion patterns
            pattern_conversion_rule: Optional conversion rule for attention patterns, passed to parent AttentionBridge
            qkv_conversion_rule: Optional conversion rule for QKV reshaping. If None, uses default RearrangeHookConversion
        """
        super().__init__(
            name,
            model_config,
            submodules=submodules,
            pattern_conversion_rule=pattern_conversion_rule,
        )

        self.qkv_config = qkv_config
        if self.qkv_config is None:
            raise RuntimeError(
                f"QKV config not set for {self.name}. QKV config is required for QKV separation."
            )
        if "split_qkv_matrix" not in self.qkv_config:
            raise RuntimeError(f"Config for {self.name} must include 'split_qkv_matrix' function.")

        # Create conversion rules for Q, K, V based on configuration
        if qkv_conversion_rule is not None:
            final_qkv_conversion_rule = qkv_conversion_rule
        else:
            final_qkv_conversion_rule = self._create_qkv_conversion_rule()

        # Create custom LinearBridge components for Q, K, and V activations with conversion rules only on output
        self.q = self._create_qkv_linear_bridge("q", model_config, final_qkv_conversion_rule)
        self.k = self._create_qkv_linear_bridge("k", model_config, final_qkv_conversion_rule)
        self.v = self._create_qkv_linear_bridge("v", model_config, final_qkv_conversion_rule)

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
            # Default pattern for individual Q/K/V: (batch, seq, n_heads*d_head) -> (batch, seq, n_heads, d_head)
            pattern = (
                "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head"
            )

        # Get number of heads from model config (passed to parent AttentionBridge)
        model_config = getattr(self, "config", None)
        if model_config is None:
            raise RuntimeError("Cannot create QKV conversion rule: model config not available")

        n_heads = getattr(model_config, "n_heads", None) or getattr(
            model_config, "num_attention_heads", None
        )
        if n_heads is None:
            raise RuntimeError(
                "Cannot create QKV conversion rule: num_attention_heads not found in config"
            )

        return RearrangeHookConversion(
            pattern,
            num_attention_heads=n_heads,
        )

    def _create_qkv_linear_bridge(
        self, name: str, model_config: Any, conversion_rule: BaseHookConversion
    ) -> LinearBridge:
        """Create a LinearBridge that only applies conversion rule to output hooks.

        Args:
            name: Name for the linear bridge
            model_config: Model configuration
            conversion_rule: Conversion rule to apply only to output

        Returns:
            LinearBridge with conversion rule applied only to output
        """
        # Create LinearBridge without conversion rule
        bridge = LinearBridge(name=name, config=model_config)

        # Manually apply conversion rule only to output hook
        bridge.hook_out.hook_conversion = conversion_rule

        return bridge

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

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the QKV linear transformation with hooks.

        Args:
            *args: Positional arguments (first should be input tensor)
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after QKV linear transformation
        """
        output = super().forward(*args, **kwargs)

        # Extract input tensor to run through individual Q, K, and V transformations
        # in order to hook their outputs
        input_tensor = (
            args[0] if len(args) > 0 else kwargs.get("input", kwargs.get("hidden_states"))
        )
        if input_tensor is not None:
            self.q(input_tensor)
            self.k(input_tensor)
            self.v(input_tensor)

        return output
