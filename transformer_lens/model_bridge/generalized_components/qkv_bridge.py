"""QKV bridge component for wrapping linear layers that are a joint qkv projection."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.rearrange_hook_conversion import (
    RearrangeHookConversion,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class QKVBridge(GeneralizedComponent):
    """Bridge component for QKV linear layers.

    This component wraps linear layers that are used for joint QKV projections
    in attention mechanisms.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        base_conversion_rule: Optional[BaseHookConversion] = None,
        qkv_conversion_rule: Optional[BaseHookConversion] = None,
        qkv_separation_rule: Optional[BaseHookConversion] = None,
    ) -> None:
        """Initialize the QKVBridge.

        Args:
            name: The name of this component
            config: Model configuration
            submodules: Dictionary of GeneralizedComponent submodules to register
            base_conversion_rule: Optional conversion rule for hook_in and hook_out of this component. If None, no conversion is applied to hook_in and hook_out.
            qkv_conversion_rule: Optional conversion rule for QKV reshaping. If None, uses default RearrangeHookConversion
            qkv_separation_rule: Optional separation rule for the output of the QKV layer. If None, uses default RearrangeHookConversion
        """
        super().__init__(name, config, submodules=submodules, conversion_rule=base_conversion_rule)

        self.q_hook_in = HookPoint()
        self.k_hook_in = HookPoint()
        self.v_hook_in = HookPoint()
        self.q_hook_out = HookPoint()
        self.k_hook_out = HookPoint()
        self.v_hook_out = HookPoint()

        if qkv_conversion_rule is not None:
            self.qkv_conversion_rule = qkv_conversion_rule
        else:
            self.qkv_conversion_rule = self._create_qkv_conversion_rule()

        if qkv_separation_rule is not None:
            self.qkv_separation_rule = qkv_separation_rule
        else:
            self.qkv_separation_rule = self._create_qkv_separation_rule()

        self.q_hook_in.hook_conversion = self.qkv_conversion_rule
        self.k_hook_in.hook_conversion = self.qkv_conversion_rule
        self.v_hook_in.hook_conversion = self.qkv_conversion_rule

        self.q_hook_out.hook_conversion = self.qkv_conversion_rule
        self.k_hook_out.hook_conversion = self.qkv_conversion_rule
        self.v_hook_out.hook_conversion = self.qkv_conversion_rule

    def _create_qkv_conversion_rule(self) -> RearrangeHookConversion:
        """Create the appropriate conversion rule for joint QKV matrices.

        Returns:
            RearrangeHookConversion for joint QKV reshaping
        """
        pattern = "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head"

        assert self.config is not None, "Config is required to create QKV conversion rule"

        return RearrangeHookConversion(
            pattern,
            num_attention_heads=self.config.n_heads,
        )

    def _create_qkv_separation_rule(self) -> RearrangeHookConversion:
        """Create the appropriate separation rule for QKV outputs.

        Returns:
            RearrangeHookConversion for QKV output separation
        """

        pattern = "batch seq (three d_model) -> three batch seq d_model"

        return RearrangeHookConversion(
            pattern,
            three=3,
        )

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the QKV linear layer.

        Args:
            input: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after QKV projection
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        input = self.q_hook_in(input)
        input = self.k_hook_in(input)
        input = self.v_hook_in(input)

        output = self.original_component(input, *args, **kwargs)

        has_hooks = (
            self.q_hook_out.has_hooks()
            or self.k_hook_out.has_hooks()
            or self.v_hook_out.has_hooks()
        )

        if has_hooks:
            q_output, k_output, v_output = self.qkv_separation_rule.handle_conversion(output)

            q_output = self.q_hook_out(q_output)
            k_output = self.k_hook_out(k_output)
            v_output = self.v_hook_out(v_output)

            original_output = torch.stack((q_output, k_output, v_output), dim=0)

            output = self.qkv_separation_rule.revert(original_output)

            return output

        return output

    def __repr__(self) -> str:
        """String representation of the QKVBridge."""
        if self.original_component is not None:
            return f"QKVBridge({self.in_features} -> {self.out_features}, bias={self.bias})"
        else:
            return f"QKVBridge(name={self.name}, original_component=None)"
