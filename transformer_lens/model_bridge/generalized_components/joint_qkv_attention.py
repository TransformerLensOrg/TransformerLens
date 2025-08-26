"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused QKV matrix.
"""

from typing import Any, Dict, Optional

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.hook_point_wrapper import HookPointWrapper


class JointQKVAttentionBridge(AttentionBridge):
    """Joint QKV attention bridge that wraps an attention layer that uses a fused QKV matrix.

    This component wraps attention layers that use a fused QKV matrix such that both
    the q, k, and v activations are accessible as separate hook points. The actual
    separation of the q, k, and v activations is handled by the QKVBridge component.
    This bridge is used to make the q, k, and v activations accessible as hookpoints
    under a standardized naming scheme.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        conversion_rule: Optional[BaseHookConversion] = None,
        pattern_conversion_rule: Optional[BaseHookConversion] = None,
    ):
        """Initialize the joint QKV attention bridge.

        Args:
            name: The name of this component
            config: Model configuration (required for auto-conversion detection)
            submodules: Dictionary of submodules to register (e.g., q_proj, k_proj, etc.)
            conversion_rule: Optional conversion rule. If None, AttentionAutoConversion will be used
            pattern_conversion_rule: Optional conversion rule for attention patterns. If None,
                                   uses default RearrangeHookConversion to reshape to (batch, n_heads, pos, pos)
        """
        super().__init__(
            name,
            config,
            submodules=submodules,
            conversion_rule=conversion_rule,
            pattern_conversion_rule=pattern_conversion_rule,
        )

    @property
    def q(self) -> HookPointWrapper:
        return HookPointWrapper(
            self.submodules["qkv"].q_hook_in,
            self.submodules["qkv"].q_hook_out,
        )

    @property
    def k(self) -> HookPointWrapper:
        return HookPointWrapper(
            self.submodules["qkv"].k_hook_in,
            self.submodules["qkv"].k_hook_out,
        )

    @property
    def v(self) -> HookPointWrapper:
        return HookPointWrapper(
            self.submodules["qkv"].v_hook_in,
            self.submodules["qkv"].v_hook_out,
        )
