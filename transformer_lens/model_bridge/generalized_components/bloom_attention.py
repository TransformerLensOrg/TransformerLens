"""BLOOM-specific attention bridge component.

BLOOM attention requires special arguments (residual, alibi, attention_mask) that standard
JointQKVAttentionBridge doesn't handle. This custom component passes these arguments through.
"""
from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
    JointQKVAttentionBridge,
)


class BloomAttentionBridge(JointQKVAttentionBridge):
    """Attention bridge for BLOOM models that handles residual connections and ALiBi.

    BLOOM attention has a unique forward signature that requires:
    - residual: The residual connection tensor from before the attention layer
    - alibi: ALiBi positional encoding bias
    - attention_mask: Attention mask for padding/causality

    This bridge ensures these arguments are properly passed through to the original component.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        split_qkv_matrix: Optional[Callable] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        qkv_conversion_rule: Optional[BaseTensorConversion] = None,
        attn_conversion_rule: Optional[BaseTensorConversion] = None,
        pattern_conversion_rule: Optional[BaseTensorConversion] = None,
    ):
        """Initialize the BLOOM attention bridge.

        Args:
            name: The name of this component
            config: Model configuration
            split_qkv_matrix: Function to split the qkv matrix into q, k, and v
            submodules: Dictionary of submodules to register
            qkv_conversion_rule: Optional conversion rule for q, k, v matrices
            attn_conversion_rule: Optional conversion rule for attention output
            pattern_conversion_rule: Optional conversion rule for attention patterns
        """
        # BLOOM attention doesn't require attention_mask as a constructor arg,
        # but it DOES require it in forward(), so we don't set requires_attention_mask=True
        super().__init__(
            name=name,
            config=config,
            split_qkv_matrix=split_qkv_matrix,
            submodules=submodules,
            qkv_conversion_rule=qkv_conversion_rule,
            attn_conversion_rule=attn_conversion_rule,
            pattern_conversion_rule=pattern_conversion_rule,
            requires_position_embeddings=False,
            requires_attention_mask=False,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through BLOOM attention with hooks.

        BLOOM attention requires these arguments:
        - hidden_states (first positional arg)
        - residual (second positional arg)
        - alibi, attention_mask, layer_past, etc. (keyword args)

        Args:
            *args: Input arguments (hidden_states, residual)
            **kwargs: Additional keyword arguments including alibi, attention_mask

        Returns:
            Output from BLOOM attention (tuple of hidden_states and optionally attention_weights)
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply hook_in to hidden_states (first positional argument)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        # BLOOM attention requires residual as second positional arg
        # The original BLOOM block passes it, so we just pass everything through
        # No need to validate since the original component will handle it

        # Call the original BLOOM attention component with all arguments
        # BLOOM attention returns (hidden_states,) or (hidden_states, attention_weights)
        output = self.original_component(*args, **kwargs)

        # Apply hook_out to the hidden_states (first element of tuple)
        if isinstance(output, tuple) and len(output) > 0:
            processed_output = list(output)
            if isinstance(output[0], torch.Tensor):
                processed_output[0] = self.hook_out(output[0])
            output = tuple(processed_output)
        elif isinstance(output, torch.Tensor):
            output = self.hook_out(output)

        return output
