"""BLOOM-specific attention bridge component.

BLOOM attention requires special arguments (residual, alibi, attention_mask) that standard
JointQKVAttentionBridge doesn't handle. This custom component passes these arguments through.
"""
from typing import Any, Callable, Dict, Mapping, Optional

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

    def set_processed_weights(
        self, weights: Mapping[str, torch.Tensor | None], verbose: bool = False
    ) -> None:
        """Set processed weights and recombine Q/K/V back into combined QKV.

        BloomAttentionBridge's forward() delegates to the original HF attention
        component which uses the combined query_key_value weight. After weight
        processing (fold_ln etc.) modifies the split Q/K/V weights, we must
        recombine them back into the interleaved QKV format so the original
        component uses the processed weights.
        """
        # First, let the parent distribute weights to Q/K/V/O submodules
        super().set_processed_weights(dict(weights), verbose=verbose)  # type: ignore[arg-type]

        if self.original_component is None:
            return

        # Get the processed Q/K/V weights from split components
        assert self.q.original_component is not None
        assert self.k.original_component is not None
        assert self.v.original_component is not None
        q_weight: torch.Tensor = self.q.original_component.weight.data  # type: ignore[union-attr, assignment]
        k_weight: torch.Tensor = self.k.original_component.weight.data  # type: ignore[union-attr, assignment]
        v_weight: torch.Tensor = self.v.original_component.weight.data  # type: ignore[union-attr, assignment]

        assert self.config is not None
        n_heads: int = self.config.n_heads
        d_head: int = self.config.d_head
        d_model = int(q_weight.shape[1])

        # Reverse the split: recombine into interleaved QKV format
        # [n_heads*d_head, d_model] -> [d_model, n_heads, d_head]
        W_Q = q_weight.T.reshape(d_model, n_heads, d_head)
        W_K = k_weight.T.reshape(d_model, n_heads, d_head)
        W_V = v_weight.T.reshape(d_model, n_heads, d_head)

        # Stack into [d_model, n_heads, 3, d_head] (interleaved format)
        W_combined = torch.stack([W_Q, W_K, W_V], dim=2)

        # Reshape to [d_model, 3*n_heads*d_head] and transpose to nn.Linear format
        qkv_weight = W_combined.reshape(d_model, 3 * n_heads * d_head).T

        # Update the original component's combined QKV weight
        self.original_component.query_key_value.weight = torch.nn.Parameter(  # type: ignore[union-attr]
            qkv_weight
        )

        # Also recombine biases
        q_bias = self.q.original_component.bias  # type: ignore[union-attr]
        if q_bias is not None:
            assert self.k.original_component is not None
            assert self.v.original_component is not None
            k_bias = self.k.original_component.bias.data  # type: ignore[union-attr]
            v_bias = self.v.original_component.bias.data  # type: ignore[union-attr]

            # [n_heads*d_head] -> [n_heads, d_head]
            b_Q = q_bias.data.reshape(n_heads, d_head)  # type: ignore[union-attr, operator]
            b_K = k_bias.reshape(n_heads, d_head)  # type: ignore[operator]
            b_V = v_bias.reshape(n_heads, d_head)  # type: ignore[operator]

            # Stack into [n_heads, 3, d_head] and flatten
            qkv_bias = torch.stack([b_Q, b_K, b_V], dim=1).reshape(3 * n_heads * d_head)
            self.original_component.query_key_value.bias = torch.nn.Parameter(  # type: ignore[union-attr]
                qkv_bias
            )
