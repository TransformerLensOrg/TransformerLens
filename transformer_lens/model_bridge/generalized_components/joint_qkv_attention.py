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
        # Extract input tensor to run through individual Q, K, and V transformations
        input_tensor = (
            args[0] if len(args) > 0 else kwargs.get("input", kwargs.get("hidden_states"))
        )

        if input_tensor is not None:
            # Check if any hooks are registered on Q, K, or V (both input and output, forward and backward)
            has_hooks = (
                self.q.hook_in.has_hooks()
                or self.q.hook_out.has_hooks()
                or self.k.hook_in.has_hooks()
                or self.k.hook_out.has_hooks()
                or self.v.hook_in.has_hooks()
                or self.v.hook_out.has_hooks()
            )

            if has_hooks:
                # If hooks are present, we need to reconstruct the attention computation
                # using the hooked Q, K, V values instead of the fused computation

                # Apply input hook
                hooked_input = input_tensor
                if "query_input" in kwargs:
                    hooked_input = self.hook_in(kwargs["query_input"])
                elif "hidden_states" in kwargs:
                    hooked_input = self.hook_in(kwargs["hidden_states"])
                elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                    hooked_input = self.hook_in(args[0])

                # Run individual Q, K, V transformations (these can be hooked)
                q_output = self.q(hooked_input)  # This can be modified by hooks
                k_output = self.k(hooked_input)  # This can be modified by hooks
                v_output = self.v(hooked_input)  # This can be modified by hooks

                # Reconstruct attention computation using hooked Q, K, V
                output = self._reconstruct_attention(q_output, k_output, v_output, **kwargs)

                # Apply output hooks
                output = self._process_output(output)
                return output

        # Run the original fused computation (no hooks on Q, K, V)
        output = super().forward(*args, **kwargs)
        return output

    def _reconstruct_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> tuple:
        """Reconstruct attention computation using separate Q, K, V tensors.

        This method uses the original attention component's _attn method when possible,
        or falls back to manual computation when hooks have modified the Q, K, or V values.
        """
        original_component = self.original_component

        # Try to use the original _attn method if available
        if hasattr(original_component, "_attn"):
            # The original _attn method expects [batch, heads, seq, head_dim] format
            # Convert our Q, K, V tensors to this format

            if len(q.shape) == 4:
                # Format: [batch, pos, head_index, d_head] -> [batch, head_index, pos, d_head]
                q_attn = q.transpose(1, 2)
                k_attn = k.transpose(1, 2)
                v_attn = v.transpose(1, 2)
            elif len(q.shape) == 3:
                # Format: [batch, pos, hidden_size] -> need to reshape
                batch_size, seq_len, hidden_size = q.shape
                num_heads = original_component.num_heads
                head_dim = hidden_size // num_heads

                q_attn = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k_attn = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                v_attn = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            else:
                raise ValueError(f"Unexpected Q tensor shape: {q.shape}")

            # Call the original _attn method
            attn_output, attn_weights = original_component._attn(
                q_attn,
                k_attn,
                v_attn,
                attention_mask=kwargs.get("attention_mask"),
                head_mask=kwargs.get("head_mask"),
            )

            # The _attn method returns [batch, heads, seq, head_dim], need to merge heads
            # Use the original component's _merge_heads method for exact equivalence
            if hasattr(original_component, "_merge_heads"):
                attn_output_merged = original_component._merge_heads(
                    attn_output, original_component.num_heads, original_component.head_dim
                )
            else:
                # Fallback to manual reshaping if _merge_heads not available
                batch_size, num_heads, seq_len, head_dim = attn_output.shape
                hidden_size = num_heads * head_dim
                attn_output_merged = (
                    attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
                )

            # Apply output projection through our bridge component
            if hasattr(self, "o") and self.o is not None:
                attn_output_merged = self.o(attn_output_merged)

            return (attn_output_merged, attn_weights, None)

        # Fallback: manual computation (original implementation)
        else:
            return self._manual_attention_computation(q, k, v, **kwargs)

    def _manual_attention_computation(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> tuple:
        """Manual attention computation as fallback."""
        original_component = self.original_component

        # Extract attention parameters
        if hasattr(original_component, "num_heads"):
            num_heads = original_component.num_heads
        elif hasattr(original_component, "num_attention_heads"):
            num_heads = original_component.num_attention_heads
        else:
            raise ValueError("Cannot determine number of attention heads")

        # Handle both formats: [batch, pos, hidden_size] and [batch, pos, head_index, d_head]
        if len(q.shape) == 3:
            # Format: [batch, pos, hidden_size] - need to reshape to multi-head format
            batch_size, seq_len, hidden_size = q.shape
            head_dim = hidden_size // num_heads

            # Reshape Q, K, V for multi-head attention
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        elif len(q.shape) == 4:
            # Format: [batch, pos, head_index, d_head] - already in multi-head format
            batch_size, seq_len, num_heads_tensor, head_dim = q.shape

            # Verify the tensor dimensions match expected head count
            assert (
                num_heads_tensor == num_heads
            ), f"Expected {num_heads} heads, got {num_heads_tensor}"

            # Transpose to [batch, heads, seq, head_dim] format
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected Q tensor shape: {q.shape}. Expected 3D or 4D tensor.")

        # Compute attention scores
        scale = head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask (for GPT-2)
        if (
            hasattr(original_component, "register_buffer")
            or "gpt" in str(type(original_component)).lower()
        ):
            # Create causal mask
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
            )
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        # Apply attention mask if provided
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        # Apply dropout if configured (though we're in eval mode, so this should be a no-op)
        if hasattr(original_component, "attn_dropout"):
            attn_weights = original_component.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # Reshape back to original format
        hidden_size = num_heads * head_dim
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        )

        # Apply output projection (this should be handled by the 'o' component)
        if hasattr(self, "o") and self.o is not None:
            attn_output = self.o(attn_output)

        # Return in the same format as the original component (tuple with hidden_states, weights, patterns)
        return (attn_output, None, None)
