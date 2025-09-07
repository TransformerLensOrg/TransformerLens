"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused qkv matrix.
"""

from typing import Any, Callable, Dict, Optional

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
    """Joint QKV attention bridge that wraps a joint qkv linear layer.

    This component wraps attention layers that use a fused qkv matrix such that
    the individual activations from the separated q, k, and v matrices are hooked and accessible.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        split_qkv_matrix: Callable,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        qkv_conversion_rule: Optional[BaseHookConversion] = None,
        attn_conversion_rule: Optional[BaseHookConversion] = None,
        pattern_conversion_rule: Optional[BaseHookConversion] = None,
    ):
        """Initialize the Joint QKV attention bridge.

        Args:
            name: The name of this component
            config: Model configuration (required for auto-conversion detection)
            split_qkv_matrix: Function to split the qkv matrix into q, k, and v linear transformations
            submodules: Dictionary of submodules to register (e.g., q_proj, k_proj, etc.)
            qkv_conversion_rule: Optional conversion rule for the individual q, k, and v matrices to convert their output shapes to HookedTransformer format. If None, uses default RearrangeHookConversion
            attn_conversion_rule: Optional conversion rule. Passed to parent AttentionBridge. If None, AttentionAutoConversion will be used
            pattern_conversion_rule: Optional conversion rule for attention patterns. If None,
                                   uses AttentionPatternConversion to ensure [n_heads, pos, pos] shape
        """
        super().__init__(
            name,
            config,
            submodules=submodules,
            conversion_rule=attn_conversion_rule,
            pattern_conversion_rule=pattern_conversion_rule,
        )

        self.split_qkv_matrix = split_qkv_matrix

        if qkv_conversion_rule is not None:
            self.qkv_conversion_rule = qkv_conversion_rule
        else:
            self.qkv_conversion_rule = self._create_qkv_conversion_rule()

        # Create LinearBridge components for q, k, and v activations
        self.q = LinearBridge(name="q")
        self.k = LinearBridge(name="k")
        self.v = LinearBridge(name="v")

        self.q.hook_in.hook_conversion = self.qkv_conversion_rule
        self.k.hook_in.hook_conversion = self.qkv_conversion_rule
        self.v.hook_in.hook_conversion = self.qkv_conversion_rule
        self.q.hook_out.hook_conversion = self.qkv_conversion_rule
        self.k.hook_out.hook_conversion = self.qkv_conversion_rule
        self.v.hook_out.hook_conversion = self.qkv_conversion_rule

    def _create_qkv_conversion_rule(self) -> RearrangeHookConversion:
        """Create the appropriate conversion rule for the individual q, k, and v matrices.

        Returns:
            RearrangeHookConversion for individual q, k, and v matrices
        """
        pattern = "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head"

        # keep mypy happy
        assert self.config is not None

        return RearrangeHookConversion(
            pattern,
            num_attention_heads=self.config.n_heads,
        )

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component that this bridge wraps and initialize LinearBridges for q, k, and v transformations.

        Args:
            original_component: The original attention layer to wrap
        """

        super().set_original_component(original_component)

        q_transformation, k_transformation, v_transformation = self.split_qkv_matrix(
            original_component
        )

        # Initialize LinearBridges for q, k, and v transformations
        self.q.set_original_component(q_transformation)
        self.k.set_original_component(k_transformation)
        self.v.set_original_component(v_transformation)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the qkv linear transformation with hooks.

        Args:
            *args: Input arguments, where the first argument should be the input tensor
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after qkv linear transformation
        """

        has_hooks = (
            self.q.hook_in.has_hooks()
            or self.k.hook_in.has_hooks()
            or self.v.hook_in.has_hooks()
            or self.q.hook_out.has_hooks()
            or self.k.hook_out.has_hooks()
            or self.v.hook_out.has_hooks()
        )

        if has_hooks:
            # Apply input hook the same way as the super class
            hooked_input = self._apply_attention_input_hook(*args, **kwargs)

            q_output = self.q(hooked_input)
            k_output = self.k(hooked_input)
            v_output = self.v(hooked_input)

            # Reconstruct attention computation using hooked Q, K, V
            output = self._reconstruct_attention(q_output, k_output, v_output, **kwargs)
            output = self._process_output(output)

            return output

        return super().forward(*args, **kwargs)

    def _apply_attention_input_hook(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply attention input hook to the input tensor.

        This method extracts the input tensor from args/kwargs and applies the attention
        input hook in the same way as the super class.

        Args:
            *args: Input arguments, where the first argument should be the input tensor
            **kwargs: Additional keyword arguments that might contain input

        Returns:
            Input tensor with attention input hook applied

        Raises:
            ValueError: If no input tensor is found in args or kwargs
        """
        # Extract input tensor using the same logic as the parent class
        input_tensor = None

        if "query_input" in kwargs:
            input_tensor = kwargs["query_input"]
        elif "hidden_states" in kwargs:
            input_tensor = kwargs["hidden_states"]
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
        else:
            raise ValueError("No input tensor found in args or kwargs")

        return self.hook_in(input_tensor)

    def _reconstruct_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> tuple:
        """Reconstruct attention computation using separate Q, K, V tensors."""
        original_component = self.original_component
        assert original_component is not None

        # Try to use the original _attn method if available
        if hasattr(original_component, "_attn"):
            if len(q.shape) == 4:
                q_attn = q.transpose(1, 2)
                k_attn = k.transpose(1, 2)
                v_attn = v.transpose(1, 2)
            elif len(q.shape) == 3:
                batch_size, seq_len, hidden_size = q.shape
                num_heads = int(original_component.num_heads)  # type: ignore[arg-type]
                head_dim: int = hidden_size // num_heads

                q_attn = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k_attn = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                v_attn = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            else:
                raise ValueError(f"Unexpected Q tensor shape: {q.shape}")

            attn_result = original_component._attn(  # type: ignore[operator]
                q_attn,
                k_attn,
                v_attn,
                attention_mask=kwargs.get("attention_mask"),
                head_mask=kwargs.get("head_mask"),
            )
            # Handle different return formats from _attn method
            if len(attn_result) == 2:
                attn_output, attn_weights = attn_result
            elif len(attn_result) == 3:
                attn_output, attn_weights, _ = attn_result  # Ignore past_key_value
            else:
                raise ValueError(
                    f"Unexpected number of return values from _attn: {len(attn_result)}"
                )

            if hasattr(original_component, "_merge_heads"):
                attn_output_merged = original_component._merge_heads(  # type: ignore[operator]
                    attn_output, original_component.num_heads, original_component.head_dim
                )
            else:
                batch_size, num_heads, seq_len, head_dim = attn_output.shape
                hidden_size = num_heads * head_dim
                attn_output_merged = (
                    attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
                )

            if hasattr(self, "o") and self.o is not None:
                attn_output_merged = self.o(attn_output_merged)

            # Return format should match what GPT2Block expects (exactly 2 values)
            # The GPT2Block handles past_key_value separately
            return (attn_output_merged, attn_weights)
        else:
            return self._manual_attention_computation(q, k, v, **kwargs)

    def _manual_attention_computation(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> tuple:
        """Manual attention computation as fallback."""
        original_component = self.original_component
        assert original_component is not None

        # keep mypy happy
        assert self.config is not None
        num_heads = self.config.n_heads

        if len(q.shape) == 3:
            batch_size, seq_len, hidden_size = q.shape
            head_dim: int = hidden_size // num_heads
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        elif len(q.shape) == 4:
            batch_size, seq_len, num_heads_tensor, head_dim = q.shape
            assert (
                num_heads_tensor == num_heads
            ), f"Expected {num_heads} heads, got {num_heads_tensor}"
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected Q tensor shape: {q.shape}. Expected 3D or 4D tensor.")

        scale = head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask for GPT-2
        if (
            hasattr(original_component, "register_buffer")
            or "gpt" in str(type(original_component)).lower()
        ):
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
            )
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            # Handle attention mask shape mismatch - slice to match sequence length
            if attention_mask.shape[-1] != seq_len:
                # Slice the attention mask to match the sequence length
                attention_mask = attention_mask[..., :seq_len]
            if attention_mask.shape[-2] != seq_len:
                attention_mask = attention_mask[..., :seq_len, :]
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        if hasattr(original_component, "attn_dropout"):
            attn_weights = original_component.attn_dropout(attn_weights)  # type: ignore[operator]

        attn_output = torch.matmul(attn_weights, v)

        final_hidden_size: int = num_heads * head_dim
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, final_hidden_size)
        )

        if hasattr(self, "o") and self.o is not None:
            attn_output = self.o(attn_output)

        # Return format should match what GPT2Block expects (exactly 2 values)
        # The GPT2Block handles past_key_value separately
        return (attn_output, attn_weights)  # (output, weights)
