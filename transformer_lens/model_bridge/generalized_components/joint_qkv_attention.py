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

        if qkv_conversion_rule is not None:
            final_qkv_conversion_rule = qkv_conversion_rule
        else:
            final_qkv_conversion_rule = self._create_qkv_conversion_rule()

        self.q = self._create_qkv_linear_bridge("q", model_config, final_qkv_conversion_rule)
        self.k = self._create_qkv_linear_bridge("k", model_config, final_qkv_conversion_rule)
        self.v = self._create_qkv_linear_bridge("v", model_config, final_qkv_conversion_rule)

    def _create_qkv_conversion_rule(self) -> RearrangeHookConversion:
        """Create the appropriate conversion rule for joint QKV matrices.

        Returns:
            RearrangeHookConversion for joint QKV reshaping
        """
        assert self.qkv_config is not None

        if "qkv_pattern" in self.qkv_config:
            pattern = self.qkv_config["qkv_pattern"]
        else:
            pattern = (
                "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head"
            )

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
        bridge = LinearBridge(name=name, config=model_config)
        bridge.hook_out.hook_conversion = conversion_rule

        return bridge

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component that this bridge wraps and initialize LinearBridges for Q, K, and V transformations.

        Args:
            original_component: The original attention layer to wrap
        """

        super().set_original_component(original_component)

        assert self.qkv_config is not None
        W_Q_transformation, W_K_transformation, W_V_transformation = self.qkv_config[
            "split_qkv_matrix"
        ](original_component)

        self.q.set_original_component(W_Q_transformation)
        self.k.set_original_component(W_K_transformation)
        self.v.set_original_component(W_V_transformation)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass that handles both fused and separate Q, K, V computations.

        When hooks are present on Q, K, or V components, this method reconstructs
        the attention computation using the hooked values. Otherwise, it falls back
        to the original fused computation for efficiency.
        """
        raw_input_tensor = (
            args[0] if len(args) > 0 else kwargs.get("input", kwargs.get("hidden_states"))
        )

        if raw_input_tensor is not None:
            has_hooks = (
                self.q.hook_in.has_hooks()
                or self.q.hook_out.has_hooks()
                or self.k.hook_in.has_hooks()
                or self.k.hook_out.has_hooks()
                or self.v.hook_in.has_hooks()
                or self.v.hook_out.has_hooks()
            )

            if has_hooks:
                # Apply input hook manually to match AttentionBridge behavior
                if "query_input" in kwargs:
                    self.hook_in(kwargs["query_input"])
                elif "hidden_states" in kwargs:
                    self.hook_in(kwargs["hidden_states"])
                elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                    self.hook_in(args[0])

                # Run individual Q, K, V transformations using raw input
                assert self.q.original_component is not None
                assert self.k.original_component is not None
                assert self.v.original_component is not None
                q_output = self.q.original_component(raw_input_tensor)
                k_output = self.k.original_component(raw_input_tensor)
                v_output = self.v.original_component(raw_input_tensor)

                # Apply output hooks
                q_output = self.q.hook_out(q_output)
                k_output = self.k.hook_out(k_output)
                v_output = self.v.hook_out(v_output)

                # Reconstruct attention computation using hooked Q, K, V
                output = self._reconstruct_attention(q_output, k_output, v_output, **kwargs)
                output = self._process_output(output)
                return output

        # Run the original fused computation (no hooks on Q, K, V)
        output = super().forward(*args, **kwargs)
        return output

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

        if hasattr(original_component, "num_heads"):
            num_heads = int(original_component.num_heads)  # type: ignore[arg-type]
        elif hasattr(original_component, "num_attention_heads"):
            num_heads = int(original_component.num_attention_heads)  # type: ignore[arg-type]
        else:
            raise ValueError("Cannot determine number of attention heads")

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
