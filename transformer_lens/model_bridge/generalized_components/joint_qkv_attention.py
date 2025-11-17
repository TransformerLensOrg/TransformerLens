"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused qkv matrix.
"""
from typing import Any, Callable, Dict, Optional

import einops
import torch

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
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

    # Property aliases point to the linear bridge weights
    property_aliases = {
        "W_Q": "q.weight",
        "W_K": "k.weight",
        "W_V": "v.weight",
        "W_O": "o.weight",
        "b_Q": "q.bias",
        "b_K": "k.bias",
        "b_V": "v.bias",
        "b_O": "o.bias",
    }

    def __init__(
        self,
        name: str,
        config: Any,
        split_qkv_matrix: Callable,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        qkv_conversion_rule: Optional[BaseHookConversion] = None,
        attn_conversion_rule: Optional[BaseHookConversion] = None,
        pattern_conversion_rule: Optional[BaseHookConversion] = None,
        requires_position_embeddings: bool = False,
        requires_attention_mask: bool = False,
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
            requires_position_embeddings: Whether this attention requires position_embeddings as input
            requires_attention_mask: Whether this attention requires attention_mask as input
        """
        super().__init__(
            name,
            config,
            submodules=submodules,
            conversion_rule=attn_conversion_rule,
            pattern_conversion_rule=pattern_conversion_rule,
            requires_position_embeddings=requires_position_embeddings,
            requires_attention_mask=requires_attention_mask,
        )
        self.split_qkv_matrix = split_qkv_matrix
        if qkv_conversion_rule is not None:
            self.qkv_conversion_rule = qkv_conversion_rule
        else:
            self.qkv_conversion_rule = self._create_qkv_conversion_rule()
        self.q = LinearBridge(name="q")
        self.k = LinearBridge(name="k")
        self.v = LinearBridge(name="v")
        for submodule_name, submodule in (submodules or {}).items():
            if not hasattr(self, submodule_name):
                setattr(self, submodule_name, submodule)
        self.submodules["q"] = self.q
        self.submodules["k"] = self.k
        self.submodules["v"] = self.v
        self.q.hook_out.hook_conversion = self.qkv_conversion_rule
        self.k.hook_out.hook_conversion = self.qkv_conversion_rule
        self.v.hook_out.hook_conversion = self.qkv_conversion_rule

        # Register q, k, v LinearBridges in real_components for weight distribution
        # This allows set_processed_weights to distribute weights to these submodules
        self.real_components["q"] = ("q", self.q)
        self.real_components["k"] = ("k", self.k)
        self.real_components["v"] = ("v", self.v)
        if hasattr(self, "o"):
            self.real_components["o"] = ("o", self.o)

        self._processed_weights: Optional[Dict[str, torch.Tensor]] = None
        self._hooked_weights_extracted = False
        self._W_Q: Optional[torch.Tensor] = None
        self._W_K: Optional[torch.Tensor] = None
        self._W_V: Optional[torch.Tensor] = None
        self._W_O: Optional[torch.Tensor] = None
        self._b_Q: Optional[torch.Tensor] = None
        self._b_K: Optional[torch.Tensor] = None
        self._b_V: Optional[torch.Tensor] = None
        self._b_O: Optional[torch.Tensor] = None
        self._reference_model: Optional[Any] = None
        self._layer_idx: Optional[int] = None

    def _create_qkv_conversion_rule(self) -> BaseHookConversion:
        """Create the appropriate conversion rule for the individual q, k, and v matrices.

        Returns:
            BaseHookConversion for individual q, k, and v matrices
        """
        assert self.config is not None

        class ConditionalRearrangeConversion(BaseHookConversion):
            def __init__(self, n_heads: int):
                super().__init__()
                self.n_heads = n_heads
                self.pattern = (
                    "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head"
                )

            def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
                if input_value.ndim == 4:
                    return input_value
                elif input_value.ndim == 3:
                    return einops.rearrange(
                        input_value, self.pattern, num_attention_heads=self.n_heads
                    )
                else:
                    raise ValueError(
                        f"Expected 3D or 4D tensor, got {input_value.ndim}D with shape {input_value.shape}"
                    )

            def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
                if input_value.ndim == 3:
                    return input_value
                elif input_value.ndim == 4:
                    return einops.rearrange(
                        input_value,
                        "batch seq num_attention_heads d_head -> batch seq (num_attention_heads d_head)",
                        num_attention_heads=self.n_heads,
                    )
                else:
                    raise ValueError(
                        f"Expected 3D or 4D tensor, got {input_value.ndim}D with shape {input_value.shape}"
                    )

        return ConditionalRearrangeConversion(self.config.n_heads)

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component that this bridge wraps and initialize LinearBridges for q, k, v, and o transformations.

        Args:
            original_component: The original attention layer to wrap
        """
        super().set_original_component(original_component)
        q_transformation, k_transformation, v_transformation = self.split_qkv_matrix(
            original_component
        )
        self.q.set_original_component(q_transformation)
        self.k.set_original_component(k_transformation)
        self.v.set_original_component(v_transformation)
        if hasattr(self, "o") and hasattr(original_component, "c_proj"):
            self.o.set_original_component(original_component.c_proj)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the qkv linear transformation with hooks.

        Args:
            *args: Input arguments, where the first argument should be the input tensor
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after qkv linear transformation
        """
        hooked_input = self._apply_attention_input_hook(*args, **kwargs)
        q_output = self.q(hooked_input)
        k_output = self.k(hooked_input)
        v_output = self.v(hooked_input)
        output = self._reconstruct_attention(q_output, k_output, v_output, **kwargs)
        output = self._process_output(output)
        return output

    def _process_output(self, output: Any) -> Any:
        """Process the output from _reconstruct_attention.

        This override skips the duplicate hook_pattern call since
        _reconstruct_attention already applies both hook_attn_scores
        and hook_pattern correctly.

        Args:
            output: Output tuple from _reconstruct_attention (attn_output, attn_weights)

        Returns:
            Processed output with hook_out applied
        """
        attn_pattern = None
        if isinstance(output, tuple) and len(output) >= 2:
            attn_pattern = output[1]
        if attn_pattern is not None:
            self._pattern = attn_pattern
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            processed_output = list(output)
            processed_output[0] = self.hook_hidden_states(output[0])
            output = tuple(processed_output)
        if isinstance(output, torch.Tensor):
            output = self.hook_out(output)
        elif isinstance(output, tuple) and len(output) > 0:
            processed_tuple = list(output)
            if isinstance(output[0], torch.Tensor):
                processed_tuple[0] = self.hook_out(output[0])
            if len(processed_tuple) == 1:
                return processed_tuple[0]
            output = tuple(processed_tuple)
        return output

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
        """Manual attention computation as fallback using TransformerLens computation logic."""
        original_component = self.original_component
        assert original_component is not None
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
        scale = head_dim ** (-0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if attention_mask.shape[-1] != seq_len:
                attention_mask = attention_mask[..., :seq_len]
            if attention_mask.shape[-2] != seq_len:
                attention_mask = attention_mask[..., :seq_len, :]
            attn_scores = attn_scores + attention_mask
        attn_scores = self.hook_attn_scores(attn_scores)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        if hasattr(original_component, "attn_dropout"):
            attn_weights = original_component.attn_dropout(attn_weights)  # type: ignore[operator]
        attn_weights = self.hook_pattern(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        final_hidden_size: int = num_heads * head_dim
        attn_output = attn_output.view(batch_size, seq_len, final_hidden_size)
        if hasattr(self, "o") and self.o is not None:
            attn_output = self.o(attn_output)
        return (attn_output, attn_weights)
