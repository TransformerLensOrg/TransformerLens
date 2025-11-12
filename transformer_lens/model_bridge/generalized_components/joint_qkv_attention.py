"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused qkv matrix.
"""

from typing import Any, Callable, Dict, Optional, cast

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Create LinearBridge components for q, k, v activations
        # PyTorch automatically registers them as submodules when assigned as attributes
        self.q = LinearBridge(name="q")
        self.k = LinearBridge(name="k")
        self.v = LinearBridge(name="v")

        # Register submodules passed from architecture adapter (e.g., o, qkv)
        # This ensures they're properly registered as nn.Module children for compatibility_mode
        for submodule_name, submodule in (submodules or {}).items():
            if not hasattr(self, submodule_name):
                setattr(self, submodule_name, submodule)

        # Apply hook conversion ONLY to hook_out, not hook_in
        # hook_in should remain 3D [batch, seq, d_model] (input to the linear layer)
        # hook_out: Convert 3D linear output [batch, seq, n_heads*d_head] to 4D [batch, seq, n_heads, d_head]
        # self.q.hook_in.hook_conversion = self.qkv_conversion_rule  # Removed - should be 3D
        # self.k.hook_in.hook_conversion = self.qkv_conversion_rule  # Removed - should be 3D
        # self.v.hook_in.hook_conversion = self.qkv_conversion_rule  # Removed - should be 3D

        self.q.hook_out.hook_conversion = self.qkv_conversion_rule
        self.k.hook_out.hook_conversion = self.qkv_conversion_rule
        self.v.hook_out.hook_conversion = self.qkv_conversion_rule

        # Store processed weights after weight processing
        self._processed_weights: Optional[Dict[str, torch.Tensor]] = None
        self._hooked_weights_extracted = False

        # HookedTransformer-style weights (populated lazily)
        self._W_Q: Optional[torch.Tensor] = None
        self._W_K: Optional[torch.Tensor] = None
        self._W_V: Optional[torch.Tensor] = None
        self._W_O: Optional[torch.Tensor] = None
        self._b_Q: Optional[torch.Tensor] = None
        self._b_K: Optional[torch.Tensor] = None
        self._b_V: Optional[torch.Tensor] = None
        self._b_O: Optional[torch.Tensor] = None

        # Cache attributes (populated by bridge during weight loading)
        self._reference_model: Optional[Any] = None
        self._layer_idx: Optional[int] = None

    def _create_qkv_conversion_rule(self) -> BaseHookConversion:
        """Create the appropriate conversion rule for the individual q, k, and v matrices.

        Returns:
            BaseHookConversion for individual q, k, and v matrices
        """
        # keep mypy happy
        assert self.config is not None

        # Create a custom conversion that checks tensor shape before applying rearrange
        # This handles both unprocessed (3D) and processed (4D) weights
        class ConditionalRearrangeConversion(BaseHookConversion):
            def __init__(self, n_heads: int):
                super().__init__()
                self.n_heads = n_heads
                self.pattern = (
                    "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head"
                )

            def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
                if input_value.ndim == 4:
                    # Already in correct format [batch, seq, n_heads, d_head], no conversion needed
                    return input_value
                elif input_value.ndim == 3:
                    # Need conversion from [batch, seq, d_model] to [batch, seq, n_heads, d_head]
                    return einops.rearrange(
                        input_value,
                        self.pattern,
                        num_attention_heads=self.n_heads,
                    )
                else:
                    raise ValueError(
                        f"Expected 3D or 4D tensor, got {input_value.ndim}D with shape {input_value.shape}"
                    )

            def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
                if input_value.ndim == 3:
                    # Already in flat format, no revert needed
                    return input_value
                elif input_value.ndim == 4:
                    # Revert from [batch, seq, n_heads, d_head] to [batch, seq, d_model]
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

        # Initialize LinearBridges for q, k, and v transformations
        self.q.set_original_component(q_transformation)
        self.k.set_original_component(k_transformation)
        self.v.set_original_component(v_transformation)

        # Initialize o (output projection) LinearBridge if it exists as a submodule
        # It's provided by the architecture adapter in the submodules dict
        if hasattr(self, "o") and hasattr(original_component, "c_proj"):
            self.o.set_original_component(original_component.c_proj)

        # Note: hook_z reshaping is setup during compatibility mode via setup_no_processing_hooks()
        # Don't set it up here to avoid breaking non-compatibility mode tests

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the qkv linear transformation with hooks.

        Args:
            *args: Input arguments, where the first argument should be the input tensor
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after qkv linear transformation
        """
        # If we're using processed weights (_use_processed_weights is True),
        # use the compatibility mode forward that uses 3D weights like HookedTransformer
        # if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
        #     return self._compatibility_mode_forward_with_hooks(*args, **kwargs)

        # Otherwise, use the standard forward pass with 2D weights
        hooked_input = self._apply_attention_input_hook(*args, **kwargs)

        q_output = self.q(hooked_input)
        k_output = self.k(hooked_input)
        v_output = self.v(hooked_input)

        # Reconstruct attention computation using hooked Q, K, V
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
        # Extract attention pattern from the output tuple if present
        attn_pattern = None
        if isinstance(output, tuple) and len(output) >= 2:
            attn_pattern = output[1]

        # Store the pattern for potential use (but don't re-hook it)
        if attn_pattern is not None:
            self._pattern = attn_pattern

        # Apply hook_hidden_states to the attention output (first element of tuple)
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            processed_output = list(output)
            processed_output[0] = self.hook_hidden_states(output[0])
            output = tuple(processed_output)

        # Apply hook_out to the main output
        output = self._apply_hook_out_to_output(output)

        return output

    def _forward_folded(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass using folded weights (split QKV with standard c_attn).

        This implements the HookedTransformer-style attention computation using
        the standard HF c_attn component but with split QKV logic.
        """
        # Extract hidden_states from args or kwargs
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        else:
            raise ValueError("No hidden_states found in input")

        # Apply input hook
        hidden_states = self.hook_in(hidden_states)

        batch_size, seq_len, d_model = hidden_states.shape
        cfg = self.config

        # Get the original HF attention component
        original_attn = self.original_component

        # Apply QKV projection using processed weights if available
        # Check if we have processed weights extracted
        hooked_weights_available = (
            hasattr(self, "_hooked_weights_extracted") and self._hooked_weights_extracted
        )
        if hooked_weights_available:
            print(f"🔧 Using processed weights for layer attention forward pass")
        else:
            print(
                f"⚠️  Falling back to original weights (hooked_weights_extracted: {getattr(self, '_hooked_weights_extracted', 'missing')})"
            )

        if hooked_weights_available:
            # Use the processed weights directly (like HookedTransformer would)
            if hasattr(self, "_W_Q") and hasattr(self, "_W_K") and hasattr(self, "_W_V"):
                # Apply the QKV projection manually using processed weights
                W_Q = self._W_Q  # [n_heads, d_model, d_head]
                W_K = self._W_K  # [n_heads, d_model, d_head]
                W_V = self._W_V  # [n_heads, d_model, d_head]
                b_Q = self._b_Q if hasattr(self, "_b_Q") else None  # [n_heads, d_head]
                b_K = self._b_K if hasattr(self, "_b_K") else None  # [n_heads, d_head]
                b_V = self._b_V if hasattr(self, "_b_V") else None  # [n_heads, d_head]

                # Convert to format needed for matrix multiplication
                # Reshape weights: [n_heads, d_model, d_head] -> [d_model, n_heads * d_head]
                W_Q_flat = (
                    W_Q.transpose(0, 1).contiguous().view(cfg.d_model, -1)  # type: ignore[union-attr]
                )  # [d_model, n_heads*d_head]
                W_K_flat = (
                    W_K.transpose(0, 1).contiguous().view(cfg.d_model, -1)  # type: ignore[union-attr]
                )  # [d_model, n_heads*d_head]
                W_V_flat = (
                    W_V.transpose(0, 1).contiguous().view(cfg.d_model, -1)  # type: ignore[union-attr]
                )  # [d_model, n_heads*d_head]

                # Apply projections
                q_flat = torch.matmul(hidden_states, W_Q_flat)  # [batch, seq_len, n_heads*d_head]
                k_flat = torch.matmul(hidden_states, W_K_flat)  # [batch, seq_len, n_heads*d_head]
                v_flat = torch.matmul(hidden_states, W_V_flat)  # [batch, seq_len, n_heads*d_head]

                # Add biases if they exist
                if b_Q is not None:
                    b_Q_flat = b_Q.view(-1)  # [n_heads*d_head]
                    q_flat = q_flat + b_Q_flat
                if b_K is not None:
                    b_K_flat = b_K.view(-1)  # [n_heads*d_head]
                    k_flat = k_flat + b_K_flat
                if b_V is not None:
                    b_V_flat = b_V.view(-1)  # [n_heads*d_head]
                    v_flat = v_flat + b_V_flat

                # Split into separate Q, K, V tensors
                q = q_flat
                k = k_flat
                v = v_flat
            else:
                # Fallback to original weights if processed weights not available
                qkv = original_attn.c_attn(hidden_states)  # type: ignore[operator, union-attr]  # [batch, seq_len, 3*d_model]
                q, k, v = qkv.split(cfg.d_model, dim=2)  # type: ignore[union-attr]
        else:
            # Use original weights (unprocessed)
            qkv = original_attn.c_attn(hidden_states)  # type: ignore[operator, union-attr]  # [batch, seq_len, 3*d_model]
            # Split into Q, K, V
            q, k, v = qkv.split(cfg.d_model, dim=2)  # type: ignore[union-attr]

        # Reshape to multi-head format: [batch, n_heads, seq_len, d_head]
        q = q.view(batch_size, seq_len, cfg.n_heads, cfg.d_head).transpose(1, 2)  # type: ignore[union-attr]
        k = k.view(batch_size, seq_len, cfg.n_heads, cfg.d_head).transpose(1, 2)  # type: ignore[union-attr]
        v = v.view(batch_size, seq_len, cfg.n_heads, cfg.d_head).transpose(1, 2)  # type: ignore[union-attr]

        # Apply V hook if it exists (important for interpretability)
        # Note: We need to apply hooks directly to the correct format without conversion
        # since we're bypassing the normal QKV projection pathway in folded mode
        if hasattr(self, "v") and hasattr(self.v, "hook_out") and self.v.hook_out.has_hooks():
            # Convert to [batch, seq, heads, d_head] format for hook
            v_for_hook = v.transpose(1, 2)  # [batch, seq, heads, d_head]

            # Apply hook directly without conversion (bypass the conversion rule)
            # Store the original conversion rule temporarily
            original_conversion = getattr(self.v.hook_out, "hook_conversion", None)
            self.v.hook_out.hook_conversion = None

            try:
                v_hooked = self.v.hook_out(v_for_hook)  # [batch, seq, heads, d_head]
            finally:
                # Restore the original conversion rule
                self.v.hook_out.hook_conversion = original_conversion

            # Convert back to attention format: [batch, heads, seq, d_head]
            v = v_hooked.transpose(1, 2)  # [batch, heads, seq, d_head]

        # Attention scores: [batch, n_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (cfg.d_head**0.5)  # type: ignore[union-attr]

        # Apply causal mask for GPT-2 (always causal for GPT-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # Apply attention scores hook (for compatibility with HookedTransformer)
        attn_scores = self.hook_attn_scores(attn_scores)

        # Softmax attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply pattern hook (for compatibility with HookedTransformer)
        attn_weights = self.hook_pattern(attn_weights)

        # Apply attention to values: [batch, n_heads, seq_len, d_head]
        attn_out = torch.matmul(attn_weights, v)

        # Reshape back to [batch, seq_len, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Apply output projection through LinearBridge (captures hook_z via o.hook_in)
        # hook_z is aliased to o.hook_in in AttentionBridge.hook_aliases
        result = self.o(attn_out)

        # Apply output hook
        result = self.hook_out(result)

        # Return in HuggingFace format (output, weights) - GPT-2 always expects both
        return (result, attn_weights)

    def _forward_standard(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass using standard HF attention component and hook processing."""
        # has_hooks = (
        # self.q.hook_in.has_hooks()
        # or self.k.hook_in.has_hooks()
        # or self.v.hook_in.has_hooks()
        # or self.q.hook_out.has_hooks()
        # or self.k.hook_out.has_hooks()
        # or self.v.hook_out.has_hooks()
        # )

        # In compatibility mode, ALWAYS use the split Q/K/V path to ensure
        # backward hooks fire 3 times through ln1 (matching HookedTransformer)
        # This is critical for backward hook parity - even without user hooks on q/k/v,
        # the split computation creates 3 separate backward paths that are needed
        # for ln1 backward hooks to match HookedTransformer's behavior
        # if getattr(self, "compatibility_mode", False):
        #     return self._compatibility_mode_forward_with_hooks(*args, **kwargs)

        # if has_hooks:
        # Apply input hook the same way as the super class

        # return super().forward(*args, **kwargs)

    def _compatibility_mode_forward_with_hooks(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass in compatibility mode that matches HookedTransformer behavior exactly.

        This method ensures that when hooks are applied in compatibility mode,
        the computation path matches HookedTransformer exactly by computing V values
        using the same method as HookedTransformer (simple_attn_linear).
        """
        # Get the original input
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
        elif "hidden_states" in kwargs:
            input_tensor = kwargs["hidden_states"]
        elif "query_input" in kwargs:
            input_tensor = kwargs["query_input"]
        else:
            raise ValueError("No input tensor found in args or kwargs")

        cached_pre_ln: Optional[torch.Tensor] = None
        if hasattr(self, "_ln1") and self._ln1 is not None:
            get_cached = getattr(self._ln1, "get_last_input_before_norm", None)
            if callable(get_cached):
                cached_pre_ln = get_cached()

        if cached_pre_ln is not None:
            input_tensor = cached_pre_ln

        input_tensor = self.hook_in(input_tensor)

        original_component = self.original_component
        assert original_component is not None

        # Extract HookedTransformer-compatible weights if not already done
        if not hasattr(self, "_hooked_weights_extracted") or not self._hooked_weights_extracted:
            self._extract_hooked_transformer_weights()

        # Fall back to original component if weight extraction failed
        if (
            not self._hooked_weights_extracted
            or not hasattr(self, "_W_Q")
            or not hasattr(self, "_W_K")
            or not hasattr(self, "_W_V")
            or self._W_Q is None
            or self._W_K is None
            or self._W_V is None
        ):
            return super().forward(*args, **kwargs)

        # Create zero bias tensors if needed (cached to avoid recreation on every forward)
        if self._b_Q is None:
            self._b_Q = torch.zeros(
                self._W_Q.shape[0],
                self._W_Q.shape[2],
                dtype=self._W_Q.dtype,
                device=self._W_Q.device,
            )
        if self._b_K is None:
            self._b_K = torch.zeros(
                self._W_K.shape[0],
                self._W_K.shape[2],
                dtype=self._W_K.dtype,
                device=self._W_K.device,
            )
        if self._b_V is None:
            self._b_V = torch.zeros(
                self._W_V.shape[0],
                self._W_V.shape[2],
                dtype=self._W_V.dtype,
                device=self._W_V.device,
            )

        # CRITICAL: To match HookedTransformer's behavior where ln1 backward hooks fire 3 times,
        # we need to call ln1 THREE SEPARATE TIMES (once for Q, K, V inputs).
        # HookedTransformer does: self.ln1(query_input), self.ln1(key_input), self.ln1(value_input)
        # This creates 3 separate forward passes through ln1, which means 3 backward hook firings.

        # Check if we have ln1 reference (set during compatibility mode setup)
        if hasattr(self, "_ln1") and self._ln1 is not None:
            # Input tensor is pre-ln1 residual, call ln1 three times
            q_input = self._ln1(input_tensor)
            k_input = self._ln1(input_tensor)
            v_input = self._ln1(input_tensor)
        else:
            # Fallback: input tensor is already post-ln1, use it directly
            # This won't fire ln1 hooks 3 times, but keeps functionality working
            q_input = input_tensor
            k_input = input_tensor
            v_input = input_tensor

        # Compute Q, K, V using the extracted 3D weights directly (like HookedTransformer)
        # This matches HookedTransformer's simple_attn_linear computation
        # W_Q, W_K, W_V: [n_heads, d_model, d_head]
        # input: [batch, seq, d_model]
        # output: [batch, seq, n_heads, d_head]
        import einops

        # Use simple_attn_linear pattern from HookedTransformer while manually firing hooks
        # This ensures all hooks fire properly while using 3D weights
        from transformer_lens.utilities.attention import simple_attn_linear

        # Fire hook_in hooks manually for Q, K, V inputs
        q_input = self.q.hook_in(q_input)
        k_input = self.k.hook_in(k_input)
        v_input = self.v.hook_in(v_input)

        # Temporarily disable conversion rules since simple_attn_linear produces correct 4D format
        original_q_conversion = self.q.hook_out.hook_conversion
        original_k_conversion = self.k.hook_out.hook_conversion
        original_v_conversion = self.v.hook_out.hook_conversion
        self.q.hook_out.hook_conversion = None
        self.k.hook_out.hook_conversion = None
        self.v.hook_out.hook_conversion = None

        try:
            # Calculate Q, K, V using 3D weights and fire hook_out
            # simple_attn_linear: [batch, pos, d_model] x [n_heads, d_model, d_head] -> [batch, pos, n_heads, d_head]
            q = self.q.hook_out(simple_attn_linear(q_input, self._W_Q, self._b_Q))
            k = self.k.hook_out(simple_attn_linear(k_input, self._W_K, self._b_K))
            v = self.v.hook_out(simple_attn_linear(v_input, self._W_V, self._b_V))
        finally:
            # Restore conversion rules
            self.q.hook_out.hook_conversion = original_q_conversion
            self.k.hook_out.hook_conversion = original_k_conversion
            self.v.hook_out.hook_conversion = original_v_conversion

        # Handle KV caching
        # Don't use "or" because DynamicCache might evaluate to False in boolean context
        # Try newer parameter name first (past_key_values), then fall back to older names
        past_key_value_arg = kwargs.get("past_key_values")
        if past_key_value_arg is None:
            past_key_value_arg = kwargs.get("past_key_value")
        if past_key_value_arg is None:
            past_key_value_arg = kwargs.get("layer_past")
        use_cache = kwargs.get("use_cache", False)

        # Transpose for attention computation: [batch, seq, heads, d_head] -> [batch, heads, seq, d_head]
        # IMPORTANT: Reassign to q to ensure hook modifications are preserved
        q = q.transpose(1, 2)
        k_new = k.transpose(1, 2)
        v_new = v.transpose(1, 2)

        if past_key_value_arg is not None and hasattr(past_key_value_arg, "update"):
            layer_idx = getattr(self, "layer_idx", 0)
            k, v = past_key_value_arg.update(k_new, v_new, layer_idx)
        else:
            k = k_new
            v = v_new

        import torch.nn.functional as F

        head_dim = q.shape[-1]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)

        # Apply causal mask (GPT-2 style)
        # When using KV cache, q_len might be 1 (new token) but k_len is past + new
        q_len = q.shape[2]
        k_len = k.shape[2]

        # Create causal mask: [q_len, k_len]
        # For position i in query, can attend to positions 0...(kv_offset + i) in key
        kv_offset = k_len - q_len
        causal_mask = torch.tril(torch.ones(q_len, k_len, device=q.device), diagonal=kv_offset)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # Apply attention mask if provided
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            attn_scores = attn_scores + kwargs["attention_mask"]

        # Apply attention scores hook
        attn_scores = self.hook_attn_scores(attn_scores)

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply pattern hook
        attn_weights = self.hook_pattern(attn_weights)

        # Apply dropout if the original component has it
        if hasattr(original_component, "attn_dropout"):
            attn_weights = original_component.attn_dropout(attn_weights)  # type: ignore[operator]

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back: [batch, heads, seq, d_head] -> [batch, seq, heads, d_head]
        z = attn_output.transpose(1, 2).contiguous()

        # Apply hook_z (on 4D format before output projection)
        z = self.o.hook_in(z)

        # Use HookedTransformer's pattern for output projection with 3D W_O
        if hasattr(self, "_W_O") and self._W_O is not None:
            # Rearrange W_O [n_heads, d_head, d_model] -> [d_model, n_heads*d_head] for F.linear
            w = einops.rearrange(
                self._W_O, "head_index d_head d_model -> d_model (head_index d_head)"
            )

            # Flatten z [batch, seq, n_heads, d_head] -> [batch, seq, n_heads*d_head]
            z_flat = z.reshape(z.shape[0], z.shape[1], -1)

            # Apply F.linear
            import torch.nn.functional as F

            attn_output = F.linear(z_flat, w, self._b_O if hasattr(self, "_b_O") else None)

            # Apply hook_out from LinearBridge
            attn_output = self.o.hook_out(attn_output)
        else:
            # Fall back to using LinearBridge with 2D weights
            # Reshape to flat: [batch, seq, heads * d_head]
            z_flat = z.view(z.shape[0], z.shape[1], -1)
            attn_output = self.o(z_flat)

        attn_output = self.hook_out(attn_output)

        # Return format: (attn_output, cache/weights)
        if use_cache and past_key_value_arg is not None:
            return (attn_output, past_key_value_arg)
        else:
            return (attn_output, attn_weights)

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
        """Manual attention computation as fallback using TransformerLens computation logic."""
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

        # Compute attention scores using TransformerLens logic
        scale = head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask using the same approach as WorkingAttention
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            # Handle attention mask shape mismatch - slice to match sequence length
            if attention_mask.shape[-1] != seq_len:
                # Slice the attention mask to match the sequence length
                attention_mask = attention_mask[..., :seq_len]
            if attention_mask.shape[-2] != seq_len:
                attention_mask = attention_mask[..., :seq_len, :]
            attn_scores = attn_scores + attention_mask

        # Apply hook_attn_scores to raw scores BEFORE softmax
        attn_scores = self.hook_attn_scores(attn_scores)

        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        if hasattr(original_component, "attn_dropout"):
            attn_weights = original_component.attn_dropout(attn_weights)  # type: ignore[operator]  # type: ignore[operator]

        # Apply hook_pattern to probabilities AFTER softmax
        attn_weights = self.hook_pattern(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq, d_head]

        # Transpose back to [batch, seq, n_heads, d_head]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq, n_heads, d_head]

        # Flatten to [batch, seq, d_model] for output projection
        final_hidden_size: int = num_heads * head_dim
        attn_output = attn_output.view(batch_size, seq_len, final_hidden_size)

        # Apply output projection through LinearBridge
        # Following the exact pattern from AttentionBridge and _compatibility_mode_forward_with_hooks:
        # 1. Pass flat tensor [batch, seq, d_model] to self.o()
        # 2. self.o.hook_in fires (this is hook_z)
        # 3. The hook_conversion on o.hook_in reshapes to [batch, seq, n_heads, d_head] for hooks
        # 4. Then reshapes back to flat for the actual linear layer (c_proj)
        # 5. self.o.hook_out fires with the output
        if hasattr(self, "o") and self.o is not None:
            attn_output = self.o(attn_output)
        elif hasattr(original_component, "c_proj"):
            # Fallback: if self.o doesn't exist for some reason, use functional linear
            attn_output = self._apply_output_projection_with_functional_linear(attn_output)

        # Return format should match what GPT2Block expects (exactly 2 values)
        # The GPT2Block handles past_key_value separately
        return (attn_output, attn_weights)  # (output, weights)

    def _apply_qkv_projection_with_functional_linear(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply QKV projection using torch.nn.functional.linear with transposed weights.

        This method implements the TransformerLens computation logic from WorkingAttention,
        using torch.nn.functional.linear for weight projection.

        Args:
            hidden_states: Input hidden states tensor [batch_size, seq_len, d_model]

        Returns:
            Tuple of (q, k, v) tensors after projection
        """
        original_component = self.original_component
        assert original_component is not None
        assert self.config is not None

        # Extract weights from original attention component
        if hasattr(original_component, "c_attn"):
            c_attn = cast(nn.Module, original_component.c_attn)
            qkv_weight = c_attn.weight  # Shape: [d_model, 3*d_model]
            qkv_bias = c_attn.bias  # Shape: [3*d_model]
        else:
            raise AttributeError(
                "Original component doesn't have c_attn attribute for QKV projection"
            )

        batch_size, seq_len, d_model = hidden_states.shape

        # Apply QKV projection using torch.nn.functional.linear
        # Note: torch.nn.functional.linear expects weight to be [output_features, input_features]
        # but HuggingFace stores it as [input_features, output_features], so we transpose
        qkv = torch.nn.functional.linear(
            hidden_states, cast(torch.Tensor, qkv_weight.T), cast(torch.Tensor, qkv_bias)
        )

        # Split into Q, K, V - reshape to separate the 3 components
        qkv = qkv.view(batch_size, seq_len, 3, d_model)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        return q, k, v

    def _apply_output_projection_with_functional_linear(
        self, attn_output: torch.Tensor
    ) -> torch.Tensor:
        """Apply output projection using torch.nn.functional.linear with transposed weights.

        This method implements the output projection logic from WorkingAttention,
        using torch.nn.functional.linear for weight projection.

        Args:
            attn_output: Attention output tensor [batch_size, seq_len, d_model]

        Returns:
            Final output tensor after projection
        """
        original_component = self.original_component
        assert original_component is not None

        # Extract output projection weights from original attention component
        if hasattr(original_component, "c_proj"):
            c_proj = cast(nn.Module, original_component.c_proj)
            proj_weight = c_proj.weight  # Shape: [d_model, d_model]
            proj_bias = c_proj.bias  # Shape: [d_model]
        else:
            # If no output projection, return the input unchanged
            return attn_output

        # Apply output projection using torch.nn.functional.linear
        # Note: torch.nn.functional.linear expects weight to be [output_features, input_features]
        # but HuggingFace stores it as [input_features, output_features], so we transpose
        output = torch.nn.functional.linear(
            attn_output, cast(torch.Tensor, proj_weight.T), cast(torch.Tensor, proj_bias)
        )

        return output

    def set_processed_weights(
        self,
        W_Q: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        W_O: torch.Tensor,
        b_Q: Optional[torch.Tensor] = None,
        b_K: Optional[torch.Tensor] = None,
        b_V: Optional[torch.Tensor] = None,
        b_O: Optional[torch.Tensor] = None,
    ) -> None:
        """Set processed weights for Joint QKV attention.

        For Joint QKV attention, the Q/K/V weights are stored in a single c_attn component.
        This method handles both 2D format [d_model, (n_heads*d_head)] and 3D TL format
        [n_heads, d_model, d_head] for Q/K/V weights.

        Args:
            W_Q: Query weight tensor (2D or 3D format)
            W_K: Key weight tensor (2D or 3D format)
            W_V: Value weight tensor (2D or 3D format)
            W_O: Output projection weight (2D HF format [d_model, d_model] or 3D TL format)
            b_Q: Query bias tensor (optional)
            b_K: Key bias tensor (optional)
            b_V: Value bias tensor (optional)
            b_O: Output bias tensor (optional)
        """
        import einops

        # Convert Q/K/V weights to TL format [n_heads, d_model, d_head] if needed
        if W_Q.ndim == 2:
            # 2D format [d_model, (n_heads*d_head)] -> 3D TL format [n_heads, d_model, d_head]
            assert self.config is not None
            n_heads = self.config.n_heads
            W_Q = einops.rearrange(
                W_Q, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
            )
            W_K = einops.rearrange(
                W_K, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
            )
            W_V = einops.rearrange(
                W_V, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
            )

            # Also reshape biases if they exist and are 1D
            if b_Q is not None and b_Q.ndim == 1:
                b_Q = einops.rearrange(b_Q, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
            if b_K is not None and b_K.ndim == 1:
                b_K = einops.rearrange(b_K, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
            if b_V is not None and b_V.ndim == 1:
                b_V = einops.rearrange(b_V, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)

        # Store W_Q/W_K/W_V attributes for property access (e.g., bridge.blocks[0].attn.W_Q)
        # These are now in TL format [n_heads, d_model, d_head]
        self._W_Q = W_Q
        self._W_K = W_K
        self._W_V = W_V
        self._b_Q = b_Q
        self._b_K = b_K
        self._b_V = b_V

        if hasattr(self, "q") and self.q is not None and hasattr(self.q, "set_processed_weights"):
            self.q.set_processed_weights(weight=W_Q, bias=b_Q)
        if hasattr(self, "k") and self.k is not None and hasattr(self.k, "set_processed_weights"):
            self.k.set_processed_weights(weight=W_K, bias=b_K)
        if hasattr(self, "v") and self.v is not None and hasattr(self.v, "set_processed_weights"):
            self.v.set_processed_weights(weight=W_V, bias=b_V)
        if hasattr(self, "o") and self.o is not None and hasattr(self.o, "set_processed_weights"):
            self.o.set_processed_weights(weight=W_O, bias=b_O)

    def process_weights(
        self,
        fold_ln: bool = False,
        center_writing_weights: bool = False,
        center_unembed: bool = False,
        fold_value_biases: bool = False,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process QKV weights according to GPT2 pretrained logic.

        Ports the weight processing from transformer_lens.pretrained.weight_conversions.gpt2
        to work with the architecture adapter.
        """
        import einops

        original_component = self.original_component
        if original_component is None:
            return

        # Get the combined QKV weight and bias from the original component
        if hasattr(original_component, "c_attn"):
            c_attn = cast(nn.Module, original_component.c_attn)
            qkv_weight = c_attn.weight  # Shape: [d_model, 3*d_model]
            qkv_bias = c_attn.bias  # Shape: [3*n_heads*d_head]
        else:
            # Try to get from submodules mapping
            qkv_submodule = None
            for name, module in self.submodules.items():
                if hasattr(module, "name") and module.name == "c_attn":
                    qkv_submodule = getattr(original_component, module.name, None)
                    break

            if qkv_submodule is None:
                return

            qkv_weight = cast(torch.Tensor, qkv_submodule.weight)
            qkv_bias = cast(torch.Tensor, qkv_submodule.bias)

        # Split QKV weights: [d_model, 3*d_model] -> 3 x [d_model, d_model]
        W_Q, W_K, W_V = torch.tensor_split(cast(torch.Tensor, qkv_weight), 3, dim=1)

        # Rearrange Q, K, V weights following GPT2 pretrained logic
        # "m (i h)->i m h" where m=d_model, i=n_heads, h=d_head
        assert self.config is not None
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=self.config.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=self.config.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=self.config.n_heads)

        # Process QKV bias following GPT2 pretrained logic
        qkv_bias_tensor = cast(torch.Tensor, qkv_bias)
        qkv_bias = einops.rearrange(
            qkv_bias_tensor,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=self.config.n_heads,
            head=self.config.d_head,
        )
        b_Q, b_K, b_V = (
            cast(torch.Tensor, qkv_bias[0]),
            cast(torch.Tensor, qkv_bias[1]),
            cast(torch.Tensor, qkv_bias[2]),
        )

        # Process output projection weight if it exists
        W_O = None
        b_O = None
        if hasattr(original_component, "c_proj"):
            c_proj = cast(nn.Module, original_component.c_proj)
            W_O = cast(torch.Tensor, c_proj.weight)
            b_O = cast(torch.Tensor, c_proj.bias)
            # Rearrange W_O following GPT2 pretrained logic: "(i h) m->i h m"
            W_O = einops.rearrange(W_O, "(i h) m->i h m", i=self.config.n_heads)
        else:
            # Try to get from submodules mapping
            for name, module in self.submodules.items():
                if hasattr(module, "name") and module.name == "c_proj":
                    proj_submodule = getattr(original_component, module.name, None)
                    if proj_submodule is not None:
                        W_O = proj_submodule.weight
                        b_O = proj_submodule.bias
                        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=self.config.n_heads)
                    break

        # Store processed weights in TransformerLens format
        self._processed_weights = {
            "W_Q": W_Q,
            "W_K": W_K,
            "W_V": W_V,
            "b_Q": b_Q,
            "b_K": b_K,
            "b_V": b_V,
        }

        if W_O is not None:
            self._processed_weights["W_O"] = W_O
        if b_O is not None:
            self._processed_weights["b_O"] = b_O

    def get_processed_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the processed weights in TransformerLens format.

        Returns:
            Dictionary mapping TransformerLens parameter names to processed tensors
        """
        if self._processed_weights is None:
            # If weights haven't been processed, return empty dict
            return {}

        return self._processed_weights.copy()

    def get_expected_parameter_names(self, prefix: str = "") -> list[str]:
        """Get the expected TransformerLens parameter names for this QKV attention component.

        Args:
            prefix: Prefix to add to parameter names (e.g., "blocks.0")

        Returns:
            List of expected parameter names in TransformerLens format
        """
        # QKV attention components always have Q, K, V weights and biases, and output projection
        base_names = ["W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O", "b_O"]

        if prefix:
            return [f"{prefix}.{name}" for name in base_names]
        else:
            return base_names

    def custom_weight_processing(
        self, hf_state_dict: Dict[str, torch.Tensor], component_prefix: str, **processing_kwargs
    ) -> Dict[str, torch.Tensor]:
        """Custom weight processing for QKV attention - handles QKV splitting.

        Args:
            hf_state_dict: Raw HuggingFace state dict
            component_prefix: Prefix for this component's weights (e.g., "transformer.h.0.attn")
            **processing_kwargs: Additional processing arguments

        Returns:
            Dictionary of processed weights for Q, K, V components
        """
        processed_weights = {}

        # Handle QKV weight splitting
        qkv_weight_key = f"{component_prefix}.c_attn.weight"
        qkv_bias_key = f"{component_prefix}.c_attn.bias"

        if qkv_weight_key in hf_state_dict:
            qkv_weight = hf_state_dict[qkv_weight_key]
            # Split into Q, K, V (assuming equal sizes)
            d_model = qkv_weight.shape[0]
            split_size = qkv_weight.shape[1] // 3

            q_weight = qkv_weight[:, :split_size]
            k_weight = qkv_weight[:, split_size : 2 * split_size]
            v_weight = qkv_weight[:, 2 * split_size :]

            # Rearrange for attention heads
            import einops

            assert self.config is not None
            n_heads = self.config.n_heads
            d_head = self.config.d_head

            processed_weights["W_Q"] = einops.rearrange(
                q_weight,
                "d_model (n_heads d_head) -> n_heads d_model d_head",
                n_heads=n_heads,
                d_head=d_head,
            )
            processed_weights["W_K"] = einops.rearrange(
                k_weight,
                "d_model (n_heads d_head) -> n_heads d_model d_head",
                n_heads=n_heads,
                d_head=d_head,
            )
            processed_weights["W_V"] = einops.rearrange(
                v_weight,
                "d_model (n_heads d_head) -> n_heads d_model d_head",
                n_heads=n_heads,
                d_head=d_head,
            )

        if qkv_bias_key in hf_state_dict:
            qkv_bias = hf_state_dict[qkv_bias_key]
            split_size = qkv_bias.shape[0] // 3

            q_bias = qkv_bias[:split_size]
            k_bias = qkv_bias[split_size : 2 * split_size]
            v_bias = qkv_bias[2 * split_size :]

            # Rearrange bias for attention heads
            import einops

            assert self.config is not None
            n_heads = self.config.n_heads
            d_head = self.config.d_head

            processed_weights["b_Q"] = einops.rearrange(
                q_bias, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads, d_head=d_head
            )
            processed_weights["b_K"] = einops.rearrange(
                k_bias, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads, d_head=d_head
            )
            processed_weights["b_V"] = einops.rearrange(
                v_bias, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads, d_head=d_head
            )

        # Handle output projection
        out_weight_key = f"{component_prefix}.c_proj.weight"
        out_bias_key = f"{component_prefix}.c_proj.bias"

        if out_weight_key in hf_state_dict:
            out_weight = hf_state_dict[out_weight_key]
            processed_weights["W_O"] = einops.rearrange(
                out_weight,
                "(n_heads d_head) d_model -> n_heads d_head d_model",
                n_heads=n_heads,
                d_head=d_head,
            )

        if out_bias_key in hf_state_dict:
            processed_weights["b_O"] = hf_state_dict[out_bias_key]

        return processed_weights

    def _extract_hooked_transformer_weights(self) -> None:
        """Extract weights in HookedTransformer format for exact compatibility."""
        # If weights were already set via set_processed_weights (e.g., from architecture adapter),
        # use those weights directly
        if (
            hasattr(self, "_processed_W_Q")
            and hasattr(self, "_processed_W_K")
            and hasattr(self, "_processed_W_V")
        ):
            # Use the processed weights directly as tensors
            # We don't need nn.Parameter because the real solution for backward hook parity
            # is calling ln1 three times (which we do below), not creating separate Parameters
            self._W_Q = self._processed_W_Q
            self._W_K = self._processed_W_K
            self._W_V = self._processed_W_V
            self._b_Q = self._processed_b_Q
            self._b_K = self._processed_b_K
            self._b_V = self._processed_b_V

            if hasattr(self, "_processed_W_O") and self._processed_W_O is not None:
                # W_O is already in TL format [n_heads, d_head, d_model] from bridge processing
                self._W_O = self._processed_W_O
            else:
                # Set to None explicitly if _processed_W_O doesn't exist
                self._W_O = None

            if hasattr(self, "_processed_b_O") and self._processed_b_O is not None:
                self._b_O = self._processed_b_O
            else:
                self._b_O = None

            self._hooked_weights_extracted = True
            return

        # Use cached reference model if available
        try:
            if hasattr(self, "_reference_model") and self._reference_model is not None:
                reference_model = self._reference_model
                layer_num = getattr(self, "_layer_idx", 0)
                reference_attn = reference_model.blocks[layer_num].attn

                self._W_Q = reference_attn.W_Q.clone()  # type: ignore[union-attr, operator]
                self._W_K = reference_attn.W_K.clone()  # type: ignore[union-attr, operator]
                self._W_V = reference_attn.W_V.clone()  # type: ignore[union-attr, operator]
                self._b_Q = reference_attn.b_Q.clone()  # type: ignore[union-attr, operator]
                self._b_K = reference_attn.b_K.clone()  # type: ignore[union-attr, operator]
                self._b_V = reference_attn.b_V.clone()  # type: ignore[union-attr, operator]

                if hasattr(reference_attn, "W_O"):
                    self._W_O = reference_attn.W_O.clone()  # type: ignore[operator]
                if hasattr(reference_attn, "b_O"):
                    self._b_O = reference_attn.b_O.clone()  # type: ignore[operator]

                self._hooked_weights_extracted = True
                self._reference_model = None
                return
        except Exception:
            pass

        # Fallback: Load a new reference model (expensive, rarely used)
        # Skip if _processed_weights is None (indicates no_processing=True was used)
        if self._processed_weights is None:
            return

        try:
            from transformer_lens import HookedTransformer

            model_name = getattr(self.config, "model_name", "gpt2")
            device = next(self.parameters()).device if list(self.parameters()) else "cpu"

            reference_model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
                refactor_factored_attn_matrices=False,
            )

            layer_num = 0
            current = self
            while hasattr(current, "parent") and current.parent is not None:
                parent = current.parent
                if hasattr(parent, "blocks"):
                    for i, block in enumerate(parent.blocks):
                        if hasattr(block, "attn") and block.attn is self:
                            layer_num = i
                            break
                    break
                current = parent

            reference_attn = reference_model.blocks[layer_num].attn

            self._W_Q = reference_attn.W_Q.clone()  # type: ignore[union-attr, operator]
            self._W_K = reference_attn.W_K.clone()  # type: ignore[union-attr, operator]
            self._W_V = reference_attn.W_V.clone()  # type: ignore[union-attr, operator]
            self._b_Q = reference_attn.b_Q.clone()  # type: ignore[union-attr, operator]
            self._b_K = reference_attn.b_K.clone()  # type: ignore[union-attr, operator]
            self._b_V = reference_attn.b_V.clone()  # type: ignore[union-attr, operator]

            if hasattr(reference_attn, "W_O"):
                self._W_O = reference_attn.W_O.clone()  # type: ignore[operator]
            if hasattr(reference_attn, "b_O"):
                self._b_O = reference_attn.b_O.clone()  # type: ignore[operator]

            del reference_model
            self._hooked_weights_extracted = True
            return
        except Exception:
            pass

        if self._processed_weights is None:
            try:
                self.process_weights(
                    fold_ln=True,
                    center_writing_weights=True,
                    center_unembed=True,
                    fold_value_biases=True,
                    refactor_factored_attn_matrices=False,
                )
            except Exception as e:
                print(f"⚠️  Failed to process weights manually: {e}")

        if self._processed_weights is not None:
            self._W_Q = self._processed_weights["W_Q"]
            self._W_K = self._processed_weights["W_K"]
            self._W_V = self._processed_weights["W_V"]
            self._b_Q = self._processed_weights["b_Q"]
            self._b_K = self._processed_weights["b_K"]
            self._b_V = self._processed_weights["b_V"]

            if "W_O" in self._processed_weights:
                self._W_O = self._processed_weights["W_O"]
            if "b_O" in self._processed_weights:
                self._b_O = self._processed_weights["b_O"]

            print(f"✅ Extracted HookedTransformer weights from processed weights")
            self._hooked_weights_extracted = True
        else:
            print(f"⚠️  Unable to extract HookedTransformer weights for {self.name}")
            print("Will attempt to use original component computation")
            self._hooked_weights_extracted = False

    def _load_reference_weights(self, reference_attn) -> None:
        """Load weights directly from a reference HookedTransformer attention component.

        Args:
            reference_attn: The HookedTransformer attention component to copy weights from
        """
        print(f"Loading reference weights for layer attention...")

        # Store the reference weights directly
        self._W_Q = reference_attn.W_Q.clone()
        self._W_K = reference_attn.W_K.clone()
        self._W_V = reference_attn.W_V.clone()
        self._b_Q = reference_attn.b_Q.clone()
        self._b_K = reference_attn.b_K.clone()
        self._b_V = reference_attn.b_V.clone()

        if hasattr(reference_attn, "W_O"):
            self._W_O = reference_attn.W_O.clone()
        if hasattr(reference_attn, "b_O"):
            self._b_O = reference_attn.b_O.clone()

        # Mark as extracted
        self._hooked_weights_extracted = True

        print(f"✅ Loaded reference weights with shapes:")
        print(f"  W_V: {self._W_V.shape}")
        print(f"  W_Q: {self._W_Q.shape}")
        print(f"  W_K: {self._W_K.shape}")
