"""Attention bridge component.

This module contains the bridge component for attention layers.
"""

from typing import Any, Dict, Mapping, Optional

import torch

from transformer_lens.conversion_utils.conversion_steps.attention_auto_conversion import (
    AttentionAutoConversion,
)
from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class AttentionBridge(GeneralizedComponent):
    """Bridge component for attention layers.

    This component handles the conversion between Hugging Face attention layers
    and TransformerLens attention components.
    """

    hook_aliases = {
        "hook_result": "hook_out",
        "hook_q": "q.hook_out",
        "hook_k": "k.hook_out",
        "hook_v": "v.hook_out",
        "hook_z": "o.hook_in",
    }

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
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        conversion_rule: Optional[BaseHookConversion] = None,
        pattern_conversion_rule: Optional[BaseHookConversion] = None,
        maintain_native_attention: bool = False,
        requires_position_embeddings: bool = False,
        requires_attention_mask: bool = False,
    ):
        """Initialize the attention bridge.

        Args:
            name: The name of this component
            config: Model configuration (required for auto-conversion detection)
            submodules: Dictionary of submodules to register (e.g., q_proj, k_proj, etc.)
            conversion_rule: Optional conversion rule. If None, AttentionAutoConversion will be used
            pattern_conversion_rule: Optional conversion rule for attention patterns. If None,
                                   uses AttentionPatternConversion to ensure [n_heads, pos, pos] shape
            maintain_native_attention: If True, preserve the original HF attention implementation
                                      without wrapping. Use for models with custom attention
                                      (e.g., attention sinks, specialized RoPE). Defaults to False.
            requires_position_embeddings: If True, this attention requires position_embeddings argument
                                        (e.g., Gemma-3 with dual RoPE). Defaults to False.
            requires_attention_mask: If True, this attention requires attention_mask argument
                                    (e.g., GPTNeoX/Pythia). Defaults to False.
        """
        # Set up conversion rule - use AttentionAutoConversion if None
        if conversion_rule is None:
            conversion_rule = AttentionAutoConversion(config)

        super().__init__(
            name, config=config, submodules=submodules or {}, conversion_rule=conversion_rule
        )

        # Create only the hook points that are actually used for attention processing
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_hidden_states = HookPoint()

        # Add rotary embedding hooks if using rotary positional embeddings
        if (
            hasattr(config, "positional_embedding_type")
            and config.positional_embedding_type == "rotary"
        ):
            self.hook_rot_k = HookPoint()
            self.hook_rot_q = HookPoint()

        # Apply conversion rule to attention-specific hooks
        self.hook_hidden_states.hook_conversion = conversion_rule

        # Set up pattern conversion rule if provided
        if pattern_conversion_rule is not None:
            self.hook_pattern.hook_conversion = pattern_conversion_rule

        # Store intermediate values for pattern creation
        self._attn_scores = None
        self._pattern = None

        # Flag to track if HF attention forward has been wrapped for no_processing mode
        self._hf_forward_wrapped = False

        # Store whether to maintain native attention implementation
        self.maintain_native_attention = maintain_native_attention

        # Store input requirements for testing
        self.requires_position_embeddings = requires_position_embeddings
        self.requires_attention_mask = requires_attention_mask

    def setup_no_processing_hooks(self) -> None:
        """Setup hooks for no_processing mode.

        In no_processing mode, we use a simplified forward pass that just wraps
        the original component with hook_in and hook_out.
        """
        if self._hf_forward_wrapped:
            return  # Already set up

        # Setup hook_z reshaping if we have an 'o' submodule
        if hasattr(self, "o") and self.o is not None and hasattr(self.config, "n_heads"):
            self._setup_hook_z_reshape()

        self._hf_forward_wrapped = True

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Get random inputs for testing this attention component.

        Generates appropriate inputs based on the attention's requirements
        (position_embeddings, attention_mask, etc.).

        Args:
            batch_size: Batch size for the test inputs
            seq_len: Sequence length for the test inputs
            device: Device to create tensors on (defaults to CPU)
            dtype: Dtype for generated tensors (defaults to float32)

        Returns:
            Dictionary of keyword arguments to pass to forward()
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        # Start with base hidden_states
        d_model = self.config.d_model if self.config and hasattr(self.config, "d_model") else 768
        inputs: Dict[str, Any] = {
            "hidden_states": torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
        }

        # Add position_embeddings if required (e.g., Gemma-3)
        if self.requires_position_embeddings:
            # Generate dummy cos/sin tensors for testing
            # Note: We use a fallback approach instead of calling rotary_emb directly
            # because different models have different rotary_emb interfaces and shapes

            # Try to get head dimension from config (different models use different attribute names)
            if self.config:
                if hasattr(self.config, "d_head"):
                    d_head = self.config.d_head
                elif hasattr(self.config, "head_dim"):
                    d_head = self.config.head_dim
                else:
                    d_head = 64  # fallback
            else:
                d_head = 64

            # Calculate rotary dimension (some models use partial rotary)
            rotary_pct = getattr(self.config, "rotary_pct", 1.0) if self.config else 1.0
            rotary_ndims = int(rotary_pct * d_head)
            # Create dummy rotary embeddings: shape [1, seq_len, rotary_ndims]
            # Note: First dimension is 1 (not batch_size) because RoPE embeddings
            # are typically broadcast across the batch dimension
            cos = torch.ones(1, seq_len, rotary_ndims, device=device, dtype=dtype)
            sin = torch.zeros(1, seq_len, rotary_ndims, device=device, dtype=dtype)
            inputs["position_embeddings"] = (cos, sin)

        # Add attention_mask if required (e.g., GPTNeoX/Pythia)
        if self.requires_attention_mask:
            # Generate a causal attention mask (lower triangular matrix)
            # Shape: [batch_size, seq_len] with 1s for allowed positions
            # For causal masking, we want to attend to all previous positions
            inputs["attention_mask"] = torch.ones(batch_size, seq_len, device=device)

        return inputs

    def _setup_hook_z_reshape(self) -> None:
        """Setup hook_z (o.hook_in) to reshape from [batch, seq, d_model] to [batch, seq, n_heads, d_head]."""
        from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
            BaseHookConversion,
        )

        class ReshapeForAttentionHeads(BaseHookConversion):
            """Reshape tensors to split attention heads for hook_z compatibility."""

            def __init__(self, n_heads: int, d_head: int):
                super().__init__()
                self.n_heads = n_heads
                self.d_head = d_head

            def handle_conversion(self, input_value, *full_context):
                """Convert from [batch, seq, d_model] to [batch, seq, n_heads, d_head]."""
                if len(input_value.shape) == 3:
                    b, s, d = input_value.shape
                    if d == self.n_heads * self.d_head:
                        return input_value.view(b, s, self.n_heads, self.d_head)
                return input_value

            def revert(self, input_value, *full_context):
                """Revert from [batch, seq, n_heads, d_head] to [batch, seq, d_model]."""
                if len(input_value.shape) == 4:
                    b, s, n_h, d_h = input_value.shape
                    if n_h == self.n_heads and d_h == self.d_head:
                        return input_value.view(b, s, n_h * d_h)
                return input_value

        # Get dimensions
        if self.config is None:
            raise RuntimeError(f"Config not set for {self.name}")
        n_heads = self.config.n_heads if hasattr(self.config, "n_heads") else self.config.n_head
        d_model = self.config.d_model if hasattr(self.config, "d_model") else self.config.n_embd
        d_head = d_model // n_heads

        # Apply conversion to o.hook_in (which is aliased as hook_z)
        reshape_conv = ReshapeForAttentionHeads(n_heads, d_head)
        self.o.hook_in.hook_conversion = reshape_conv

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Simplified forward pass - minimal wrapping around original component.

        This does minimal wrapping: hook_in → delegate to HF → hook_out.
        This ensures we match HuggingFace's exact output without complex intermediate processing.

        Args:
            *args: Input arguments to pass to the original component
            **kwargs: Input keyword arguments to pass to the original component

        Returns:
            The output from the original component, with only input/output hooks applied
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Get the target dtype from the original component's parameters
        target_dtype = None
        try:
            target_dtype = next(self.original_component.parameters()).dtype
        except StopIteration:
            # Component has no parameters, keep inputs as-is
            pass

        # Apply input hook - check for various input formats
        if "query_input" in kwargs:
            # Some models use query_input parameter
            hooked = self.hook_in(kwargs["query_input"])
            # Cast to target dtype if needed
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            kwargs["query_input"] = hooked
        elif "hidden_states" in kwargs:
            # Most models use hidden_states parameter
            hooked = self.hook_in(kwargs["hidden_states"])
            # Cast to target dtype if needed
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            kwargs["hidden_states"] = hooked
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            # hidden_states passed as first positional argument
            hooked = self.hook_in(args[0])
            # Cast to target dtype if needed
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            args = (hooked,) + args[1:]
            # Move it to kwargs for HF compatibility
            kwargs["hidden_states"] = args[0]
            args = args[1:]

        # Forward through original component
        output = self.original_component(*args, **kwargs)

        # Apply hook_out to the output
        # HF attention returns a tuple: (hidden_states, attention_weights, ...)
        # We apply hooks to the first element and preserve the rest
        if isinstance(output, tuple) and len(output) >= 2:
            # Return with hooked first element: (hooked_hidden_states, attention_weights, ...)
            output = (self.hook_out(output[0]), output[1])
        elif isinstance(output, tuple) and len(output) == 1:
            # Single element tuple
            output = (self.hook_out(output[0]),)
        else:
            # Not a tuple, just hook the output directly
            output = self.hook_out(output)

        return output

    def set_processed_weights(self, weights: Mapping[str, torch.Tensor | None]) -> None:
        """Set the processed weights by delegating to LinearBridge submodules.

        This uses LinearBridge's set_processed_weights method for Q/K/V/O submodules,
        so when forward() delegates to original_component, it uses the processed weights.

        The weights should already be in the correct 2D format from weight processing.

        Args:
            weights: Dictionary containing processed weight tensors with keys:
                - "W_Q": Query weight tensor (already in 2D format)
                - "W_K": Key weight tensor (already in 2D format)
                - "W_V": Value weight tensor (already in 2D format)
                - "W_O": Output projection weight tensor (already in 2D format)
                - "b_Q": Query bias tensor (optional)
                - "b_K": Key bias tensor (optional)
                - "b_V": Value bias tensor (optional)
                - "b_O": Output bias tensor (optional)
        """
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")

        W_Q = weights.get("W_Q")
        W_K = weights.get("W_K")
        W_V = weights.get("W_V")
        W_O = weights.get("W_O")
        if W_Q is None or W_K is None or W_V is None or W_O is None:
            raise ValueError(
                "Processed attention weights must include W_Q, W_K, W_V, and W_O tensors."
            )
        b_Q = weights.get("b_Q")
        b_K = weights.get("b_K")
        b_V = weights.get("b_V")
        b_O = weights.get("b_O")

        # Get Q/K/V/O submodules (LinearBridge instances)
        q_module = getattr(self, "q", None)
        k_module = getattr(self, "k", None)
        v_module = getattr(self, "v", None)
        o_module = getattr(self, "o", None)

        # Use LinearBridge's set_processed_weights for each submodule
        # Weights should already be in 2D format [in, out] from weight processing
        if q_module and hasattr(q_module, "set_processed_weights"):
            q_module.set_processed_weights({"weight": W_Q, "bias": b_Q})

        if k_module and hasattr(k_module, "set_processed_weights"):
            k_module.set_processed_weights({"weight": W_K, "bias": b_K})

        if v_module and hasattr(v_module, "set_processed_weights"):
            v_module.set_processed_weights({"weight": W_V, "bias": b_V})

        if o_module and hasattr(o_module, "set_processed_weights"):
            o_module.set_processed_weights({"weight": W_O, "bias": b_O})

    def _forward_with_processed_weights(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Direct implementation of reference model's attention computation with hooks."""
        # Extract input from args/kwargs
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            x = args[0]
        elif "hidden_states" in kwargs:
            x = kwargs["hidden_states"]
        else:
            raise ValueError("No valid input tensor found in args or kwargs")

        # Apply input hook
        x = self.hook_in(x)

        batch_size, seq_len, d_model = x.shape

        # Compute Q, K, V using TransformerLens format weights
        # W_Q shape: [n_heads, d_model, d_head], b_Q shape: [n_heads, d_head] or None
        # x shape: [batch, seq, d_model]
        q = torch.stack(
            [x @ self._processed_W_Q[h] for h in range(self._processed_W_Q.shape[0])], dim=2
        )
        if self._processed_b_Q is not None:
            q = q + self._processed_b_Q.unsqueeze(0).unsqueeze(0)

        k = torch.stack(
            [x @ self._processed_W_K[h] for h in range(self._processed_W_K.shape[0])], dim=2
        )
        if self._processed_b_K is not None:
            k = k + self._processed_b_K.unsqueeze(0).unsqueeze(0)

        v = torch.stack(
            [x @ self._processed_W_V[h] for h in range(self._processed_W_V.shape[0])], dim=2
        )
        if self._processed_b_V is not None:
            v = v + self._processed_b_V.unsqueeze(0).unsqueeze(0)

        # Apply hook for V if it exists (this is what gets ablated in the comparison script)
        # Check for hook_v (compatibility mode) or v.hook_out (new architecture)
        if hasattr(self, "v") and hasattr(self.v, "hook_out"):
            v = self.v.hook_out(v)
        elif "hook_v" in self.hook_aliases:
            # In compatibility mode, use the aliased hook_v
            # Temporarily disable warnings for this internal access
            original_disable_warnings = getattr(self, "disable_warnings", False)
            self.disable_warnings = True
            try:
                v = self.hook_v(v)
            finally:
                self.disable_warnings = original_disable_warnings

        # Transpose to [batch, n_heads, seq, d_head] for attention computation
        q = q.transpose(1, 2)  # [batch, n_heads, seq, d_head]
        k = k.transpose(1, 2)  # [batch, n_key_value_heads, seq, d_head]
        v = v.transpose(1, 2)  # [batch, n_key_value_heads, seq, d_head]

        # For GQA (Grouped Query Attention): expand K and V heads to match Q heads
        # Each key/value head is shared across n_heads // n_key_value_heads query heads
        n_heads_q = q.shape[1]
        n_heads_kv = k.shape[1]
        if n_heads_kv < n_heads_q:
            # GQA: repeat each K/V head to match the number of Q heads
            repeats = n_heads_q // n_heads_kv
            k = k.repeat_interleave(repeats, dim=1)  # [batch, n_heads, seq, d_head]
            v = v.repeat_interleave(repeats, dim=1)  # [batch, n_heads, seq, d_head]

        # Compute attention scores
        d_head = self._processed_W_Q.shape[-1]  # Get d_head from weight shape
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head**0.5)

        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # Apply attention scores hook (for compatibility with HookedTransformer)
        attn_scores = self.hook_attn_scores(attn_scores)

        # Apply softmax
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        # Apply pattern hook (for compatibility with HookedTransformer)
        attn_weights = self.hook_pattern(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # [batch, n_heads, seq, d_head]

        # Transpose back to [batch, seq, n_heads, d_head] for output projection
        attn_out = attn_out.transpose(1, 2)

        # Apply hook_z (o.hook_in) - this is the z tensor before output projection
        # In compatibility mode, this hook is aliased as "blocks.L.attn.hook_z"
        if hasattr(self, "o") and hasattr(self.o, "hook_in"):
            attn_out = self.o.hook_in(attn_out)

        # Apply output projection using TransformerLens format
        # attn_out: [batch, seq, n_heads, d_head], W_O: [n_heads, d_head, d_model]
        result = torch.stack(
            [
                attn_out[:, :, h, :] @ self._processed_W_O[h]
                for h in range(self._processed_W_O.shape[0])
            ],
            dim=2,
        ).sum(dim=2)

        # Add output bias if it exists (models like Gemma/LLaMA/Qwen don't have biases)
        if self._processed_b_O is not None:
            result = result + self._processed_b_O.unsqueeze(0).unsqueeze(0)

        # Apply output hook
        result = self.hook_out(result)

        # Return both result and attention weights to match HF's expected return format
        # The patched block forward expects (output, attn_weights)
        return (result, attn_weights)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get cached attention weights if available.

        Returns:
            Attention weights tensor or None if not cached
        """
        return getattr(self, "_cached_attention_weights", None)

    def get_attention_patterns(self) -> Optional[torch.Tensor]:
        """Get cached attention patterns if available.

        Returns:
            Attention patterns tensor or None if not cached
        """
        return getattr(self, "_cached_attention_patterns", None)

    def __repr__(self) -> str:
        """String representation of the AttentionBridge."""
        return f"AttentionBridge(name={self.name})"
