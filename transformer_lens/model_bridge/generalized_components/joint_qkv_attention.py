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

        # Store processed weights after weight processing
        self._processed_weights: Optional[Dict[str, torch.Tensor]] = None

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
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            # Handle attention mask shape mismatch - slice to match sequence length
            if attention_mask.shape[-1] != seq_len:
                # Slice the attention mask to match the sequence length
                attention_mask = attention_mask[..., :seq_len]
            if attention_mask.shape[-2] != seq_len:
                attention_mask = attention_mask[..., :seq_len, :]
            attn_scores = attn_scores + attention_mask

        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        if hasattr(original_component, "attn_dropout"):
            attn_weights = original_component.attn_dropout(attn_weights)  # type: ignore[operator]

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to original format
        final_hidden_size: int = num_heads * head_dim
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, final_hidden_size)
        )

        # Apply output projection - use functional linear if available
        if hasattr(original_component, 'c_proj'):
            attn_output = self._apply_output_projection_with_functional_linear(attn_output)
        elif hasattr(self, "o") and self.o is not None:
            attn_output = self.o(attn_output)

        # Return format should match what GPT2Block expects (exactly 2 values)
        # The GPT2Block handles past_key_value separately
        return (attn_output, attn_weights)  # (output, weights)

    def _apply_qkv_projection_with_functional_linear(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if hasattr(original_component, 'c_attn'):
            qkv_weight = original_component.c_attn.weight  # Shape: [d_model, 3*d_model]
            qkv_bias = original_component.c_attn.bias      # Shape: [3*d_model]
        else:
            raise AttributeError("Original component doesn't have c_attn attribute for QKV projection")

        batch_size, seq_len, d_model = hidden_states.shape

        # Apply QKV projection using torch.nn.functional.linear
        # Note: torch.nn.functional.linear expects weight to be [output_features, input_features]
        # but HuggingFace stores it as [input_features, output_features], so we transpose
        qkv = torch.nn.functional.linear(hidden_states, qkv_weight.T, qkv_bias)

        # Split into Q, K, V - reshape to separate the 3 components
        qkv = qkv.view(batch_size, seq_len, 3, d_model)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        return q, k, v

    def _apply_output_projection_with_functional_linear(self, attn_output: torch.Tensor) -> torch.Tensor:
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
        if hasattr(original_component, 'c_proj'):
            proj_weight = original_component.c_proj.weight  # Shape: [d_model, d_model]
            proj_bias = original_component.c_proj.bias      # Shape: [d_model]
        else:
            # If no output projection, return the input unchanged
            return attn_output

        # Apply output projection using torch.nn.functional.linear
        # Note: torch.nn.functional.linear expects weight to be [output_features, input_features]
        # but HuggingFace stores it as [input_features, output_features], so we transpose
        output = torch.nn.functional.linear(attn_output, proj_weight.T, proj_bias)

        return output

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
        if hasattr(original_component, 'c_attn'):
            qkv_weight = original_component.c_attn.weight  # Shape: [d_model, 3*d_model]
            qkv_bias = original_component.c_attn.bias      # Shape: [3*n_heads*d_head]
        else:
            # Try to get from submodules mapping
            qkv_submodule = None
            for name, module in self.submodules.items():
                if hasattr(module, 'name') and module.name == "c_attn":
                    qkv_submodule = getattr(original_component, module.name, None)
                    break

            if qkv_submodule is None:
                return

            qkv_weight = qkv_submodule.weight
            qkv_bias = qkv_submodule.bias

        # Split QKV weights: [d_model, 3*d_model] -> 3 x [d_model, d_model]
        W_Q, W_K, W_V = torch.tensor_split(qkv_weight, 3, dim=1)

        # Rearrange Q, K, V weights following GPT2 pretrained logic
        # "m (i h)->i m h" where m=d_model, i=n_heads, h=d_head
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=self.config.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=self.config.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=self.config.n_heads)

        # Process QKV bias following GPT2 pretrained logic
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=self.config.n_heads,
            head=self.config.d_head,
        )
        b_Q, b_K, b_V = qkv_bias[0], qkv_bias[1], qkv_bias[2]

        # Process output projection weight if it exists
        W_O = None
        b_O = None
        if hasattr(original_component, 'c_proj'):
            W_O = original_component.c_proj.weight
            b_O = original_component.c_proj.bias
            # Rearrange W_O following GPT2 pretrained logic: "(i h) m->i h m"
            W_O = einops.rearrange(W_O, "(i h) m->i h m", i=self.config.n_heads)
        else:
            # Try to get from submodules mapping
            for name, module in self.submodules.items():
                if hasattr(module, 'name') and module.name == "c_proj":
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
        self,
        hf_state_dict: Dict[str, torch.Tensor],
        component_prefix: str,
        **processing_kwargs
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
            k_weight = qkv_weight[:, split_size:2*split_size]
            v_weight = qkv_weight[:, 2*split_size:]

            # Rearrange for attention heads
            import einops
            n_heads = self.config.n_heads
            d_head = self.config.d_head

            processed_weights["W_Q"] = einops.rearrange(
                q_weight, "d_model (n_heads d_head) -> n_heads d_model d_head",
                n_heads=n_heads, d_head=d_head
            )
            processed_weights["W_K"] = einops.rearrange(
                k_weight, "d_model (n_heads d_head) -> n_heads d_model d_head",
                n_heads=n_heads, d_head=d_head
            )
            processed_weights["W_V"] = einops.rearrange(
                v_weight, "d_model (n_heads d_head) -> n_heads d_model d_head",
                n_heads=n_heads, d_head=d_head
            )

        if qkv_bias_key in hf_state_dict:
            qkv_bias = hf_state_dict[qkv_bias_key]
            split_size = qkv_bias.shape[0] // 3

            q_bias = qkv_bias[:split_size]
            k_bias = qkv_bias[split_size:2*split_size]
            v_bias = qkv_bias[2*split_size:]

            # Rearrange bias for attention heads
            import einops
            n_heads = self.config.n_heads
            d_head = self.config.d_head

            processed_weights["b_Q"] = einops.rearrange(
                q_bias, "(n_heads d_head) -> n_heads d_head",
                n_heads=n_heads, d_head=d_head
            )
            processed_weights["b_K"] = einops.rearrange(
                k_bias, "(n_heads d_head) -> n_heads d_head",
                n_heads=n_heads, d_head=d_head
            )
            processed_weights["b_V"] = einops.rearrange(
                v_bias, "(n_heads d_head) -> n_heads d_head",
                n_heads=n_heads, d_head=d_head
            )

        # Handle output projection
        out_weight_key = f"{component_prefix}.c_proj.weight"
        out_bias_key = f"{component_prefix}.c_proj.bias"

        if out_weight_key in hf_state_dict:
            out_weight = hf_state_dict[out_weight_key]
            processed_weights["W_O"] = einops.rearrange(
                out_weight, "(n_heads d_head) d_model -> n_heads d_head d_model",
                n_heads=n_heads, d_head=d_head
            )

        if out_bias_key in hf_state_dict:
            processed_weights["b_O"] = hf_state_dict[out_bias_key]

        return processed_weights
