"""Attention bridge component.

This module contains the bridge component for attention layers.
"""
from typing import Any, Dict, Optional

import einops
import torch

from transformer_lens.conversion_utils.conversion_steps.attention_auto_conversion import (
    AttentionAutoConversion,
)
from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.utilities.hf_utils import get_rotary_pct_from_config


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
        conversion_rule: Optional[BaseTensorConversion] = None,
        pattern_conversion_rule: Optional[BaseTensorConversion] = None,
        maintain_native_attention: bool = False,
        requires_position_embeddings: bool = False,
        requires_attention_mask: bool = False,
        attention_mask_4d: bool = False,
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
            attention_mask_4d: If True, generate 4D attention_mask [batch, 1, tgt_len, src_len]
                             instead of 2D [batch, seq_len]. Required for OPT. Defaults to False.
        """
        if conversion_rule is None:
            conversion_rule = AttentionAutoConversion(config)
        super().__init__(
            name, config=config, submodules=submodules or {}, conversion_rule=conversion_rule
        )
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_hidden_states = HookPoint()
        if (
            hasattr(config, "positional_embedding_type")
            and config.positional_embedding_type == "rotary"
        ):
            self.hook_rot_k = HookPoint()
            self.hook_rot_q = HookPoint()
        self.hook_hidden_states.hook_conversion = conversion_rule
        if pattern_conversion_rule is not None:
            self.hook_pattern.hook_conversion = pattern_conversion_rule
        self._attn_scores = None
        self._pattern = None
        self._hf_forward_wrapped = False
        self.maintain_native_attention = maintain_native_attention
        self.requires_position_embeddings = requires_position_embeddings
        self.requires_attention_mask = requires_attention_mask
        self.attention_mask_4d = attention_mask_4d

    def setup_hook_compatibility(self) -> None:
        """Setup hook compatibility transformations to match HookedTransformer behavior.

        This sets up hook conversions that ensure Bridge hooks have the same shapes
        as HookedTransformer hooks. This includes reshaping Q/K/V/Z hooks from
        [batch, seq, d_model] to [batch, seq, n_heads, d_head] format.

        This is called during Bridge.__init__ and should always be run.
        Note: This method is idempotent - can be called multiple times safely.
        """
        if self._hf_forward_wrapped:
            return
        if hasattr(self.config, "n_heads"):
            self._setup_qkv_hook_reshaping()
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
        d_model = self.config.d_model if self.config and hasattr(self.config, "d_model") else 768
        inputs: Dict[str, Any] = {
            "hidden_states": torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
        }
        if self.requires_position_embeddings:
            if self.config:
                if hasattr(self.config, "d_head"):
                    d_head = self.config.d_head
                elif hasattr(self.config, "head_dim"):
                    d_head = self.config.head_dim
                else:
                    d_head = 64
            else:
                d_head = 64
            rotary_pct = get_rotary_pct_from_config(self.config)
            rotary_ndims = int(rotary_pct * d_head)
            cos = torch.ones(1, seq_len, rotary_ndims, device=device, dtype=dtype)
            sin = torch.zeros(1, seq_len, rotary_ndims, device=device, dtype=dtype)
            inputs["position_embeddings"] = (cos, sin)
        if self.requires_attention_mask:
            if self.attention_mask_4d:
                # Generate 4D attention mask [batch, 1, tgt_len, src_len] for models like OPT
                inputs["attention_mask"] = torch.ones(
                    batch_size, 1, seq_len, seq_len, device=device
                )
            else:
                # Generate 2D attention mask [batch, seq_len] for most models
                inputs["attention_mask"] = torch.ones(batch_size, seq_len, device=device)
        return inputs

    def _setup_qkv_hook_reshaping(self) -> None:
        """Setup hook reshaping for Q/K/V/Z to match HookedTransformer shapes.

        Reshapes hooks from [batch, seq, d_model] to [batch, seq, n_heads, d_head] format.
        For models with Grouped Query Attention (GQA), K and V use n_kv_heads instead of n_heads.

        Sets up conversions for:
        - q.hook_out (aliased as hook_q)
        - k.hook_out (aliased as hook_k) - uses n_kv_heads if GQA
        - v.hook_out (aliased as hook_v) - uses n_kv_heads if GQA
        - o.hook_in (aliased as hook_z)
        """

        class ReshapeForAttentionHeads(BaseTensorConversion):
            """Reshape tensors to split attention heads for Q/K/V/Z compatibility."""

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

        if self.config is None:
            raise RuntimeError(f"Config not set for {self.name}")

        # Get n_heads (try n_heads first, then n_head)
        if hasattr(self.config, "n_heads"):
            n_heads = self.config.n_heads
        elif hasattr(self.config, "n_head"):
            n_heads = self.config.n_head
        else:
            # Can't setup reshaping without knowing number of heads
            return

        # Get d_head (try d_head first, then compute from d_model or n_embd)
        if hasattr(self.config, "d_head"):
            d_head = self.config.d_head
        elif hasattr(self.config, "d_model"):
            d_head = self.config.d_model // n_heads
        elif hasattr(self.config, "n_embd"):
            d_head = self.config.n_embd // n_heads
        else:
            # Can't setup reshaping without knowing head dimension
            return
        n_kv_heads = n_heads
        if hasattr(self.config, "n_key_value_heads") and self.config.n_key_value_heads is not None:
            n_kv_heads = self.config.n_key_value_heads
        if hasattr(self, "q") and self.q is not None and hasattr(self.q, "hook_out"):
            q_reshape = ReshapeForAttentionHeads(n_heads, d_head)
            self.q.hook_out.hook_conversion = q_reshape
        if hasattr(self, "k") and self.k is not None and hasattr(self.k, "hook_out"):
            k_reshape = ReshapeForAttentionHeads(n_kv_heads, d_head)
            self.k.hook_out.hook_conversion = k_reshape
        if hasattr(self, "v") and self.v is not None and hasattr(self.v, "hook_out"):
            v_reshape = ReshapeForAttentionHeads(n_kv_heads, d_head)
            self.v.hook_out.hook_conversion = v_reshape
        if hasattr(self, "o") and self.o is not None and hasattr(self.o, "hook_in"):
            z_reshape = ReshapeForAttentionHeads(n_heads, d_head)
            self.o.hook_in.hook_conversion = z_reshape

        class TransposeRotaryHeads(BaseTensorConversion):
            """Transpose rotary hook tensors from HF format to HookedTransformer format."""

            def handle_conversion(self, input_value, *full_context):
                """Convert from [batch, n_heads, seq, d_head] to [batch, seq, n_heads, d_head]."""
                if len(input_value.shape) == 4:
                    return input_value.transpose(1, 2)
                return input_value

            def revert(self, input_value, *full_context):
                """Revert from [batch, seq, n_heads, d_head] to [batch, n_heads, seq, d_head]."""
                if len(input_value.shape) == 4:
                    return input_value.transpose(1, 2)
                return input_value

        if hasattr(self, "hook_rot_q"):
            self.hook_rot_q.hook_conversion = TransposeRotaryHeads()
        if hasattr(self, "hook_rot_k"):
            self.hook_rot_k.hook_conversion = TransposeRotaryHeads()

    def _setup_hook_z_reshape(self) -> None:
        """Backward compatibility alias for _setup_qkv_hook_reshaping."""
        self._setup_qkv_hook_reshaping()

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
        target_dtype = None
        try:
            target_dtype = next(self.original_component.parameters()).dtype
        except StopIteration:
            pass
        if "query_input" in kwargs:
            hooked = self.hook_in(kwargs["query_input"])
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            kwargs["query_input"] = hooked
        elif "hidden_states" in kwargs:
            hooked = self.hook_in(kwargs["hidden_states"])
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            kwargs["hidden_states"] = hooked
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked = self.hook_in(args[0])
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            args = (hooked,) + args[1:]
            kwargs["hidden_states"] = args[0]
            args = args[1:]
        output = self.original_component(*args, **kwargs)
        if isinstance(output, tuple) and len(output) >= 2:
            # output[0] is attention output
            # output[1] may be attention weights (pattern) or position_bias (T5)
            # Additional elements may include position_bias, attention weights, etc.
            attn_output = self.hook_out(output[0])
            second_element = output[1]

            # Fire hook_pattern if the second element is attention weights (4D tensor)
            # For T5, second element is position_bias which should be passed through
            if isinstance(second_element, torch.Tensor) and second_element.dim() == 4:
                # This looks like attention weights [batch, heads, seq, seq]
                second_element = self.hook_pattern(second_element)
                # Also store for potential hook_attn_scores (before softmax)
                # Note: Most HF implementations return post-softmax weights
                self.hook_attn_scores(second_element)

            # Preserve all output elements (important for T5 position_bias and other models)
            output = (attn_output, second_element) + output[2:]
        elif isinstance(output, tuple) and len(output) == 1:
            output = (self.hook_out(output[0]),)
        else:
            output = self.hook_out(output)
        return output

    @property
    def W_Q(self) -> torch.Tensor:
        """Get W_Q in 3D format [n_heads, d_model, d_head] from 2D linear bridge weight."""

        weight = (
            self.q.weight
        )  # 2D: [d_model, n_heads*d_head] for Conv1D or [n_heads*d_head, d_model] for Linear
        if weight.ndim == 2 and self.config is not None:
            n_heads = self.config.n_heads if hasattr(self.config, "n_heads") else self.config.n_head
            # Detect format based on weight shape
            # Linear format: [(n_heads*d_head), d_model]
            # Conv1D format: [d_model, (n_heads*d_head)]
            if weight.shape[0] % n_heads == 0:
                # Linear format - first dimension is (n_heads*d_head)
                return einops.rearrange(
                    weight, "(n_heads d_head) d_model -> n_heads d_model d_head", n_heads=n_heads
                )
            else:
                # Conv1D format - second dimension is (n_heads*d_head)
                return einops.rearrange(
                    weight, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
                )
        return weight

    @property
    def W_K(self) -> torch.Tensor:
        """Get W_K in 3D format [n_heads, d_model, d_head] from 2D linear bridge weight."""

        weight = self.k.weight
        if weight.ndim == 2 and self.config is not None:
            n_heads = (
                self.config.n_key_value_heads
                if hasattr(self.config, "n_key_value_heads") and self.config.n_key_value_heads
                else (
                    self.config.n_heads if hasattr(self.config, "n_heads") else self.config.n_head
                )
            )
            # Detect format based on weight shape
            # Linear format: [(n_heads*d_head), d_model]
            # Conv1D format: [d_model, (n_heads*d_head)]
            if weight.shape[0] % n_heads == 0:
                # Linear format - first dimension is (n_heads*d_head)
                return einops.rearrange(
                    weight, "(n_heads d_head) d_model -> n_heads d_model d_head", n_heads=n_heads
                )
            else:
                # Conv1D format - second dimension is (n_heads*d_head)
                return einops.rearrange(
                    weight, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
                )
        return weight

    @property
    def W_V(self) -> torch.Tensor:
        """Get W_V in 3D format [n_heads, d_model, d_head] from 2D linear bridge weight."""

        weight = self.v.weight
        if weight.ndim == 2 and self.config is not None:
            n_heads = (
                self.config.n_key_value_heads
                if hasattr(self.config, "n_key_value_heads") and self.config.n_key_value_heads
                else (
                    self.config.n_heads if hasattr(self.config, "n_heads") else self.config.n_head
                )
            )
            # Detect format based on weight shape
            # Linear format: [(n_heads*d_head), d_model]
            # Conv1D format: [d_model, (n_heads*d_head)]
            if weight.shape[0] % n_heads == 0:
                # Linear format - first dimension is (n_heads*d_head)
                return einops.rearrange(
                    weight, "(n_heads d_head) d_model -> n_heads d_model d_head", n_heads=n_heads
                )
            else:
                # Conv1D format - second dimension is (n_heads*d_head)
                return einops.rearrange(
                    weight, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
                )
        return weight

    @property
    def W_O(self) -> torch.Tensor:
        """Get W_O in 3D format [n_heads, d_head, d_model] from 2D linear bridge weight."""

        weight = self.o.weight
        if weight.ndim == 2 and self.config is not None:
            n_heads = self.config.n_heads if hasattr(self.config, "n_heads") else self.config.n_head
            if weight.shape[0] == n_heads * (
                weight.shape[1] // n_heads
                if weight.shape[1] % n_heads == 0
                else weight.shape[0] // n_heads
            ):
                return einops.rearrange(
                    weight, "(n_heads d_head) d_model -> n_heads d_head d_model", n_heads=n_heads
                )
            else:
                return einops.rearrange(
                    weight.T, "(n_heads d_head) d_model -> n_heads d_head d_model", n_heads=n_heads
                )
        return weight

    @property
    def b_Q(self) -> Optional[torch.Tensor]:
        """Get b_Q in 2D format [n_heads, d_head] from 1D linear bridge bias."""

        bias = self.q.bias
        if bias is not None and bias.ndim == 1 and self.config is not None:
            n_heads = self.config.n_heads if hasattr(self.config, "n_heads") else self.config.n_head
            return einops.rearrange(bias, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
        return bias

    @property
    def b_K(self) -> Optional[torch.Tensor]:
        """Get b_K in 2D format [n_heads, d_head] from 1D linear bridge bias."""

        bias = self.k.bias
        if bias is not None and bias.ndim == 1 and self.config is not None:
            n_heads = (
                self.config.n_key_value_heads
                if hasattr(self.config, "n_key_value_heads") and self.config.n_key_value_heads
                else (
                    self.config.n_heads if hasattr(self.config, "n_heads") else self.config.n_head
                )
            )
            return einops.rearrange(bias, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
        return bias

    @property
    def b_V(self) -> Optional[torch.Tensor]:
        """Get b_V in 2D format [n_heads, d_head] from 1D linear bridge bias."""

        bias = self.v.bias
        if bias is not None and bias.ndim == 1 and self.config is not None:
            n_heads = (
                self.config.n_key_value_heads
                if hasattr(self.config, "n_key_value_heads") and self.config.n_key_value_heads
                else (
                    self.config.n_heads if hasattr(self.config, "n_heads") else self.config.n_head
                )
            )
            return einops.rearrange(bias, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
        return bias

    @property
    def b_O(self) -> Optional[torch.Tensor]:
        """Get b_O bias from linear bridge."""
        return self.o.bias
