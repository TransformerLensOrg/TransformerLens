"""Attention bridge component.

This module contains the bridge component for attention layers.
"""
import logging
from typing import Any, Dict, Optional

import einops
import torch

logger = logging.getLogger(__name__)

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
        self._layer_idx: Optional[int] = None

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set original component and capture layer index for KV caching."""
        super().set_original_component(original_component)
        layer_idx_raw = getattr(original_component, "layer_idx", None)
        if layer_idx_raw is not None:
            self._layer_idx = int(layer_idx_raw)

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
        # For models with internal rotary embeddings (e.g., GPT-J), the HF attention
        # forward expects position_ids to index into embed_positions. Models using
        # requires_position_embeddings get (cos, sin) tuples instead.
        if (
            self.config
            and hasattr(self.config, "positional_embedding_type")
            and self.config.positional_embedding_type == "rotary"
            and not self.requires_position_embeddings
        ):
            inputs["position_ids"] = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )
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

    def _update_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache if provided, returning the (possibly extended) K and V.

        Call this after K/V projections and any positional embeddings (e.g. RoPE)
        have been applied, but before computing attention scores. If no cache is
        present in kwargs, K and V are returned unchanged.
        """
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is None:
            return k, v
        layer_idx = getattr(self, "_layer_idx", None)
        if layer_idx is None:
            logger.warning(
                "%s: past_key_values provided but _layer_idx is None "
                "(HF component missing layer_idx attribute). "
                "KV cache update skipped — generation will be slow.",
                self.name,
            )
            return k, v
        k, v = past_key_values.update(k, v, layer_idx)
        return k, v

    def _reshape_qkv_to_heads(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """Reshape Q/K/V from [batch, seq, hidden] or [batch, seq, heads, head_dim]
        to [batch, heads, seq, head_dim]. Returns (q, k, v, batch_size, seq_len, head_dim).

        Args:
            num_kv_heads: If provided and differs from num_heads (GQA), K/V use
                this head count for the 3D reshape path.
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if q.ndim == 3:
            batch_size, seq_len, q_hidden = q.shape
            head_dim: int = q_hidden // num_heads
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        elif q.ndim == 4:
            batch_size, seq_len = q.shape[0], q.shape[1]
            head_dim = q.shape[-1]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected Q tensor shape: {q.shape}. Expected 3D or 4D.")
        return q, k, v, batch_size, seq_len, head_dim

    def _apply_attn_dropout(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply attention dropout from the original HF component if present."""
        if self.original_component is not None:
            dropout_fn = getattr(self.original_component, "attn_dropout", None)
            if dropout_fn is None:
                dropout_fn = getattr(self.original_component, "attention_dropout", None)
            if dropout_fn is not None:
                attn_weights = dropout_fn(attn_weights)
        return attn_weights

    def _apply_output_projection(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Apply the output projection (self.o) if present."""
        if hasattr(self, "o") and self.o is not None:
            attn_output = self.o(attn_output)
        return attn_output

    def _apply_reconstruct_attention_mask(
        self,
        attn_scores: torch.Tensor,
        attention_mask: torch.Tensor | None,
        seq_len: int,
        q_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Apply causal and optional attention masking to reconstructed scores.

        HuggingFace-style 4D masks already encode causal semantics, so they are
        treated as authoritative. Lower-rank masks do not, so the local causal
        mask is still applied before adding the caller-provided padding mask.

        Args:
            attn_scores: Attention scores [batch, heads, q_seq_len, kv_seq_len].
            attention_mask: Optional mask from the caller.
            seq_len: The KV sequence length (total positions including cache).
            q_seq_len: The query sequence length. When using KV cache this is
                shorter than seq_len. Defaults to seq_len when not provided.
        """
        if q_seq_len is None:
            q_seq_len = seq_len
        min_dtype = torch.finfo(attn_scores.dtype).min
        use_direct_hf_mask = attention_mask is not None and attention_mask.ndim >= 4
        if not use_direct_hf_mask:
            # Rectangular causal mask: query i attends to KV 0..(offset+i)
            # where offset = kv_seq_len - q_seq_len (cached positions).
            causal_mask = torch.ones(
                q_seq_len, seq_len, device=attn_scores.device, dtype=torch.bool
            )
            causal_mask = torch.tril(causal_mask, diagonal=seq_len - q_seq_len)
            attn_scores = attn_scores.masked_fill(~causal_mask, min_dtype)

        if attention_mask is None:
            return attn_scores

        if attention_mask.shape[-1] != seq_len:
            attention_mask = attention_mask[..., :seq_len]
        if attention_mask.ndim >= 3 and attention_mask.shape[-2] != q_seq_len:
            attention_mask = attention_mask[..., :q_seq_len, :]

        if attention_mask.dtype == torch.bool:
            attention_mask = torch.where(
                attention_mask,
                torch.zeros((), dtype=attn_scores.dtype, device=attn_scores.device),
                torch.full((), min_dtype, dtype=attn_scores.dtype, device=attn_scores.device),
            )
        else:
            attention_mask = attention_mask.to(dtype=attn_scores.dtype)

        return attn_scores + attention_mask

    def _get_n_heads(self, use_kv: bool = False) -> int:
        """Resolve the number of attention heads from config.

        Args:
            use_kv: If True, return n_key_value_heads (for GQA) when available.
        """
        assert self.config is not None, "config required to resolve n_heads"
        if use_kv:
            if hasattr(self.config, "n_key_value_heads") and self.config.n_key_value_heads:
                return self.config.n_key_value_heads
        if hasattr(self.config, "n_heads"):
            return self.config.n_heads
        return self.config.n_head

    def _reshape_weight_to_3d(
        self, weight: torch.Tensor, n_heads: int, pattern: str = "qkv"
    ) -> torch.Tensor:
        """Reshape a 2D weight to 3D by splitting heads, auto-detecting Linear vs Conv1D.

        Args:
            weight: 2D weight tensor
            n_heads: Number of heads to split into
            pattern: "qkv" for [n_heads, d_model, d_head], "o" for [n_heads, d_head, d_model]
        """
        if pattern == "o":
            if weight.shape[0] == n_heads * (
                weight.shape[1] // n_heads
                if weight.shape[1] % n_heads == 0
                else weight.shape[0] // n_heads
            ):
                return einops.rearrange(
                    weight, "(n_heads d_head) d_model -> n_heads d_head d_model", n_heads=n_heads
                )
            return einops.rearrange(
                weight.T, "(n_heads d_head) d_model -> n_heads d_head d_model", n_heads=n_heads
            )
        # QKV pattern
        if weight.shape[0] % n_heads == 0:
            return einops.rearrange(
                weight, "(n_heads d_head) d_model -> n_heads d_model d_head", n_heads=n_heads
            )
        return einops.rearrange(
            weight, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
        )

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
        """Get W_Q in 3D format [n_heads, d_model, d_head]."""
        weight = self.q.weight
        if weight.ndim == 2 and self.config is not None:
            return self._reshape_weight_to_3d(weight, self._get_n_heads())
        return weight

    @property
    def W_K(self) -> torch.Tensor:
        """Get W_K in 3D format [n_heads, d_model, d_head] (uses n_kv_heads for GQA)."""
        weight = self.k.weight
        if weight.ndim == 2 and self.config is not None:
            return self._reshape_weight_to_3d(weight, self._get_n_heads(use_kv=True))
        return weight

    @property
    def W_V(self) -> torch.Tensor:
        """Get W_V in 3D format [n_heads, d_model, d_head] (uses n_kv_heads for GQA)."""
        weight = self.v.weight
        if weight.ndim == 2 and self.config is not None:
            return self._reshape_weight_to_3d(weight, self._get_n_heads(use_kv=True))
        return weight

    @property
    def W_O(self) -> torch.Tensor:
        """Get W_O in 3D format [n_heads, d_head, d_model]."""
        weight = self.o.weight
        if weight.ndim == 2 and self.config is not None:
            return self._reshape_weight_to_3d(weight, self._get_n_heads(), pattern="o")
        return weight

    def _reshape_bias(
        self, bias: Optional[torch.Tensor], use_kv: bool = False
    ) -> Optional[torch.Tensor]:
        """Reshape 1D bias to [n_heads, d_head]."""
        if bias is not None and bias.ndim == 1 and self.config is not None:
            n_heads = self._get_n_heads(use_kv=use_kv)
            return einops.rearrange(bias, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
        return bias

    @property
    def b_Q(self) -> Optional[torch.Tensor]:
        """Get b_Q in 2D format [n_heads, d_head]."""
        return self._reshape_bias(self.q.bias)

    @property
    def b_K(self) -> Optional[torch.Tensor]:
        """Get b_K in 2D format [n_heads, d_head] (uses n_kv_heads for GQA)."""
        return self._reshape_bias(self.k.bias, use_kv=True)

    @property
    def b_V(self) -> Optional[torch.Tensor]:
        """Get b_V in 2D format [n_heads, d_head] (uses n_kv_heads for GQA)."""
        return self._reshape_bias(self.v.bias, use_kv=True)

    @property
    def b_O(self) -> Optional[torch.Tensor]:
        """Get b_O bias from linear bridge."""
        return self.o.bias
