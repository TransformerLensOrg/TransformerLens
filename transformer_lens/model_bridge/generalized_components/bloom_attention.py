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

        Uses the parent's hooked Q/K/V split path so that hook_q, hook_k, hook_v,
        hook_attn_scores, and hook_pattern all fire correctly. ALiBi bias and
        attention masking are handled in _reconstruct_attention.

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

        # Extract hidden_states (first positional arg) and residual (second positional arg)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            hidden_states = kwargs["hidden_states"]
        else:
            raise ValueError("Could not find hidden_states in args or kwargs")

        residual = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) else None

        # Apply input hook
        hooked_input = self.hook_in(hidden_states)

        # Run through split Q/K/V projections (these fire hook_q, hook_k, hook_v)
        q_output = self.q(hooked_input)
        k_output = self.k(hooked_input)
        v_output = self.v(hooked_input)

        # Reconstruct attention with ALiBi (fires hook_attn_scores, hook_pattern)
        attn_output, attn_weights = self._reconstruct_attention(
            q_output, k_output, v_output, **kwargs
        )

        # BLOOM's original attention applies dropout_add(dense_output, residual, ...)
        # inside the attention module, not in the block. We must replicate this.
        if residual is not None:
            assert self.original_component is not None
            hidden_dropout = getattr(self.original_component, "hidden_dropout", 0.0)
            if self.training:
                attn_output = torch.nn.functional.dropout(
                    attn_output, p=hidden_dropout, training=True
                )
            attn_output = attn_output + residual

        # Apply output hook
        output = (attn_output, attn_weights)
        output = self._process_output(output)

        return output

    def _reconstruct_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> tuple:
        """Reconstruct attention using BLOOM's ALiBi-based score computation.

        BLOOM fuses the ALiBi positional bias into scores via baddbmm.
        """
        assert self.original_component is not None
        assert self.config is not None
        num_heads = self.config.n_heads

        q, k, v, batch_size, seq_len, head_dim = self._reshape_qkv_to_heads(q, k, v, num_heads)

        # KV cache: extend K/V with cached positions.
        k, v = self._update_kv_cache(k, v, **kwargs)

        kv_seq_len = k.shape[-2]  # Includes cached positions
        # Reshape to [batch*heads, seq, head_dim] for baddbmm
        q_bh = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_bh = k.reshape(batch_size * num_heads, kv_seq_len, head_dim)
        v_bh = v.reshape(batch_size * num_heads, kv_seq_len, head_dim)

        inv_norm_factor = head_dim ** (-0.5)

        alibi = kwargs.get("alibi", None)
        if alibi is not None:
            # Resize alibi to match kv_seq_len (may differ after cache update).
            alibi_kv_len = alibi.shape[-1]
            if alibi_kv_len < kv_seq_len:
                # ALiBi is slope * position — recompute for the extended length.
                if alibi.ndim == 3 and alibi.shape[1] == 1:
                    slopes = alibi[:, 0, 1:2]  # [batch*heads, 1]
                    if slopes.numel() > 0 and slopes.abs().sum() > 0:
                        positions = torch.arange(
                            kv_seq_len, device=alibi.device, dtype=alibi.dtype
                        ).unsqueeze(0)
                        alibi = slopes * positions  # [batch*heads, kv_seq_len]
                        alibi = alibi.unsqueeze(1)  # [batch*heads, 1, kv_seq_len]
            elif alibi_kv_len > kv_seq_len:
                alibi = alibi[..., :kv_seq_len]
            attn_scores = alibi.baddbmm(
                batch1=q_bh,
                batch2=k_bh.transpose(-1, -2),
                beta=1.0,
                alpha=inv_norm_factor,
            )
        else:
            attn_scores = torch.bmm(q_bh, k_bh.transpose(-1, -2)) * inv_norm_factor

        attn_scores = attn_scores.view(batch_size, num_heads, seq_len, -1)

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : attn_scores.shape[-1]]
            attn_scores = attn_scores + causal_mask

        attn_scores = self.hook_attn_scores(attn_scores)

        # Softmax in float32 for numerical stability (matches HF BLOOM)
        attn_weights = self._softmax_dropout_pattern(
            attn_scores, target_dtype=q.dtype, upcast_to_fp32=True
        )

        # bmm in [batch*heads, seq, seq] format for BLOOM compatibility
        attn_weights_bh = attn_weights.reshape(batch_size * num_heads, seq_len, -1)
        attn_output = torch.bmm(attn_weights_bh, v_bh)

        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)
        attn_output = self._reshape_attn_output(
            attn_output, batch_size, seq_len, num_heads, head_dim
        )
        attn_output = self._apply_output_projection(attn_output)

        return (attn_output, attn_weights)

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
