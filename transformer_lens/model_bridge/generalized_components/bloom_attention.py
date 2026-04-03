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

        BLOOM computes attention scores via alibi.baddbmm(Q, K^T) which fuses the
        ALiBi positional bias directly into the score computation. This override
        mirrors that behavior while keeping all hook points active.
        """
        assert self.original_component is not None
        assert self.config is not None
        num_heads = self.config.n_heads
        head_dim: int = self.config.d_head

        # Reshape Q/K/V from [batch, seq, hidden] to [batch, heads, seq, head_dim]
        if q.ndim == 3:
            batch_size, seq_len, _ = q.shape
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        elif q.ndim == 4:
            batch_size, seq_len = q.shape[0], q.shape[1]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected Q tensor shape: {q.shape}")

        # BLOOM uses baddbmm: alibi + Q @ K^T * inv_norm_factor
        # Reshape to [batch*heads, seq, head_dim] for baddbmm
        q_bh = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_bh = k.reshape(batch_size * num_heads, seq_len, head_dim)
        v_bh = v.reshape(batch_size * num_heads, seq_len, head_dim)

        inv_norm_factor = head_dim ** (-0.5)
        beta = 1.0

        alibi = kwargs.get("alibi", None)
        if alibi is not None:
            # alibi shape: [batch*heads, 1, seq] or [batch*heads, seq, seq]
            # baddbmm: beta * alibi + alpha * (Q @ K^T)
            attn_scores = alibi.baddbmm(
                batch1=q_bh,
                batch2=k_bh.transpose(-1, -2),
                beta=beta,
                alpha=inv_norm_factor,
            )
        else:
            attn_scores = torch.bmm(q_bh, k_bh.transpose(-1, -2)) * inv_norm_factor

        # Reshape to [batch, heads, seq, seq]
        attn_scores = attn_scores.view(batch_size, num_heads, seq_len, -1)

        # Apply attention mask
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : attn_scores.shape[-1]]
            attn_scores = attn_scores + causal_mask

        attn_scores = self.hook_attn_scores(attn_scores)

        # Softmax in float32 for numerical stability (matches HF BLOOM)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)

        if hasattr(self.original_component, "attention_dropout"):
            attn_weights = self.original_component.attention_dropout(attn_weights)  # type: ignore[operator]

        attn_weights = self.hook_pattern(attn_weights)

        # Compute attention output
        # Reshape weights to [batch*heads, seq, seq] for bmm
        attn_weights_bh = attn_weights.reshape(batch_size * num_heads, seq_len, -1)
        attn_output = torch.bmm(attn_weights_bh, v_bh)

        # Reshape back to [batch, seq, hidden]
        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, num_heads * head_dim)

        # Apply output projection
        if hasattr(self, "o") and self.o is not None:
            attn_output = self.o(attn_output)

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
