"""CodeGen-specific attention bridge component.

CodeGen attention uses a fused QKV projection (qkv_proj) with a GPT-J-style
``rotate_every_two`` rotary positional encoding applied to Q and K before the
attention matmul.  The rotary embeddings are stored as a sinusoidal buffer
(``embed_positions``) on the original ``CodeGenAttention`` module and are
indexed by ``position_ids``.

Optional parameters (may be absent in some CodeGen checkpoints):
  - rotary_dim: if None, RoPE is applied to the full head dimension.
"""

from typing import Any, Callable, Dict, Optional

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


# ---------------------------------------------------------------------------
# Rotary helpers — GPT-J / CodeGen style ("rotate_every_two")
# ---------------------------------------------------------------------------


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """Rotate every pair of elements (GPT-J / CodeGen style).

    Mirrors ``rotate_every_two`` from
    ``transformers.models.codegen.modeling_codegen`` (line 56-60).

    Args:
        x: Tensor of shape ``[batch, heads, seq, head_dim]``.

    Returns:
        Tensor of the same shape with even/odd pairs rotated.
    """
    x1 = x[:, :, :, ::2]   # even-indexed dims
    x2 = x[:, :, :, 1::2]  # odd-indexed dims
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_pos_emb(
    tensor: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings (GPT-J / CodeGen style).

    Adapted from ``apply_rotary_pos_emb`` in
    ``transformers.models.codegen.modeling_codegen`` (line 64-67) to work
    with tensors in the TransformerLens ``[batch, heads, seq, head_dim]``
    layout (heads and seq are swapped relative to HuggingFace).

    Args:
        tensor: ``[batch, heads, seq, rotary_dim]`` — the slice of Q or K that
            will be rotated.
        sin: ``[batch, seq, rotary_dim // 2]`` — the sin half of the sinusoidal
            embedding (before ``repeat_interleave``).
        cos: ``[batch, seq, rotary_dim // 2]`` — the cos half.

    Returns:
        Rotated tensor with the same shape as *tensor*.
    """
    # Expand sin/cos from [batch, seq, rotary_dim//2]
    # to [batch, 1, seq, rotary_dim] so they broadcast with
    # tensor of shape [batch, heads, seq, rotary_dim].
    sin = torch.repeat_interleave(sin[:, None, :, :], 2, 3)  # [B, 1, seq, rot_dim]
    cos = torch.repeat_interleave(cos[:, None, :, :], 2, 3)  # [B, 1, seq, rot_dim]
    return (tensor * cos) + (_rotate_every_two(tensor) * sin)


class CodeGenAttentionBridge(JointQKVAttentionBridge):
    """Attention bridge for CodeGen models.

    CodeGen uses:
    - A fused ``qkv_proj`` linear (no bias).
    - GPT-J-style ``rotate_every_two`` RoPE applied to Q and K before the
      attention matmul.  Rotary embeddings are stored in the
      ``embed_positions`` buffer of the original ``CodeGenAttention`` module
      and indexed by ``position_ids``.
    - Only the first ``rotary_dim`` dimensions of each head are rotated.
      When ``rotary_dim`` is None the full head dimension is rotated.
    - An ``out_proj`` linear output projection (no bias).

    All TransformerLens hooks fire in the forward pass:
    ``hook_q``, ``hook_k``, ``hook_v``, ``hook_attn_scores``,
    ``hook_pattern``, ``hook_z`` (via ``o.hook_in``), ``hook_result``
    (via ``hook_out``).
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
    ) -> None:
        """Initialise the CodeGen attention bridge.

        Args:
            name: The name of this component.
            config: Model configuration (must have ``n_heads``, ``d_head``,
                and optionally ``rotary_dim``).
            split_qkv_matrix: Callable that splits the fused QKV weight into
                three ``nn.Linear`` modules for Q, K, and V.  Required — there
                is no sensible default for CodeGen's mp_num=4 split logic.
            submodules: Optional extra submodules to register.
            qkv_conversion_rule: Optional conversion rule for Q/K/V outputs.
            attn_conversion_rule: Optional conversion rule for the attention
                output.
            pattern_conversion_rule: Optional conversion rule for attention
                patterns.
        """
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

    # ------------------------------------------------------------------
    # Component testing inputs
    # ------------------------------------------------------------------

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device=None,
        dtype=None,
    ):
        """Return random inputs for isolated component testing.

        CodeGen attention requires ``position_ids`` (to index into
        ``embed_positions``) and a HuggingFace-style 4D causal attention mask.
        The mask is provided so that both the bridge and the HF component
        apply identical causal masking during the ``all_components`` benchmark.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            device: Target device (defaults to CPU).
            dtype: Tensor dtype (defaults to float32).

        Returns:
            Dict with ``hidden_states``, ``position_ids``, and
            ``attention_mask`` suitable for both bridge and HF forward calls.
        """
        import torch

        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        d_model = (
            self.config.d_model
            if self.config and hasattr(self.config, "d_model")
            else 768
        )

        # Build the HF-style 4D causal mask: 0 where attended, -inf where masked.
        # Shape: [batch, 1, seq_len, seq_len]
        min_val = torch.finfo(dtype).min
        causal = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)
        mask_upper = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        causal[:, 0] = causal[:, 0].masked_fill(mask_upper, min_val)

        return {
            "hidden_states": torch.randn(
                batch_size, seq_len, d_model, device=device, dtype=dtype
            ),
            "position_ids": torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1),
            "attention_mask": causal,
        }

    # ------------------------------------------------------------------
    # Component wiring
    # ------------------------------------------------------------------

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Wire the original CodeGenAttention and set up the output projection.

        The base ``JointQKVAttentionBridge.set_original_component`` hardcodes
        ``c_proj`` for the output projection wiring.  CodeGen uses ``out_proj``
        instead, so we override here to wire it correctly after calling super.

        Args:
            original_component: The original ``CodeGenAttention`` layer.
        """
        # Let the base class split QKV; it will attempt (and fail-silently) the
        # c_proj wiring because CodeGen has no c_proj attribute.
        super().set_original_component(original_component)

        # Wire out_proj explicitly.
        if hasattr(self, "o") and hasattr(original_component, "out_proj"):
            self.o.set_original_component(original_component.out_proj)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through CodeGen attention with all hooks firing.

        Manually reconstructs attention so that all TransformerLens hooks
        (hook_q, hook_k, hook_v, hook_attn_scores, hook_pattern, hook_z,
        hook_result) fire correctly.

        CodeGen passes ``position_ids`` as a keyword argument; these are used
        to index into the ``embed_positions`` sinusoidal buffer stored on the
        original ``CodeGenAttention`` module.

        Args:
            *args: Positional arguments; the first must be ``hidden_states``.
            **kwargs: Keyword arguments including ``position_ids`` (required
                for RoPE), ``attention_mask`` (optional), ``layer_past``
                (optional KV cache), and ``cache_position`` (optional).

        Returns:
            Tuple of ``(attn_output, attn_weights)``.
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        # ---- 1. Extract hidden_states ----
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            hidden_states = kwargs["hidden_states"]
        else:
            raise ValueError("Could not find hidden_states in args or kwargs.")

        # ---- 2. Input hook ----
        hooked_input = self.hook_in(hidden_states)

        # ---- 3. Q / K / V projections (fires hook_q, hook_k, hook_v) ----
        q_output = self.q(hooked_input)
        k_output = self.k(hooked_input)
        v_output = self.v(hooked_input)

        # ---- 4. Reconstruct attention with RoPE ----
        attn_output, attn_weights = self._reconstruct_attention(
            q_output, k_output, v_output, **kwargs
        )

        # ---- 5. Output hooks (fires hook_z via o.hook_in, hook_result via hook_out) ----
        output = (attn_output, attn_weights)
        output = self._process_output(output)
        return output

    def _reconstruct_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: Any,
    ) -> tuple:
        """Reconstruct attention with CodeGen's rotate_every_two RoPE.

        This method:
        1. Reshapes Q/K/V to ``[batch, heads, seq, head_dim]``.
        2. Applies ``rotate_every_two`` RoPE to Q and K (first ``rotary_dim``
           dimensions only when ``rotary_dim`` is set).
        3. Runs scaled dot-product attention (fp32, matching HF CodeGen).
        4. Fires ``hook_attn_scores`` and ``hook_pattern``.
        5. Applies the output projection via ``self.o``.

        Args:
            q: Q tensor from the Q LinearBridge.
            k: K tensor from the K LinearBridge.
            v: V tensor from the V LinearBridge.
            **kwargs: Forwarded kwargs; must include ``position_ids``.

        Returns:
            ``(attn_output, attn_weights)`` tuple.
        """
        assert self.original_component is not None
        assert self.config is not None

        num_heads: int = self.config.n_heads

        # Reshape to [batch, heads, seq, head_dim]
        q, k, v, batch_size, seq_len, head_dim = self._reshape_qkv_to_heads(q, k, v, num_heads)

        # ---- RoPE ----
        position_ids: Optional[torch.Tensor] = kwargs.get("position_ids", None)
        if position_ids is not None:
            embed_positions: torch.Tensor = self.original_component.embed_positions  # type: ignore[union-attr]
            # Move buffer to the right device if needed (mirrors HF forward)
            if embed_positions.device != position_ids.device:
                embed_positions = embed_positions.to(position_ids.device)

            # sincos: [batch, seq, rotary_dim] (full dim = sin_half + cos_half)
            sincos = embed_positions[position_ids]
            half = sincos.shape[-1] // 2
            sin, cos = sincos[:, :, :half], sincos[:, :, half:]

            rotary_dim: Optional[int] = getattr(self.original_component, "rotary_dim", None)
            if rotary_dim is not None:
                # Only rotate the first rotary_dim dimensions; pass the rest through.
                q_rot = _apply_rotary_pos_emb(q[:, :, :, :rotary_dim], sin, cos)
                k_rot = _apply_rotary_pos_emb(k[:, :, :, :rotary_dim], sin, cos)
                q = torch.cat([q_rot, q[:, :, :, rotary_dim:]], dim=-1)
                k = torch.cat([k_rot, k[:, :, :, rotary_dim:]], dim=-1)
            else:
                q = _apply_rotary_pos_emb(q, sin, cos)
                k = _apply_rotary_pos_emb(k, sin, cos)

        # ---- KV cache ----
        k, v = self._update_kv_cache(k, v, **kwargs)
        kv_seq_len = k.shape[-2]

        # ---- Scaled dot-product (fp32, matching HF CodeGen._attn) ----
        scale = self.original_component.scale_attn  # type: ignore[union-attr]
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)

        attn_scores = torch.matmul(q_f32, k_f32.transpose(-2, -1))

        attention_mask: Optional[torch.Tensor] = kwargs.get("attention_mask", None)
        attn_scores = self._apply_reconstruct_attention_mask(
            attn_scores=attn_scores,
            attention_mask=attention_mask,
            seq_len=kv_seq_len,
            q_seq_len=seq_len,
        )

        # Divide by scale_attn (CodeGen divides *after* the mask, not before)
        attn_scores = attn_scores / scale

        attn_scores = self.hook_attn_scores(attn_scores)

        # Softmax + dropout + hook_pattern
        attn_weights = self._softmax_dropout_pattern(
            attn_scores,
            target_dtype=v.dtype,
        )

        attn_output = torch.matmul(attn_weights, v)

        # Reshape [batch, heads, seq, head_dim] → [batch, seq, hidden]
        attn_output = self._reshape_attn_output(attn_output, batch_size, seq_len, num_heads, head_dim)

        # Output projection (fires hook_z via o.hook_in)
        attn_output = self._apply_output_projection(attn_output)

        return (attn_output, attn_weights)
