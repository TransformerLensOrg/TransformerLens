"""Hooked Transformer Alternating Attention Component.

This module contains the AlternatingAttention component which alternates between regular attention
and grouped query attention across layers.
"""
import math
from abc import ABC
from typing import Dict, Optional, Tuple, TypeVar, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from better_abc import abstract_attribute
from jaxtyping import Bool, Float, Int, jaxtyped
from transformers.utils import is_bitsandbytes_available

from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
from transformer_lens.utilities.attention import complex_attn_linear, simple_attn_linear
from transformer_lens.utils import get_offset_position_ids

# Import bitsandbytes if available
if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import Params4bit
    BITSANDBYTES_AVAILABLE = True
else:
    BITSANDBYTES_AVAILABLE = False
    Params4bit = None

# Type variables for attention dimensions
HeadIndex = TypeVar("HeadIndex", bound=int)
KvHeadIndex = TypeVar("KvHeadIndex", bound=int)
DHead = TypeVar("DHead", bound=int)
DModel = TypeVar("DModel", bound=int)
Batch = TypeVar("Batch", bound=int)
Pos = TypeVar("Pos", bound=int)
QueryPos = TypeVar("QueryPos", bound=int)
KeyPos = TypeVar("KeyPos", bound=int)
KvPos = TypeVar("KvPos", bound=int)
OffsetPos = TypeVar("OffsetPos", bound=int)
RotaryDim = TypeVar("RotaryDim", bound=int)

class AlternatingAttention(nn.Module):
    alibi: Union[torch.Tensor, None]

    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        attn_type: str = "global",
        layer_id: Optional[int] = None,
    ):
        """Alternating Attention Block that switches between regular and grouped query attention.
        Similar to regular attention, W_Q, W_K, and W_V all have shape [head_index, d_model, d_head].
        However, under the hood the key and value weights _W_K and _W_V are stored with shape [n_key_value_heads, d_model, d_head] and are expanded when the corresponding properties' getter is called.
        Similarly, during a forward pass, initially K and V are kept in shapes [batch, pos, n_key_value_heads, d_head] and will only be expanded to shapes [batch, pos, n_heads, d_head]
        using torch.repeat_interleave when the attention pattern and z-scores are calculated.

        Args:
            cfg (Union[Dict, HookedTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistal models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        assert self.cfg.n_key_value_heads is not None
        self.repeat_kv_heads = self.cfg.n_heads // self.cfg.n_key_value_heads

        if self.cfg.load_in_4bit:
            nq = int((self.cfg.d_model * self.cfg.d_head * self.cfg.n_heads) / 2)
            self.W_Q = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
            self.W_O = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
        else:
            self.W_Q = nn.Parameter(
                torch.empty(
                    self.cfg.n_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=self.cfg.dtype,
                )
            )
            self.W_O = nn.Parameter(
                torch.empty(
                    self.cfg.n_heads,
                    self.cfg.d_head,
                    self.cfg.d_model,
                    dtype=self.cfg.dtype,
                )
            )

        # Initialize key and value weights with grouped query attention shape
        if self.cfg.load_in_4bit:
            nkv = int((self.cfg.d_model * self.cfg.d_head * self.cfg.n_key_value_heads) / 2)
            self._W_K = Params4bit(torch.empty(nkv, 1, dtype=torch.uint8), requires_grad=False)
            self._W_V = Params4bit(torch.empty(nkv, 1, dtype=torch.uint8), requires_grad=False)
        else:
            self._W_K = nn.Parameter(
                torch.empty(
                    self.cfg.n_key_value_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=self.cfg.dtype,
                )
            )
            self._W_V = nn.Parameter(
                torch.empty(
                    self.cfg.n_key_value_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=self.cfg.dtype,
                )
            )

        self.b_Q = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self._b_K = nn.Parameter(
            torch.zeros(self.cfg.n_key_value_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self._b_V = nn.Parameter(
            torch.zeros(self.cfg.n_key_value_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        # Properties for expanded weights
        @property
        def W_K(self):
            return torch.repeat_interleave(self._W_K, dim=0, repeats=self.repeat_kv_heads)

        @W_K.setter
        def W_K(self, value):
            self._W_K = value

        @property
        def W_V(self):
            return torch.repeat_interleave(self._W_V, dim=0, repeats=self.repeat_kv_heads)

        @W_V.setter
        def W_V(self, value):
            self._W_V = value

        @property
        def b_K(self):
            return torch.repeat_interleave(self._b_K, dim=0, repeats=self.repeat_kv_heads)

        @b_K.setter
        def b_K(self, value):
            self._b_K = value

        @property
        def b_V(self):
            return torch.repeat_interleave(self._b_V, dim=0, repeats=self.repeat_kv_heads)

        @b_V.setter
        def b_V(self, value):
            self._b_V = value

        self.attn_type = attn_type
        # Create a max_ctx x max_ctx mask, with True iff that query position
        # can attend to that key position (query is first axis, key is second axis)
        causal_mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool())
        if self.attn_type == "global":
            # For global attention, this is a lower triangular matrix - key <= query
            self.register_buffer("mask", causal_mask)
        elif self.attn_type == "local":
            # For local, this is banded, query - window_size < key <= query
            if not isinstance(self.cfg.window_size, int):
                raise ValueError("Window size must be an integer for local attention")
            self.register_buffer("mask", torch.triu(causal_mask, 1 - self.cfg.window_size))
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        self.register_buffer("IGNORE", torch.tensor(-torch.inf))

        self.layer_id = layer_id

        # attn_scale is a constant that we divide the attention scores by pre-softmax. I'm not entirely sure why it matters, but it's probably a mix of softmax not being scale invariant and numerical stability?
        if self.cfg.use_attn_scale:
            self.attn_scale = self.cfg.attn_scale  # Defaults to sqrt(d_head)
        else:
            self.attn_scale = 1.0
        if self.cfg.scale_attn_by_inverse_layer_idx:
            if self.layer_id is None:  # keep mypy happy
                raise ValueError("Layer ID must be provided to scale attention scores")
            self.attn_scale *= self.layer_id + 1

        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, pos, head_index, d_model]

        # See HookedTransformerConfig for more details.
        if self.cfg.positional_embedding_type == "shortformer":
            # This tracks the input to the keys and queries, which is resid_pre + pos_embeds
            self.hook_attn_input = HookPoint()  # [batch, pos, d_model]
        elif self.cfg.positional_embedding_type == "rotary":
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position. See HookedTransformerConfig for details
            self.hook_rot_k = HookPoint()
            self.hook_rot_q = HookPoint()
            if self.cfg.rotary_dim is None:  # keep mypy happy
                raise ValueError("Rotary dim must be provided for rotary positional embeddings")
            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                base=self.cfg.rotary_base,
                dtype=self.cfg.dtype,
            )
            self.register_buffer("rotary_sin", sin.unsqueeze(1))  # [n_ctx, 1, rotary_dim]
            self.register_buffer("rotary_cos", cos.unsqueeze(1))  # [n_ctx, 1, rotary_dim]
        elif self.cfg.positional_embedding_type == "alibi":
            # ALiBi bias wil be constructed on the first forward pass.
            # Note: While computationally efficient, initializing an bias with max n_ctx (16, 1024, 1024) of float32 will occupy ~256MiB of contiguous GPU memory, which may not be optimal for memory usage.
            self.alibi = None

        elif self.cfg.positional_embedding_type == "relative_positional_bias":
            # will be overwritten by the child T5Attention class
            self.has_relative_attention_bias = False

    @property
    def OV(self) -> FactoredMatrix:
        """
        OV-Circuit, as defined in A Mathematical Framework. Because there's no non-linearity between the value vector and the output of the layer, the output is purely determined by the matrix W_OV = W_V @ W_O, and not W_V or W_O individually. (Mathematically, for a single head, output == pattern @ residual @ W_V @ W_O, see the glossary for more)

        Done in the order W_V, W_O because the paper uses left-multiplying weight matrices, and TransformerLens uses right-multiplying, sorry!

        Returns a FactoredMatrix, with left matrix W_V [head_index, d_model, d_head] and right matrix W_O [head_index, d_head, d_model] - this is a low rank factorisation of the underlying [head_index, d_model, d_model]. FactoredMatrix has helper functions to deal with these large matrices efficiently. To get the OV circuit of a head k, attn.OV[k] works.
        """
        return FactoredMatrix(self.W_V, self.W_O)

    @property
    def QK(self) -> FactoredMatrix:
        """
        QK-Circuit, as defined in A Mathematical Framework. Because there's no non-linearity in the key-query dot product, the output is purely determined by the matrix W_QK = W_Q.T @ W_K, and not W_Q or W_K individually. (Mathematically, for a single head, pattern = destination_residual.T @ W_Q.T @ W_K @ source-residual, see the glossary for more).

        Done in the order Q on the left, K on the right, because the pattern has dimensions [destination_pos, source_pos]

        Returns a FactoredMatrix, with left matrix W_Q [head_index, d_model, d_head] and right matrix W_K.T [head_index, d_head, d_model] - this is a low rank factorisation of the underlying [head_index, d_model, d_model] matrix. FactoredMatrix has helper functions to deal with these large matrices efficiently. To get the QK circuit of a head k, attn.QK[k] works.
        """
        W_K_transpose = einops.rearrange(
            self.W_K, "head_index d_model d_head -> head_index d_head d_model"
        )
        return FactoredMatrix(self.W_Q, W_K_transpose)

    def forward(
        self,
        query_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
        key_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
            Float[torch.Tensor, "batch kv_pos n_kv_heads d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
            Float[torch.Tensor, "batch kv_pos n_kv_heads d_model"],
        ],
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
        cache_position: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Forward pass for the attention layer.

        Args:
            query_input (Union[Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos head_index d_model"]]): Query input tensor
            key_input (Union[Float[torch.Tensor, "batch kv_pos d_model"], Float[torch.Tensor, "batch kv_pos head_index d_model"]]): Key input tensor
            value_input (Union[Float[torch.Tensor, "batch kv_pos d_model"], Float[torch.Tensor, "batch kv_pos head_index d_model"]]): Value input tensor
            past_kv_cache_entry (Optional[HookedTransformerKeyValueCacheEntry], optional): Past key/value cache. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            rotary_cos (Optional[torch.Tensor], optional): Cosine for rotary embeddings. Defaults to None.
            rotary_sin (Optional[torch.Tensor], optional): Sine for rotary embeddings. Defaults to None.
            cache_position (Optional[int], optional): Position in the cache. Defaults to None.

        Returns:
            Float[torch.Tensor, "batch pos d_model"]: The output tensor
        """
        print("Running forward in", self.__class__.__name__)
        q, k, v = self.calculate_qkv_matrices(query_input, key_input, value_input)

        if self.cfg.positional_embedding_type == "rotary":
            q = self.apply_rotary(q)
            k = self.apply_rotary(k)

        if past_kv_cache_entry is not None:
            past_k = past_kv_cache_entry.past_keys
            past_v = past_kv_cache_entry.past_values
            # Ensure k and v have the same number of heads as past_k and past_v
            if not self.cfg.ungroup_grouped_query_attention:
                k = k[:, :, ::self.repeat_kv_heads, :]
                v = v[:, :, ::self.repeat_kv_heads, :]
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        if self.cfg.attention_dir == "causal":
            # Create a causal mask
            causal_mask = torch.tril(torch.ones((q.size(1), k.size(1))).bool())
            attention_mask = causal_mask.to(q.device)
        elif attention_mask is not None:
            attention_mask = attention_mask.to(q.device)
        else:
            attention_mask = None

        # Expand k and v for attention score calculation
        k_expanded = torch.repeat_interleave(k, dim=2, repeats=self.repeat_kv_heads) if not self.cfg.ungroup_grouped_query_attention else k
        v_expanded = torch.repeat_interleave(v, dim=2, repeats=self.repeat_kv_heads) if not self.cfg.ungroup_grouped_query_attention else v

        attn_scores = self.calculate_attention_scores(q, k_expanded)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        pattern = F.softmax(attn_scores, dim=-1)
        pattern = self.hook_pattern(pattern)

        z = self.calculate_z_scores(v_expanded, pattern)

        if self.cfg.use_split_qkv_input or self.cfg.use_attn_in:
            out = complex_attn_linear(z, self.W_O, self.b_O)
        else:
            # Custom approach for output projection to handle the 1D bias tensor
            # Flatten z from [batch, pos, head_index, d_head] to [batch*pos, head_index*d_head]
            z_flat = z.reshape(z.shape[0] * z.shape[1], -1)
            
            # Flatten W_O from [head_index, d_head, d_model] to [head_index*d_head, d_model]
            w_o_flat = self.W_O.reshape(-1, self.W_O.shape[-1])
            
            # Apply linear transformation
            out_flat = torch.nn.functional.linear(z_flat, w_o_flat.t(), self.b_O)
            
            # Reshape back to [batch, pos, d_model]
            out = out_flat.reshape(z.shape[0], z.shape[1], -1)

        out = self.hook_result(out)

        if past_kv_cache_entry is not None:
            # Store the key/value state for future use
            _ = HookedTransformerKeyValueCacheEntry(past_keys=k, past_values=v)
        return out

    def calculate_qkv_matrices(
        self,
        query_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
        key_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
            Float[torch.Tensor, "batch kv_pos n_kv_heads d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
            Float[torch.Tensor, "batch kv_pos n_kv_heads d_model"],
        ],
    ) -> Tuple[
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch kv_pos n_kv_heads d_head"],
        Float[torch.Tensor, "batch kv_pos n_kv_heads d_head"],
    ]:
        """Calculate the Q, K, and V matrices for grouped query attention.
        This function uses the unexpanded weights _W_K and _W_V to calculate K and V.

        Args:
        query_input (Union[Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos head_index d_model"]]): The input tensor for the query projection.
        key_input (Union[Float[torch.Tensor, "batch kv_pos d_model"], Float[torch.Tensor, "batch kv_pos head_index d_model"]]): The input tensor for the key projection. Note that is has as many head dimensions as the GPA block has key-value heads.
        value_input (Union[Float[torch.Tensor, "batch kv_pos d_model"], Float[torch.Tensor, "batch kv_pos head_index d_model"]]): The input tensor for the value projection. Note that is has as many head dimensions as the GPA block has key-value heads.

        Returns:
        Tuple[Float[torch.Tensor, "batch pos head_index d_head"], Float[torch.Tensor, "batch kv_pos n_kv_heads d_head"], Float[torch.Tensor, "batch kv_pos n_kv_heads d_head"]]:
        A tuple containing the Q, K, and V matrices with the specified shapes.
        """
        attn_fn = (
            complex_attn_linear
            if self.cfg.use_split_qkv_input or self.cfg.use_attn_in
            else simple_attn_linear
        )

        q = self.hook_q(
            attn_fn(query_input, self.W_Q, self.b_Q)
        )  # [batch, pos, head_index, d_head]

        k = self.hook_k(
            attn_fn(key_input, self.W_K, self.b_K)
            if self.cfg.ungroup_grouped_query_attention
            else attn_fn(key_input, self._W_K, self._b_K)
        )  # [batch, pos, n_kv_heads, d_head]
        v = self.hook_v(
            attn_fn(value_input, self.W_V, self.b_V)
            if self.cfg.ungroup_grouped_query_attention
            else attn_fn(value_input, self._W_V, self._b_V)
        )  # [batch, pos, n_kv_heads, d_head]
        return q, k, v

    def calculate_attention_scores(
        self,
        q: Float[torch.Tensor, "batch query_pos head_index d_head"],
        k: Float[torch.Tensor, "batch key_pos head_index d_head"],
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        """Calculate attention scores from Q and K matrices.

        Args:
        q (Float[torch.Tensor, "batch query_pos head_index d_head"]): The Q tensor.
        k (Float[torch.Tensor, "batch key_pos head_index d_head"]): The K tensor.

        Returns:
            Float[torch.Tensor, "batch head_index query_pos key_pos"]: The attention scores.
        """
        # Rearrange tensors for matrix multiplication
        q_ = einops.rearrange(q, "batch query_pos head_index d_head -> batch head_index query_pos d_head")
        k_ = einops.rearrange(k, "batch key_pos head_index d_head -> batch head_index d_head key_pos")

        # Calculate attention scores
        attn_scores = q_ @ k_ / self.attn_scale

        # Apply soft cap if configured
        if self.cfg.attn_scores_soft_cap > 0:
            attn_scores = self.cfg.attn_scores_soft_cap * F.tanh(
                attn_scores / self.cfg.attn_scores_soft_cap
            )

        return attn_scores

    def calculate_z_scores(
        self,
        v: Float[torch.Tensor, "batch key_pos head_index d_head"],
        pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    ) -> Float[torch.Tensor, "batch query_pos head_index d_head"]:
        """Calculate z scores from the attention pattern and V matrix.

        Args:
        v (Float[torch.Tensor, "batch key_pos head_index d_head"]): The V tensor.
        pattern (Float[torch.Tensor, "batch head_index query_pos key_pos"]): The attention pattern.

        Returns:
            Float[torch.Tensor, "batch query_pos head_index d_head"]: The z scores.
        """
        # Rearrange tensors for matrix multiplication
        v_ = einops.rearrange(v, "batch key_pos head_index d_head -> batch head_index key_pos d_head")
        pattern_ = einops.rearrange(pattern, "batch head_index query_pos key_pos -> batch head_index query_pos key_pos")

        # Calculate z scores
        z = self.hook_z(
            einops.rearrange(
                pattern_ @ v_,
                "batch head_index query_pos d_head -> batch query_pos head_index d_head",
            )
        )
        return z

    def calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Float[torch.Tensor, "n_ctx rotary_dim"], Float[torch.Tensor, "n_ctx rotary_dim"]]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

        Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
        To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
        """
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)

        # Handle hybrid RoPE configuration for local vs global attention
        if hasattr(self.cfg, 'rope_local_base_freq') and self.attn_type == 'local':
            base = self.cfg.rope_local_base_freq

        # Llama-3.1 uses NTK-by-Parts Rotary Embedding introduced in Section 3.2 in https://arxiv.org/pdf/2309.00071
        # Implementation copied from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py#L310
        if self.cfg.use_NTK_by_parts_rope:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim)
            )
            factor = self.cfg.NTK_by_parts_factor
            low_freq_factor = self.cfg.NTK_by_parts_low_freq_factor
            high_freq_factor = self.cfg.NTK_by_parts_high_freq_factor
            old_context_len = n_ctx

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            freq = 1 / inv_freq_llama
        else:
            freq = base ** (dim / (rotary_dim / 2))
            
        # Add the rotary_adjacent_pairs check
        if hasattr(self.cfg, 'rotary_adjacent_pairs') and self.cfg.rotary_adjacent_pairs:
            freq = einops.repeat(freq, "d -> (d 2)")
        else:
            freq = einops.repeat(freq, "d -> (2 d)")
            
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    def rotate_every_two(
        self, x: Float[torch.Tensor, "... rotary_dim"]
    ) -> Float[torch.Tensor, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        if hasattr(self.cfg, 'rotary_adjacent_pairs') and self.cfg.rotary_adjacent_pairs:
            # GPT-J style: rotate adjacent pairs
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]
        else:
            # GPT-NeoX style: rotate pairs at k and k+n//2
            n = x.size(-1) // 2
            if n > 0:  # Only rotate if we have pairs to rotate
                rot_x[..., :n] = -x[..., n:2*n]
                rot_x[..., n:2*n] = x[..., :n]
        return rot_x

    def apply_rotary(
        self,
        x: Float[torch.Tensor, "batch pos head_index d_head"],
        past_kv_pos_offset=0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # Only apply rotary to first rotary_dim dimensions
        if x.device != self.rotary_sin.device:
            x = x.to(self.rotary_sin.device)

        x_pos = x.size(1)
        x_rot = x[..., :self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim:]
        x_flip = self.rotate_every_two(x_rot)

        if attention_mask is None:
            # [pos, 1, rotary_dim]
            rotary_cos = self.rotary_cos[past_kv_pos_offset:past_kv_pos_offset + x_pos]  # [pos, 1, rotary_dim]
            rotary_sin = self.rotary_sin[past_kv_pos_offset:past_kv_pos_offset + x_pos]  # [pos, 1, rotary_dim]
            
            # Reshape rotary tensors to match input dimensions [1, pos, 1, rotary_dim]
            rotary_cos = rotary_cos.unsqueeze(0)  # [1, pos, 1, rotary_dim]
            rotary_sin = rotary_sin.unsqueeze(0)  # [1, pos, 1, rotary_dim]
            
            # Expand rotary tensors to match input dimensions
            rotary_cos = rotary_cos.expand(x_rot.size(0), -1, x_rot.size(2), -1)
            rotary_sin = rotary_sin.expand(x_rot.size(0), -1, x_rot.size(2), -1)
            
            # Apply rotary embeddings
            x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
        else:
            offset_position_ids = get_offset_position_ids(past_kv_pos_offset, attention_mask)
            offset_position_ids = offset_position_ids.to(self.rotary_cos.device)
            
            # [batch, pos, 1, rotary_dim]
            mask_rotary_cos = self.rotary_cos[offset_position_ids]  # [batch, pos, 1, rotary_dim]
            mask_rotary_sin = self.rotary_sin[offset_position_ids]  # [batch, pos, 1, rotary_dim]
            
            # Expand rotary tensors to match input dimensions
            mask_rotary_cos = mask_rotary_cos.expand(-1, -1, x_rot.size(2), -1)
            mask_rotary_sin = mask_rotary_sin.expand(-1, -1, x_rot.size(2), -1)
            
            # Apply rotary embeddings
            x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin

        return torch.cat([x_rotated, x_pass], dim=-1)

    @property
    def _W_K(self) -> torch.Tensor:
        """Get the key weight matrix for grouped query attention."""
        return self.W_K

    @property
    def _b_K(self) -> torch.Tensor:
        """Get the key bias for grouped query attention."""
        return self.b_K

    @property
    def _W_V(self) -> torch.Tensor:
        """Get the value weight matrix for grouped query attention."""
        return self.W_V

    @property
    def _b_V(self) -> torch.Tensor:
        """Get the value bias for grouped query attention."""
        return self.b_V

    @staticmethod
    def create_alibi_slope(
        n_ctx: int,
        device: Optional[torch.device] = None,
    ) -> Float[torch.Tensor, "query key"]:
        """Create the slope tensor for ALiBi positional embeddings.
        
        Args:
            n_ctx (int): Maximum number of tokens in a prompt
            device (Optional[torch.device], optional): Device to create the tensor on. Defaults to None.
            
        Returns:
            Float[torch.Tensor, "query key"]: The ALiBi slope tensor
        """
        # Create a matrix of indices [0, 1, ..., n_ctx-1]
        indices = torch.arange(n_ctx, device=device)
        
        # Create a matrix where each row is [0, 1, ..., n_ctx-1]
        # and each column is [0, 0, ..., 0]
        # Then subtract to get distances
        key_pos = indices.view(1, -1)  # [1, n_ctx]
        query_pos = indices.view(-1, 1)  # [n_ctx, 1]
        
        # Calculate distances between query and key positions
        distance = key_pos - query_pos  # [n_ctx, n_ctx]
        
        # Create a lower triangular matrix of distances
        slope = torch.tril(distance)
        
        return slope

    @staticmethod
    def create_alibi_multipliers(
        n_heads: int, device: Optional[Union[str, torch.device]] = None
    ) -> Float[torch.Tensor, "head_index"]:
        """Create the ALiBi Scalar Multipliers for each Head.

        For n heads, the set of multipliers (m) is the geometric sequence that starts at 2^(-8/n), and
        uses that same value as its ratio. For example, with 8 heads the values would be [1/(2^1),
        1/(2^2), ... , 1/(2^8)]. With 16 heads the values would be [1/(2^0.5), 1/(2^1), ... , 1/(2^8)].

        See :meth:`create_alibi_bias` for the full ALiBi bias calculation.

        Examples:

        >>> AlternatingAttention.create_alibi_multipliers(8)
        tensor([0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039])

        >>> AlternatingAttention.create_alibi_multipliers(16)
        tensor([0.7071, 0.5000, 0.3536, 0.2500, 0.1768, 0.1250, 0.0884, 0.0625, 0.0442, 0.0312,
                0.0221, 0.0156, 0.0110, 0.0078, 0.0055, 0.0039])

        Args:
            n_heads: The number of heads in a layer.
            device: The device to create the tensor on.

        Returns:
            A tensor of shape (head_index,) containing the scalar multiplier for each head.
        """
        # Calculate the starting value
        start = 2 ** (-8 / n_heads)

        # Generate the indices [0, 1, ..., n_heads-1]
        indices = torch.arange(n_heads, device=device)

        # Compute the multipliers, with the starting value being the same as the ratio
        multipliers = start * (start**indices)

        return multipliers

    @staticmethod
    def create_alibi_bias(
        n_heads: int, n_ctx: int, device: Optional[Union[torch.device, str]] = None
    ) -> Float[torch.Tensor, "head_index query key"]:
        """Create the ALiBi Bias for all Heads.

        Calculate the ALiBi bias (https://arxiv.org/pdf/2108.12409.pdf) for all heads in a layer.

        The broad idea behind ALiBi is to remove the positional encoding from the original transformer
        model, and instead apply a bias to each attention score. This bias is proportional to the
        distance between the query and key (i.e. it encourage paying less attention to more distant
        tokens), and is added to the attention scores before the softmax. It is used in models such as
        Bloom.

        Examples:

        >>> AlternatingAttention.create_alibi_bias(2, 4, torch.device('cpu'))
        tensor([[[ 0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0625,  0.0000,  0.0000,  0.0000],
            [-0.1250, -0.0625,  0.0000,  0.0000],
            [-0.1875, -0.1250, -0.0625,  0.0000]],
            [[ 0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0039,  0.0000,  0.0000,  0.0000],
            [-0.0078, -0.0039,  0.0000,  0.0000],
            [-0.0117, -0.0078, -0.0039,  0.0000]]])

        Args:
            n_heads: The number of heads in a layer.
            n_ctx: The maximum number of tokens in a prompt.
            device: The device to create the tensor on.

        Returns:
            The ALiBi bias that should be added to the attention scores before the softmax.
        """
        # Create the slope matrix
        slope: Float[torch.Tensor, "query key"] = AlternatingAttention.create_alibi_slope(
            n_ctx, device
        )

        # Create the scalar multiplier for each head.
        multipliers: Float[torch.Tensor, "head_index"] = AlternatingAttention.create_alibi_multipliers(
            n_heads, device
        )

        # Add singleton dimensions to make shapes compatible for broadcasting:
        slope = einops.rearrange(slope, "query key -> 1 query key")
        multipliers = einops.rearrange(multipliers, "head_index -> head_index 1 1")

        # Element-wise multiplication of the slope and multipliers
        alibi_bias = multipliers * slope

        return alibi_bias
