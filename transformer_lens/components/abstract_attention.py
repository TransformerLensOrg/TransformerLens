import math
from abc import ABC
from typing import Dict, Optional, Tuple, Union, cast

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from better_abc import abstract_attribute
from jaxtyping import Float, Int
from torch import Tensor
from transformers.utils.import_utils import is_bitsandbytes_available

from transformer_lens.cache.key_value_cache_entry import (
    TransformerLensKeyValueCacheEntry,
)
from transformer_lens.components.rms_norm import RMSNorm
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint
from transformer_lens.utilities.attention import complex_attn_linear, simple_attn_linear
from transformer_lens.utils import get_offset_position_ids

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import Params4bit


class AbstractAttention(ABC, nn.Module):
    alibi: Union[torch.Tensor, None]
    q_norm: Optional[RMSNorm]
    k_norm: Optional[RMSNorm]
    mask: torch.Tensor
    IGNORE: torch.Tensor
    rotary_sin: torch.Tensor
    rotary_cos: torch.Tensor

    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        attn_type: str = "global",
        layer_id: Optional[int] = None,
    ):
        """Abstract Base Class of Attention Blocks, featuring common functionality of both Attention and GroupedQueryAttention blocks.

        Query and Output projections are defined in this class as they are the same for regular and grouped query attention.
        Attributes related to Key and Value projections are abstract as their implementations may differ. For example, in GroupedQueryAttention there are less query and key heads than value heads.
        To enforce implementation of W_K, W_V, b_K, and b_V by child classes, the better_abc.abstract_attribute class is used. See here for details: https://stackoverflow.com/questions/23831510/abstract-attribute-not-property.

        Args:
            cfg (Union[Dict, HookedTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistral models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)

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
        self.W_K = abstract_attribute()
        self.W_V = abstract_attribute()

        self.b_Q = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self.b_K: nn.Parameter = abstract_attribute()
        self.b_V: nn.Parameter = abstract_attribute()
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        if self.cfg.use_qk_norm:
            self.q_norm = RMSNorm(self.cfg, length=self.cfg.d_head)
            self.k_norm = RMSNorm(self.cfg, length=self.cfg.d_head)

        elif self.cfg.original_architecture in (
            "OlmoeForCausalLM",
            "Olmo2ForCausalLM",
            "Olmo3ForCausalLM",
        ):
            # Q/K norms applied on full projected vectors (before head reshape).
            # q_norm dim = n_heads * d_head = d_model
            self.q_norm: Optional[RMSNorm] = RMSNorm(self.cfg, self.cfg.d_model)
            # k_norm dim depends on whether GQA is used:
            #   OLMo 2 (MHA): n_kv_heads == n_heads, so d_model
            #   OLMo 3 / OLMoE (GQA): n_kv_heads * d_head
            if self.cfg.n_key_value_heads is not None:
                k_norm_dim = self.cfg.d_head * self.cfg.n_key_value_heads
            else:
                k_norm_dim = self.cfg.d_model
            self.k_norm: Optional[RMSNorm] = RMSNorm(self.cfg, k_norm_dim)
        else:
            self.q_norm = None
            self.k_norm = None

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
            # Use per-layer RoPE base if specified (e.g., Gemma 3 uses 10k for local, 1M for global)
            if self.cfg.rotary_base_local is not None and self.attn_type == "local":
                rope_base = self.cfg.rotary_base_local
            else:
                rope_base = self.cfg.rotary_base
            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                base=rope_base,
                dtype=self.cfg.dtype,
            )
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)
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
            Float[torch.Tensor, "batch kv_pos kv_head_index d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
            Float[torch.Tensor, "batch kv_pos kv_head_index d_model"],
        ],
        past_kv_cache_entry: Optional[TransformerLensKeyValueCacheEntry] = None,
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 kv_pos"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
        position_bias: Optional[Float[torch.Tensor, "1 head_index pos kv_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        shortformer_pos_embed is only used if self.cfg.positional_embedding_type == "shortformer", else defaults to None and is irrelevant. See HookedTransformerConfig for more details
        past_kv_cache_entry is an optional entry of past keys and values for this layer, only relevant if generating text. Defaults to None
        additive_attention_mask is an optional mask to add to the attention weights. Defaults to None.
        attention_mask is the attention mask for padded tokens. Defaults to None.
        """

        q, k, v = self.calculate_qkv_matrices(query_input, key_input, value_input)

        # OLMo-family QK-norm: applied on full projected vectors before head reshape.
        if self.cfg.original_architecture in (
            "OlmoeForCausalLM",
            "Olmo2ForCausalLM",
            "Olmo3ForCausalLM",
        ):
            assert self.q_norm is not None
            assert self.k_norm is not None
            q = einops.rearrange(
                self.q_norm(
                    einops.rearrange(
                        q,
                        "batch pos head_index d_head -> batch pos (head_index d_head)",
                    )
                ),
                "batch kv_pos (head_index d_head) -> batch kv_pos head_index d_head",
                head_index=q.shape[2],
            )
            k = einops.rearrange(
                self.k_norm(
                    einops.rearrange(
                        k,
                        "batch pos head_index d_head -> batch pos (head_index d_head)",
                    )
                ),
                "batch kv_pos (head_index d_head) -> batch kv_pos head_index d_head",
                head_index=k.shape[2],
            )

        if past_kv_cache_entry is not None:
            # Appends the new keys and values to the cached values, and automatically updates the cache
            kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
            k, v = past_kv_cache_entry.append(k, v)
        else:
            # Not using a cache
            kv_cache_pos_offset = 0

        if self.cfg.positional_embedding_type == "rotary":
            q = self.hook_rot_q(self.apply_rotary(q, kv_cache_pos_offset, attention_mask))
            k = self.hook_rot_k(
                self.apply_rotary(k, 0, attention_mask)
            )  # keys are cached so no offset

        if self.cfg.dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        attn_scores = self.calculate_attention_scores(
            q, k
        )  # [batch, head_index, query_pos, key_pos]

        if self.cfg.positional_embedding_type == "alibi":
            query_ctx = attn_scores.size(-2)
            # The key context length is the number of positions in the past - this includes all positions in the cache
            key_ctx = attn_scores.size(-1)

            # only recompute when necessary to increase efficiency.
            if self.alibi is None or key_ctx > self.alibi.size(-1):
                self.alibi = AbstractAttention.create_alibi_bias(
                    self.cfg.n_heads, key_ctx, self.cfg.device
                )

            # Take the last query_ctx positions so it also works with past_kv_cache
            if isinstance(self.alibi, torch.Tensor):
                attn_scores += self.alibi[:, -query_ctx:, :key_ctx]
            else:
                raise TypeError(
                    f"Expected self.alibi to be a Tensor, but got {type(self.alibi)}"
                )  # [batch, head_index, query_pos, key_pos]
        elif self.cfg.positional_embedding_type == "relative_positional_bias":
            if position_bias is None:
                if self.has_relative_attention_bias:
                    raise ValueError("Positional bias is required for relative_positional_bias")
                else:
                    position_bias = torch.zeros(
                        1,
                        self.cfg.n_heads,
                        attn_scores.shape[2],
                        attn_scores.shape[3],
                        device=attn_scores.device,
                    )

            if position_bias is not None:  # Add None check
                attn_scores += position_bias
        if self.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset, attention_mask
            )  # [batch, head_index, query_pos, key_pos]
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        pattern = F.softmax(attn_scores, dim=-1)
        if not isinstance(pattern, torch.Tensor):
            raise TypeError(f"Expected 'pattern' to be a Tensor, got {type(pattern)}")
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)  # [batch, head_index, query_pos, key_pos]
        pattern = pattern.to(device=v.device, dtype=v.dtype)
        z = self.calculate_z_scores(v, pattern)  # [batch, pos, head_index, d_head]
        if not self.cfg.use_attn_result:
            if self.cfg.load_in_4bit:
                # call bitsandbytes method to dequantize and multiply
                out = (
                    bnb.matmul_4bit(
                        z.reshape(z.shape[0], z.shape[1], self.cfg.d_head * self.cfg.n_heads),
                        self.W_O.t(),
                        # bias=self.W_O.t(),
                        bias=None,
                        quant_state=self.W_O.quant_state,
                    )
                    + self.b_O
                )
            else:
                w = einops.rearrange(
                    self.W_O, "head_index d_head d_model -> d_model (head_index d_head)"
                )

                # Move output projection weights and bias to the same device as z
                # so that the final linear operation occurs on the device of the inputs
                if w.device != z.device:
                    w = w.to(z.device)
                b_O: Tensor = self.b_O
                if b_O.device != z.device:
                    b_O = b_O.to(z.device)
                # Ensure z has the same dtype as weights used in the output projection
                if z.dtype != w.dtype:
                    z = z.to(w.dtype)

                out = F.linear(
                    z.reshape(z.shape[0], z.shape[1], self.cfg.d_head * self.cfg.n_heads),
                    w,
                    b_O,
                )
        else:
            # Explicitly calculate the attention result so it can be accessed by a hook
            # This is off by default because it can easily eat through your GPU memory.
            if self.cfg.load_in_4bit:
                result = self.hook_result(
                    bnb.matmul_4bit(
                        z.reshape(z.shape[0], z.shape[1], self.cfg.d_head * self.cfg.n_heads),
                        self.W_O.t(),
                        bias=None,
                        quant_state=self.W_O.quant_state,
                    )
                )
            else:
                # Add singleton dimensions to make shapes compatible for broadcasting:
                w = einops.rearrange(
                    self.W_O,
                    "head_index d_head d_model -> 1 1 head_index d_head d_model",
                )
                if w.device != z.device:
                    w = w.to(z.device)
                # Ensure z has the same dtype as w before multiplication
                if z.dtype != w.dtype:
                    z = z.to(w.dtype)
                z = einops.rearrange(
                    z, "batch pos head_index d_head -> batch pos head_index d_head 1"
                )

                # Multiply the z tensor by the W_O tensor, summing over the d_head dimension
                unhooked_result = (z * w).sum(-2)

                result = self.hook_result(unhooked_result)  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(result, "batch position index model->batch position model", "sum")
                + self.b_O
            )  # [batch, pos, d_model]
        return out

    def _apply_qk_norm(
        self, x: Float[torch.Tensor, "batch pos head_index d_head"], norm_module: RMSNorm
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        """Apply QK normalization with proper reshaping.

        Args:
            x: Input tensor with shape [batch, pos, head_index, d_head]
            norm_module: RMSNorm module to apply

        Returns:
            Normalized tensor with same shape as input
        """
        # Reshape from [batch, pos, head_index, d_head] to [batch * pos * head_index, d_head]
        d_head = x.shape[-1]
        x_normed = norm_module(x.reshape(-1, d_head))
        return x_normed.reshape(x.shape)

    def calculate_qkv_matrices(
        self,
        query_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
        key_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
        ],
    ) -> Tuple[
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch kv_pos head_index d_head"],
        Float[torch.Tensor, "batch kv_pos head_index d_head"],
    ]:
        attn_fn = (
            complex_attn_linear
            if self.cfg.use_split_qkv_input or self.cfg.use_attn_in
            else simple_attn_linear
        )
        if self.cfg.load_in_4bit:
            q = self.hook_q(
                # call bitsandbytes method to dequantize and multiply
                bnb.matmul_4bit(
                    query_input,
                    self.W_Q.t(),
                    bias=None,
                    quant_state=self.W_Q.quant_state,
                ).reshape(
                    query_input.shape[0],
                    query_input.shape[1],
                    self.cfg.n_heads,
                    self.cfg.d_head,
                )
                + self.b_Q
            )
        else:
            q = self.hook_q(attn_fn(query_input, self.W_Q, self.b_Q))
        if self.cfg.load_in_4bit:
            if not isinstance(self.W_K, Params4bit):
                raise ValueError("W_K must be a Params4bit object if load_in_4bit is True")
            k = self.hook_k(
                # call bitsandbytes method to dequantize and multiply
                bnb.matmul_4bit(
                    key_input, self.W_K.t(), bias=None, quant_state=self.W_K.quant_state
                ).reshape(
                    key_input.shape[0],
                    key_input.shape[1],
                    self.cfg.n_heads,
                    self.cfg.d_head,
                )
                + self.b_K
            )
        else:
            k = self.hook_k(attn_fn(key_input, self.W_K, self.b_K))

        if self.cfg.load_in_4bit:
            if not isinstance(self.W_V, Params4bit):
                raise ValueError("W_V must be a Params4bit object if load_in_4bit is True")
            v = self.hook_v(
                # call bitsandbytes method to dequantize and multiply
                bnb.matmul_4bit(
                    value_input,
                    self.W_V.t(),
                    bias=None,
                    quant_state=self.W_V.quant_state,
                ).reshape(
                    value_input.shape[0],
                    value_input.shape[1],
                    self.cfg.n_heads,
                    self.cfg.d_head,
                )
                + self.b_V
            )
        else:
            v = self.hook_v(attn_fn(value_input, self.W_V, self.b_V))

        if self.cfg.use_qk_norm:
            assert self.q_norm is not None
            assert self.k_norm is not None
            q = self._apply_qk_norm(q, self.q_norm)
            k = self._apply_qk_norm(k, self.k_norm)

        return q, k, v

    def calculate_attention_scores(
        self,
        q: Float[torch.Tensor, "batch query_pos head_index d_head"],
        k: Float[torch.Tensor, "batch key_pos head_index d_head"],
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        q_ = einops.rearrange(
            q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
        )
        k_ = einops.rearrange(
            k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
        )
        attn_scores = q_ @ k_ / self.attn_scale
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
        v_ = einops.rearrange(
            v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
        )
        pattern_ = einops.rearrange(
            pattern,
            "batch head_index query_pos key_pos -> batch head_index query_pos key_pos",
        )
        z = self.hook_z(
            einops.rearrange(
                pattern_ @ v_,
                "batch head_index query_pos d_head -> batch query_pos head_index d_head",
            )
        )
        return z

    def apply_causal_mask(
        self,
        attn_scores: Float[torch.Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it can be different.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        if query_ctx_length + past_kv_pos_offset != key_ctx_length:
            raise ValueError(
                f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
            )

        # Dynamically extend mask if needed for long context
        if key_ctx_length > self.mask.shape[0]:
            self._extend_mask(key_ctx_length)

        # Index back to front to ensure local attention works
        final_mask = cast(torch.Tensor, self.mask)[
            None, None, -query_ctx_length:, -key_ctx_length:
        ]  # [1, 1, pos, pos]
        if attention_mask is not None:
            # Apply a causal mask to the attention scores considering the padding

            # Add singleton dimensions to the attention mask to match the shape of the final mask
            attention_mask = einops.rearrange(
                attention_mask, "batch offset_pos -> batch 1 1 offset_pos"
            )

            final_mask = final_mask.to(attention_mask.device)

            # Element-wise multiplication of the final mask and the attention mask and cast to boolean
            final_mask = (final_mask * attention_mask).bool()  # [batch, head, pos, offset_pos]

        attn_scores = attn_scores.to(final_mask.device)
        return torch.where(final_mask, attn_scores, cast(torch.Tensor, self.IGNORE))

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

        # Llama-3.1 uses NTK-by-Parts Rotary Embedding introduced in Section 3.2 in https://arxiv.org/pdf/2309.00071
        # Implementation copied from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py#L310
        if self.cfg.use_NTK_by_parts_rope:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim)
            )
            factor = self.cfg.NTK_by_parts_factor
            low_freq_factor = self.cfg.NTK_by_parts_low_freq_factor
            high_freq_factor = self.cfg.NTK_by_parts_high_freq_factor
            old_context_len = self.cfg.NTK_original_ctx_len

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
        elif self.cfg.use_yarn_rope:
            # YARN (Yet Another RoPE extensioN) from https://arxiv.org/abs/2309.00071
            # Implementation follows HuggingFace: transformers/modeling_rope_utils.py
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=high_precision) / rotary_dim)
            )
            yarn_factor = self.cfg.yarn_factor
            # HF uses original_max_position_embeddings (the pre-extension context length)
            # for computing the correction range.
            orig_max_pos = self.cfg.yarn_original_max_position_embeddings
            beta_fast = self.cfg.yarn_beta_fast
            beta_slow = self.cfg.yarn_beta_slow

            def _find_correction_dim(num_rotations: float) -> float:
                return (rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi))) / (
                    2 * math.log(base)
                )

            low = math.floor(_find_correction_dim(beta_fast))
            high = math.ceil(_find_correction_dim(beta_slow))
            low = max(low, 0)
            high = min(high, rotary_dim - 1)

            # Linear ramp from 0 to 1 between low and high dims
            ramp = torch.arange(rotary_dim // 2, dtype=high_precision)
            high_f = float(high) + 0.001 if low == high else float(high)
            ramp = torch.clamp((ramp - low) / (high_f - low), 0, 1)

            inv_freq_interp = inv_freq / yarn_factor
            # ramp=0 (below low) → extrapolation (original freq), ramp=1 (above high) → interpolation (scaled)
            inv_freq = inv_freq_interp * ramp + inv_freq * (1 - ramp)
            freq = 1.0 / inv_freq
        else:
            freq = base ** (dim / (rotary_dim / 2))
        if self.cfg.rotary_adjacent_pairs:
            freq = einops.repeat(freq, "d -> (d 2)")
        else:
            freq = einops.repeat(freq, "d -> (2 d)")
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        sin, cos = torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)
        # YARN attention_factor scales the embeddings (default 1.0 is a no-op)
        if self.cfg.use_yarn_rope and self.cfg.yarn_attention_factor != 1.0:
            sin = sin * self.cfg.yarn_attention_factor
            cos = cos * self.cfg.yarn_attention_factor
        return sin, cos

    def rotate_every_two(
        self, x: Float[torch.Tensor, "... rotary_dim"]
    ) -> Float[torch.Tensor, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        if self.cfg.rotary_adjacent_pairs:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]
        else:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]

        return rot_x

    def apply_rotary(
        self,
        x: Float[torch.Tensor, "batch pos head_index d_head"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)

        if x.device != self.rotary_sin.device:
            x = x.to(cast(torch.device, self.rotary_sin.device))

        x_pos = x.size(1)
        x_rot = x[..., : self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)

        # Dynamically extend rotary embeddings if needed for long context
        max_pos_needed = past_kv_pos_offset + x_pos
        if max_pos_needed > self.rotary_cos.shape[0]:
            self._extend_rotary_embeddings(max_pos_needed)

        if attention_mask is None:
            rotary_cos = cast(torch.Tensor, self.rotary_cos)[
                None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
            ]
            rotary_sin = cast(torch.Tensor, self.rotary_sin)[
                None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
            ]
            x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
        else:
            offset_position_ids = get_offset_position_ids(past_kv_pos_offset, attention_mask)
            offset_position_ids = offset_position_ids.to(cast(torch.device, self.rotary_cos.device))
            mask_rotary_cos = cast(torch.Tensor, self.rotary_cos)[offset_position_ids, None, :]
            mask_rotary_sin = cast(torch.Tensor, self.rotary_sin)[offset_position_ids, None, :]
            x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin

        return torch.cat([x_rotated, x_pass], dim=-1)

    def _extend_rotary_embeddings(self, new_size: int):
        """Extend rotary embeddings to support longer contexts dynamically."""
        # Get the RoPE base from config or use default
        rope_base = getattr(self.cfg, "rotary_base", 10000)

        # Ensure rotary_dim is set
        assert self.cfg.rotary_dim is not None, "rotary_dim must be set for rotary embeddings"

        # Calculate new embeddings
        sin, cos = self.calculate_sin_cos_rotary(
            self.cfg.rotary_dim,
            new_size,
            base=rope_base,
            dtype=self.cfg.dtype,
        )

        # Update the registered buffers
        self.rotary_sin = sin.to(self.rotary_sin.device)
        self.rotary_cos = cos.to(self.rotary_cos.device)

    def _extend_mask(self, new_size: int):
        """Extend causal mask to support longer contexts dynamically."""
        causal_mask = torch.tril(torch.ones((new_size, new_size), device=self.mask.device).bool())
        if self.attn_type == "global":
            self.mask = causal_mask
        elif self.attn_type == "local":
            if not isinstance(self.cfg.window_size, int):
                raise ValueError("Window size must be an integer for local attention")
            self.mask = torch.triu(causal_mask, 1 - self.cfg.window_size)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

    @staticmethod
    def create_alibi_slope(
        n_ctx: int, device: Optional[Union[str, torch.device]] = None
    ) -> Float[torch.Tensor, "query key"]:
        """Create an ALiBi Slope Matrix.

        Create the slope matrix used in ALiBi, before it is multiplied by the head-specific scalar.

        See :meth:`create_alibi_bias` for the full ALiBi bias calculation.

        Examples:

        >>> AbstractAttention.create_alibi_slope(3)
        tensor([[ 0.,  0.,  0.],
                [-1.,  0.,  0.],
                [-2., -1.,  0.]])

        >>> AbstractAttention.create_alibi_slope(4)
        tensor([[ 0.,  0.,  0.,  0.],
                [-1.,  0.,  0.,  0.],
                [-2., -1.,  0.,  0.],
                [-3., -2., -1.,  0.]])

        Args:
            n_ctx: The maximum number of tokens in a prompt.

        Returns:
            A tensor of shape (n_ctx, n_ctx), where the upper triangle is zero and the lower
            triangle is decreasing by a constant slope of 1 (towards the bottom left corner).
        """
        # set rows as [[0,1,2...]]
        rows = torch.arange(n_ctx, device=device).unsqueeze(0)

        # Set cols as [[0],[1],[2]...]
        cols = torch.arange(n_ctx, device=device).unsqueeze(1)

        # Use broadcasting to create the desired lower triangular part of the matrix
        slope_matrix = rows - cols

        # Use the clamp method to set all positive values (upper right triangle) to
        return slope_matrix.clamp(max=0).to(torch.float32)

    @staticmethod
    def create_alibi_multipliers(
        n_heads: int, device: Optional[Union[str, torch.device]] = None
    ) -> Float[torch.Tensor, "n_heads"]:
        """Create the ALiBi Scalar Multipliers for each Head.

        For n heads, the set of multipliers (m) is the geometric sequence that starts at 2^(-8/n), and
        uses that same value as its ratio. For example, with 8 heads the values would be [1/(2^1),
        1/(2^2), ... , 1/(2^8)]. With 16 heads the values would be [1/(2^0.5), 1/(2^1), ... , 1/(2^8)].

        See :meth:`create_alibi_bias` for the full ALiBi bias calculation.

        Examples:

        >>> AbstractAttention.create_alibi_multipliers(8)
        tensor([0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039])

        >>> AbstractAttention.create_alibi_multipliers(16)
        tensor([0.7071, 0.5000, 0.3536, 0.2500, 0.1768, 0.1250, 0.0884, 0.0625, 0.0442, 0.0312,
                0.0221, 0.0156, 0.0110, 0.0078, 0.0055, 0.0039])

        Args:
            n_heads: The number of heads in a layer.
            device: The device to create the tensor on.

        Returns:
            A tensor of shape (n_heads,) containing the scalar multiplier for each head.
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
    ) -> Float[torch.Tensor, "head_idx query key"]:
        """Create the ALiBi Bias for all Heads.

        Calculate the ALiBi bias (https://arxiv.org/pdf/2108.12409.pdf) for all heads in a layer.

        The broad idea behind ALiBi is to remove the positional encoding from the original transformer
        model, and instead apply a bias to each attention score. This bias is proportional to the
        distance between the query and key (i.e. it encourage paying less attention to more distant
        tokens), and is added to the attention scores before the softmax. It is used in models such as
        Bloom.

        Examples:

        >>> AbstractAttention.create_alibi_bias(2, 4, torch.device('cpu'))
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
        slope: Float[torch.Tensor, "query key"] = AbstractAttention.create_alibi_slope(
            n_ctx, device
        )

        # Create the scalar multiplier for each head.
        multipliers: Float[torch.Tensor, "head_idx"] = AbstractAttention.create_alibi_multipliers(
            n_heads, device
        )

        # Add singleton dimensions to make shapes compatible for broadcasting:
        slope = einops.rearrange(slope, "query key -> 1 query key")
        multipliers = einops.rearrange(multipliers, "head_idx -> head_idx 1 1")

        # Element-wise multiplication of the slope and multipliers
        alibi_bias = multipliers * slope

        return alibi_bias
