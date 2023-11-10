"""Hooked Transformer Components.

This module contains all the components (e.g. :class:`Attention`, :class:`MLP`, :class:`LayerNorm`)
needed to create many different types of generative language models. They are used by
:class:`transformer_lens.HookedTransformer`.
"""
import logging
from typing import Dict, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from jaxtyping import Float, Int

from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
from transformer_lens.utils import gelu_fast, gelu_new, get_offset_position_ids, solu


# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=cfg.dtype)
        )
        # Some models (e.g. Bloom) need post embedding layer norm
        if cfg.post_embedding_ln:
            self.ln = LayerNorm(cfg)

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        if self.cfg.post_embedding_ln:
            return self.ln(self.W_E[tokens, :])
        return self.W_E[tokens, :]


class Unembed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        # Note that there's a separate variable for d_vocab_out and d_vocab (the input vocab size). For language tasks these are always the same, but for algorithmic tasks we may want them to be different.
        self.W_U: Float[torch.Tensor, "d_model d_vocab_out"] = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_vocab_out, dtype=cfg.dtype)
        )
        self.b_U: Float[torch.Tensor, "d_vocab_out"] = nn.Parameter(
            torch.zeros(self.cfg.d_vocab_out, dtype=cfg.dtype)
        )

    def forward(
        self, residual: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_vocab_out"]:
        return (
            einsum(
                "batch pos d_model, d_model vocab -> batch pos vocab",
                residual,
                self.W_U,
            )
            + self.b_U
        )


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(
            torch.empty(self.cfg.n_ctx, self.cfg.d_model, dtype=cfg.dtype)
        )

    def forward(
        self,
        tokens: Int[torch.Tensor, "batch pos"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        Forward pass for positional embeddings.

        Args:
            tokens (Int[torch.Tensor, "batch pos"]): Input tokens.
            past_kv_pos_offset (int, optional): The length of tokens in the past_kv_cache. Defaults to 0.
            attention_mask (Int[torch.Tensor, "batch pos"], optional): The attention mask for padded tokens.
                 Defaults to None.

        Returns:
            Float[torch.Tensor, "batch pos d_model"]: Absolute position embeddings.
        """
        tokens_length = tokens.size(-1)

        if attention_mask is None:
            pos_embed = self.W_pos[
                past_kv_pos_offset : tokens_length + past_kv_pos_offset, :
            ]  # [pos, d_model]
            batch_pos_embed = einops.repeat(
                pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
            )

        else:
            # Separated from the no padding case for computational efficiency
            # (this code is a bit slower than the code above)

            offset_position_ids = get_offset_position_ids(
                past_kv_pos_offset, attention_mask
            )
            pos_embed = self.W_pos[offset_position_ids]  # [batch, pos, d_model]

            # Set the position embeddings to 0 for pad tokens (this is an arbitrary choice)
            padding_mask = ~attention_mask.bool()  # [batch, tokens_length]
            offset_padding_mask = padding_mask[
                :, past_kv_pos_offset : tokens_length + past_kv_pos_offset
            ].unsqueeze(
                -1
            )  # [batch, pos, 1]
            batch_pos_embed = torch.where(offset_padding_mask, 0, pos_embed)

        return batch_pos_embed.clone()


class TokenTypeEmbed(nn.Module):
    """
    The token-type embed is a binary ids indicating whether a token belongs to sequence A or B. For example, for two sentences: "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be [0, 0, ..., 0, 1, ..., 1, 1]. `0` represents tokens from Sentence A, `1` from Sentence B. If not provided, BERT assumes a single sequence input. Typically, shape is (batch_size, sequence_length).

    See the BERT paper for more information: https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_token_type = nn.Parameter(
            torch.empty(2, self.cfg.d_model, dtype=cfg.dtype)
        )

    def forward(self, token_type_ids: Int[torch.Tensor, "batch pos"]):
        return self.W_token_type[token_type_ids, :]


class BertEmbed(nn.Module):
    """
    Custom embedding layer for a BERT-like model. This module computes the sum of the token, positional and token-type embeddings and takes the layer norm of the result.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.token_type_embed = TokenTypeEmbed(cfg)
        self.ln = LayerNorm(cfg)

        self.hook_embed = HookPoint()
        self.hook_pos_embed = HookPoint()
        self.hook_token_type_embed = HookPoint()

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch pos"],
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ):
        base_index_id = torch.arange(input_ids.shape[1], device=input_ids.device)
        index_ids = einops.repeat(
            base_index_id, "pos -> batch pos", batch=input_ids.shape[0]
        )
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings_out = self.hook_embed(self.embed(input_ids))
        position_embeddings_out = self.hook_pos_embed(self.pos_embed(index_ids))
        token_type_embeddings_out = self.hook_token_type_embed(
            self.token_type_embed(token_type_ids)
        )

        embeddings_out = (
            word_embeddings_out + position_embeddings_out + token_type_embeddings_out
        )
        layer_norm_out = self.ln(embeddings_out)
        return layer_norm_out


class BertMLMHead(nn.Module):
    """
    Transforms BERT embeddings into logits. The purpose of this module is to predict masked tokens in a sentence.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W = nn.Parameter(torch.empty(cfg.d_model, cfg.d_model, dtype=cfg.dtype))
        self.b = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))
        self.act_fn = nn.GELU()
        self.ln = LayerNorm(cfg)

    def forward(self, resid: Float[torch.Tensor, "batch pos d_model"]) -> torch.Tensor:
        resid = (
            einsum(
                "batch pos d_model_in, d_model_out d_model_in -> batch pos d_model_out",
                resid,
                self.W,
            )
            + self.b
        )
        resid = self.act_fn(resid)
        resid = self.ln(resid)
        return resid


# LayerNormPre
# I fold the LayerNorm weights and biases into later weights and biases.
# This is just the 'center and normalise' part of LayerNorm
# Centering is equivalent to just deleting one direction of residual space,
# and is equivalent to centering the weight matrices of everything writing to the residual stream
# Normalising is a funkier non-linear operation, that projects the residual stream onto the unit hypersphere
class LayerNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        """LayerNormPre - the 'center and normalise' part of LayerNorm. Length is
        normally d_model, but is d_mlp for softmax. Not needed as a parameter. This
        should only be used in inference mode after folding in LayerNorm weights"""
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        # Hook Normalized captures LN output - here it's a vector with std 1 and mean 0
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch pos d_model"],
        Float[torch.Tensor, "batch pos head_index d_model"],
    ]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale: Union[
            Float[torch.Tensor, "batch pos 1"],
            Float[torch.Tensor, "batch pos head_index 1"],
        ] = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
        return self.hook_normalized(x / scale).to(self.cfg.dtype)


class LayerNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, HookedTransformerConfig], length: Optional[int] = None
    ):
        """
        LayerNorm with optional length parameter

        length (Optional[int]): If the dimension of the LayerNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length, dtype=cfg.dtype))
        self.b = nn.Parameter(torch.zeros(self.length, dtype=cfg.dtype))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        # Hook_normalized is on the LN output
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[torch.Tensor, "batch pos d_model"],
        Float[torch.Tensor, "batch pos head_index d_model"],
    ]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        x = x / scale  # [batch, pos, length]
        return self.hook_normalized(x * self.w + self.b).to(self.cfg.dtype)


class RMSNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        """RMSNormPre - LayerNormPre without the centering and bias (RMS = Root Mean Square)"""
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        return self.hook_normalized(x / scale).to(
            self.cfg.dtype
        )  # [batch, pos, length]


class RMSNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, HookedTransformerConfig], length: Optional[int] = None
    ):
        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length, dtype=cfg.dtype))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        x = self.hook_normalized(x / scale).to(self.cfg.dtype)  # [batch, pos, length]
        return x * self.w


# Attention
class Attention(nn.Module):
    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        attn_type: str = "global",
        layer_id: Optional[int] = None,
    ):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [batch, head_index, query_pos, key_pos]

        Args:
            cfg (Union[Dict, HookedTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistal models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_Q = nn.Parameter(
            torch.empty(
                self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=cfg.dtype
            )
        )
        self.W_K = nn.Parameter(
            torch.empty(
                self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=cfg.dtype
            )
        )
        self.W_V = nn.Parameter(
            torch.empty(
                self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=cfg.dtype
            )
        )
        self.W_O = nn.Parameter(
            torch.empty(
                self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model, dtype=cfg.dtype
            )
        )
        self.b_Q = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.b_K = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.b_V = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=cfg.dtype))

        self.attn_type = attn_type
        # Create a max_ctx x max_ctx mask, with True iff that query position
        # can attend to that key position (query is first axis, key is second axis)
        causal_mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool())
        if self.attn_type == "global":
            # For global attention, this is a lower triangular matrix - key <= query
            self.register_buffer("mask", causal_mask)
        elif self.attn_type == "local":
            # For local, this is banded, query - window_size < key <= query
            assert isinstance(self.cfg.window_size, int)
            self.register_buffer(
                "mask", torch.triu(causal_mask, 1 - self.cfg.window_size)
            )
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        self.register_buffer("IGNORE", torch.tensor(-torch.inf))

        self.layer_id = layer_id

        # attn_scale is a constant that we divide the attention scores by pre-softmax. I'm not entirely sure why it matters, but it's probably a mix of softmax not being scale invariant and numerical stability?
        if self.cfg.use_attn_scale:
            self.attn_scale = np.sqrt(self.cfg.d_head)
        else:
            self.attn_scale = 1.0
        if self.cfg.scale_attn_by_inverse_layer_idx:
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
            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim, self.cfg.n_ctx, dtype=self.cfg.dtype
            )
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)
        elif self.cfg.positional_embedding_type == "alibi":
            # ALiBi bias wil be constructed on the first forward pass.
            # Note: While computationally efficient, initializing an bias with max n_ctx (16, 1024, 1024) of float32 will occupy ~256MiB of contiguous GPU memory, which may not be optimal for memory usage.
            self.alibi = None

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
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 pos"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        shortformer_pos_embed is only used if self.cfg.positional_embedding_type == "shortformer", else defaults to None and is irrelevant. See HookedTransformerConfig for more details
        past_kv_cache_entry is an optional entry of past keys and values for this layer, only relevant if generating text. Defaults to None
        additive_attention_mask is an optional mask to add to the attention weights. Defaults to None.
        attention_mask is the attention mask for padded tokens. Defaults to None.
        """

        if self.cfg.use_split_qkv_input or self.cfg.use_attn_in:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"
        q = self.hook_q(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                key_input,
                self.W_K,
            )
            + self.b_K
        )  # [batch, pos, head_index, d_head]
        v = self.hook_v(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                value_input,
                self.W_V,
            )
            + self.b_V
        )  # [batch, pos, head_index, d_head]

        if past_kv_cache_entry is not None:
            # Appends the new keys and values to the cached values, and automatically updates the cache
            kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
            k, v = past_kv_cache_entry.append(k, v)
        else:
            # Not using a cache
            kv_cache_pos_offset = 0

        if self.cfg.positional_embedding_type == "rotary":
            q = self.hook_rot_q(
                self.apply_rotary(q, kv_cache_pos_offset, attention_mask)
            )
            k = self.hook_rot_k(
                self.apply_rotary(k, 0, attention_mask)
            )  # keys are cached so no offset

        if self.cfg.dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        attn_scores = (
            einsum(
                "batch query_pos head_index d_head, \
                    batch key_pos head_index d_head \
                    -> batch head_index query_pos key_pos",
                q,
                k,
            )
            / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]

        if self.cfg.positional_embedding_type == "alibi":
            query_ctx = attn_scores.size(-2)
            # The key context length is the number of positions in the past - this includes all positions in the cache
            key_ctx = attn_scores.size(-1)

            # only recompute when necessary to increase efficiency.
            if self.alibi is None or key_ctx > self.alibi.size(-1):
                self.alibi = Attention.create_alibi_bias(
                    self.cfg.n_heads, key_ctx, self.cfg.device
                )

            attn_scores += self.alibi[
                :, :query_ctx, :key_ctx
            ]  # [batch, head_index, query_pos, key_pos]

        if self.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset, attention_mask
            )  # [batch, head_index, query_pos, key_pos]
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)  # [batch, head_index, query_pos, key_pos]
        pattern = pattern.to(self.cfg.dtype)
        z = self.hook_z(
            einsum(
                "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
                v,
                pattern,
            )
        )  # [batch, pos, head_index, d_head]
        if not self.cfg.use_attn_result:
            out = (
                (
                    einsum(
                        "batch pos head_index d_head, \
                            head_index d_head d_model -> \
                            batch pos d_model",
                        z,
                        self.W_O,
                    )
                )
                + self.b_O
            )  # [batch, pos, d_model]
        else:
            # Explicitly calculate the attention result so it can be accessed by a hook
            # This is off by default because it can easily eat through your GPU memory.
            result = self.hook_result(
                einsum(
                    "batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos head_index d_model",
                    z,
                    self.W_O,
                )
            )  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(
                    result, "batch position index model->batch position model", "sum"
                )
                + self.b_O
            )  # [batch, pos, d_model]
        return out

    def apply_causal_mask(
        self,
        attn_scores: Float[
            torch.Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"
        ],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it can be different.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        assert (
            query_ctx_length + past_kv_pos_offset == key_ctx_length
        ), f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."

        # Index back to front to ensure local attention works
        final_mask = self.mask[
            None, None, -query_ctx_length:, -key_ctx_length:
        ]  # [1, 1, pos, pos]
        if attention_mask is not None:
            # Apply a causal mask to the attention scores considering the padding
            einsum_str = "batch head pos offset_pos, batch offset_pos -> batch head pos offset_pos"
            final_mask = einops.einsum(final_mask, attention_mask, einsum_str).bool()

        return torch.where(final_mask, attn_scores, self.IGNORE)

    def calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[
        Float[torch.Tensor, "n_ctx rotary_dim"], Float[torch.Tensor, "n_ctx rotary_dim"]
    ]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

        Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
        To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
        """
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)

        # A set of frequencies evenly spaced in log space
        freq = base ** (dim / (rotary_dim / 2))
        if self.cfg.original_architecture in ["GPTNeoXForCausalLM", "LlamaForCausalLM"]:
            freq = einops.repeat(freq, "d -> (2 d)")
        else:
            freq = einops.repeat(freq, "d -> (d 2)")
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
        if self.cfg.original_architecture in ["GPTNeoXForCausalLM", "LlamaForCausalLM"]:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]
        else:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]

        return rot_x

    def apply_rotary(
        self,
        x: Float[torch.Tensor, "batch pos head_index d_head"],
        past_kv_pos_offset=0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
        x_pos = x.size(1)
        x_rot = x[..., : self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)

        if attention_mask is None:
            rotary_cos = self.rotary_cos[
                None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
            ]
            rotary_sin = self.rotary_sin[
                None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
            ]
            x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
        else:
            offset_position_ids = get_offset_position_ids(
                past_kv_pos_offset, attention_mask
            )
            mask_rotary_cos = self.rotary_cos[offset_position_ids, None, :]
            mask_rotary_sin = self.rotary_sin[offset_position_ids, None, :]
            x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin

        return torch.cat([x_rotated, x_pass], dim=-1)

    @staticmethod
    def create_alibi_slope(
        n_ctx: int, device: torch.device = None
    ) -> Float[torch.Tensor, "query key"]:
        """Create an ALiBi Slope Matrix.

        Create the slope matrix used in ALiBi, before it is multiplied by the head-specific scalar.

        See :meth:`create_alibi_bias` for the full ALiBi bias calculation.

        Examples:

        >>> Attention.create_alibi_slope(3)
        tensor([[ 0.,  0.,  0.],
                [-1.,  0.,  0.],
                [-2., -1.,  0.]])

        >>> Attention.create_alibi_slope(4)
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
        n_heads: int, device: torch.device = None
    ) -> Float[torch.Tensor, "head_idx"]:
        """Create the ALiBi Scalar Multipliers for each Head.

        For n heads, the set of multipliers (m) is the geometric sequence that starts at 2^(-8/n), and
        uses that same value as its ratio. For example, with 8 heads the values would be [1/(2^1),
        1/(2^2), ... , 1/(2^8)]. With 16 heads the values would be [1/(2^0.5), 1/(2^1), ... , 1/(2^8)].

        See :meth:`create_alibi_bias` for the full ALiBi bias calculation.

        Examples:

        >>> Attention.create_alibi_multipliers(8)
        tensor([0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039])

        >>> Attention.create_alibi_multipliers(16)
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
        n_heads: int, n_ctx: int, device: torch.device = None
    ) -> Float[torch.Tensor, "head_idx query key"]:
        """Create the ALiBi Bias for all Heads.

        Calculate the ALiBi bias (https://arxiv.org/pdf/2108.12409.pdf) for all heads in a layer.

        The broad idea behind ALiBi is to remove the positional encoding from the original transformer
        model, and instead apply a bias to each attention score. This bias is proportional to the
        distance between the query and key (i.e. it encourage paying less attention to more distant
        tokens), and is added to the attention scores before the softmax. It is used in models such as
        Bloom.

        Examples:

        >>> Attention.create_alibi_bias(2, 4, torch.device('cpu'))
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
        slope: Float[torch.Tensor, "query key"] = Attention.create_alibi_slope(
            n_ctx, device
        )

        # Create the scalar multiplier for each head.
        multipliers: Float[
            torch.Tensor, "head_idx"
        ] = Attention.create_alibi_multipliers(n_heads, device)

        # The ALiBi bias is then m * slope_matrix
        alibi_bias = torch.einsum("ij,k->kij", slope, multipliers)

        return alibi_bias


# MLP Layers
class MLP(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_in = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype)
        )
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(
            torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=cfg.dtype)
        )
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=cfg.dtype))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "silu":
            self.act_fn = F.silu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.act_fn == "gelu_fast":
            self.act_fn = gelu_fast
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_in)
            + self.b_in
        )  # [batch, pos, d_mlp]
        if not self.cfg.act_fn.endswith("_ln"):
            post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        return (
            einsum(
                "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
                post_act,
                self.W_out,
            )
            + self.b_out
        )


# TODO
# not sure whether to fold this into MLP or not
class GatedMLP(nn.Module):
    """
    The equation of a gated MLP:
    pre = x @ W_gate
    pre_linear = x @ W_in
    post = Gelu(pre) * (pre_linear) + b_in
    mlp_out = post @ W_out + b_out

    In one equation, mlp_out = (Gelu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_in = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype)
        )
        self.W_gate = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=cfg.dtype)
        )
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(
            torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=cfg.dtype)
        )
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=cfg.dtype))

        # hook on gate output but before act_fn
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "silu":
            self.act_fn = F.silu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.act_fn == "gelu_fast":
            self.act_fn = gelu_fast
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            einsum(
                "batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_gate
            )
        )  # [batch, pos, d_mlp]
        if not self.cfg.act_fn.endswith("_ln"):
            pre_linear = self.hook_pre_linear(
                einsum(
                    "batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_in
                )
            )
            post_act = self.hook_post(
                (self.act_fn(pre_act) * pre_linear) + self.b_in
            )  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        return (
            einsum(
                "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
                post_act,
                self.W_out,
            )
            + self.b_out
        )


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig], block_index):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNorm(cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln1 = LayerNormPre(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNormPre(cfg)
        elif self.cfg.normalization_type == "RMS":
            self.ln1 = RMSNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = RMSNorm(cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln1 = RMSNormPre(cfg)
            if not self.cfg.attn_only:
                self.ln2 = RMSNormPre(cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
            if not self.cfg.attn_only:
                self.ln2 = nn.Identity()
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )

        if not self.cfg.use_local_attn:
            self.attn = Attention(cfg, "global", block_index)
        else:
            assert self.cfg.attn_types is not None
            attn_type = self.cfg.attn_types[block_index]
            self.attn = Attention(cfg, attn_type, block_index)
        if not self.cfg.attn_only:
            if self.cfg.gated_mlp:
                self.mlp = GatedMLP(cfg)
            else:
                self.mlp = MLP(cfg)

        self.hook_attn_in = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]

        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        shortformer_pos_embed: Optional[
            Float[torch.Tensor, "batch pos d_model"]
        ] = None,
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """A single Transformer block.

        Args:
            resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
            cache (HookedTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
            shortformer_pos_embed (torch.Tensor, optional): Only used for positional_embeddings_type == "shortformer". The positional embeddings. See HookedTransformerConfig for details. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask for padded tokens. Defaults to None.

        Returns:
            _type_: _description_
        """
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]

        def add_head_dimension(
            tensor: Float[torch.Tensor, "batch pos d_model"],
            clone_tensor=True,
            # `einops.repeat` uses a view in torch, so we generally clone the tensor to avoid using shared storage for each head entry
        ):
            repeated_tensor = einops.repeat(
                tensor,
                "batch pos d_model -> batch pos n_heads d_model",
                n_heads=self.cfg.n_heads,
            )
            if clone_tensor:
                return repeated_tensor.clone()
            else:
                return repeated_tensor

        if self.cfg.use_attn_in or self.cfg.use_split_qkv_input:
            # We're adding a head dimension
            attn_in = add_head_dimension(resid_pre, clone_tensor=False)
            if shortformer_pos_embed is not None:
                shortformer_pos_embed = add_head_dimension(shortformer_pos_embed)
        else:
            attn_in = resid_pre

        if self.cfg.use_attn_in:
            attn_in = self.hook_attn_in(attn_in.clone())

        if self.cfg.use_split_qkv_input:
            query_input = self.hook_q_input(attn_in.clone())
            key_input = self.hook_k_input(attn_in.clone())
            value_input = self.hook_v_input(attn_in.clone())
        else:
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in

        attn_out = self.hook_attn_out(
            # hook the residual stream states that are used to calculate the
            # queries, keys and values, independently.
            # Then take the layer norm of these inputs, and pass these to the attention module.
            self.attn(
                query_input=self.ln1(query_input)
                + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
                key_input=self.ln1(key_input)
                + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
                value_input=self.ln1(value_input),
                past_kv_cache_entry=past_kv_cache_entry,
                attention_mask=attention_mask,
            )
        )  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            resid_mid = self.hook_resid_mid(
                resid_pre + attn_out
            )  # [batch, pos, d_model]
            mlp_in = (
                resid_mid
                if not self.cfg.use_hook_mlp_in
                else self.hook_mlp_in(resid_mid.clone())
            )
            normalized_resid_mid = self.ln2(mlp_in)
            mlp_out = self.hook_mlp_out(
                self.mlp(normalized_resid_mid)
            )  # [batch, pos, d_model]
            resid_post = self.hook_resid_post(
                resid_mid + mlp_out
            )  # [batch, pos, d_model]
        elif self.cfg.parallel_attn_mlp:
            # Dumb thing done by GPT-J, both MLP and Attn read from resid_pre and write to resid_post, no resid_mid used.
            # In GPT-J, LN1 and LN2 are tied, in GPT-NeoX they aren't.
            normalized_resid_pre_2 = self.ln2(
                resid_pre
                if not self.cfg.use_hook_mlp_in
                else self.hook_mlp_in(resid_pre.clone())
            )
            mlp_out = self.hook_mlp_out(
                self.mlp(normalized_resid_pre_2)
            )  # [batch, pos, d_model]
            resid_post = self.hook_resid_post(
                resid_pre + attn_out + mlp_out
            )  # [batch, pos, d_model]
        else:
            resid_post = self.hook_resid_post(
                resid_pre + attn_out
            )  # [batch, pos, d_model]
        return resid_post


class BertBlock(nn.Module):
    """
    BERT Block. Similar to the TransformerBlock, except that the LayerNorms are applied after the attention and MLP, rather than before.
    """

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.attn = Attention(cfg)
        self.ln1 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
        self.ln2 = LayerNorm(cfg)

        self.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]
        self.hook_normalized_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 pos"]] = None,
    ):
        resid_pre = self.hook_resid_pre(resid_pre)

        query_input = resid_pre
        key_input = resid_pre
        value_input = resid_pre

        if self.cfg.use_split_qkv_input:

            def add_head_dimension(tensor):
                return einops.repeat(
                    tensor,
                    "batch pos d_model -> batch pos n_heads d_model",
                    n_heads=self.cfg.n_heads,
                ).clone()

            query_input = self.hook_q_input(add_head_dimension(query_input))
            key_input = self.hook_k_input(add_head_dimension(key_input))
            value_input = self.hook_v_input(add_head_dimension(value_input))

        attn_out = self.hook_attn_out(
            self.attn(
                query_input,
                key_input,
                value_input,
                additive_attention_mask=additive_attention_mask,
            )
        )
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)

        mlp_in = (
            resid_mid
            if not self.cfg.use_hook_mlp_in
            else self.hook_mlp_in(resid_mid.clone())
        )
        normalized_resid_mid = self.ln1(mlp_in)
        mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_mid))
        resid_post = self.hook_resid_post(normalized_resid_mid + mlp_out)
        normalized_resid_post = self.hook_normalized_resid_post(self.ln2(resid_post))

        return normalized_resid_post
