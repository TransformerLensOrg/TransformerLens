from mimetypes import init
from typing import Callable, Union, List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import warnings
import logging

from functools import *

from easy_transformer.hook_points import HookPoint
from easy_transformer.utils import gelu_new, solu, gelu_fast
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig

from fancy_einsum import einsum

from easy_transformer.caching import (
    EasyTransformerKeyValueCache,
    EasyTransformerKeyValueCacheEntry,
)

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg.d_vocab, self.cfg.d_model))

    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return self.W_E[tokens, :]  # Shape [batch pos d_model]


class Unembed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):

        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        # Note that there's a separate variable for d_vocab_out and d_vocab (the input vocab size). For language tasks these are always the same, but for algorithmic tasks we may want them to be different.
        self.W_U = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_vocab_out))
        self.b_U = nn.Parameter(torch.zeros(self.cfg.d_vocab_out))

    def forward(self, residual):
        return (
            einsum(
                "batch pos d_model, d_model vocab -> batch pos vocab",
                residual,
                self.W_U,
            )
            + self.b_U
        )  # [batch, pos, d_vocab]


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg.n_ctx, self.cfg.d_model))

    def forward(self, tokens: torch.Tensor, past_kv_pos_offset: int = 0):
        """Tokens have shape [batch, pos]
        past_kv_pos_offset is the length of tokens in the past_kv_cache (if used, defaults to zero if unused)
        Output shape [pos, d_model] - will be broadcast along batch dim"""

        tokens_length = tokens.size(-1)
        pos_embed = self.W_pos[
            past_kv_pos_offset : tokens_length + past_kv_pos_offset, :
        ]  # [pos, d_model]
        broadcast_pos_embed = einops.repeat(
            pos_embed, "... -> batch ...", batch=tokens.size(0)
        )  # [batch, pos, d_model]
        return broadcast_pos_embed


# LayerNormPre
# I fold the LayerNorm weights and biases into later weights and biases.
# This is just the 'center and normalise' part of LayerNorm
# Centering is equivalent to just deleting one direction of residual space,
# and is equivalent to centering the weight matrices of everything writing to the residual stream
# Normalising is a funkier non-linear operation, that projects the residual stream onto the unit hypersphere
class LayerNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        """LayerNormPre - the 'center and normalise' part of LayerNorm. Length is
        normally d_model, but is d_mlp for softmax. Not needed as a parameter. This
        should only be used in inference mode after folding in LayerNorm weights"""
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        # Hook Normalized captures LN output - here it's a vector with std 1 and mean 0
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        return self.hook_normalized(x / scale)  # [batch, pos, length]


class LayerNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, EasyTransformerConfig], length: Optional[int] = None
    ):

        """
        LayerNorm with optional length parameter

        length (Optional[int]): If the dimension of the LayerNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length))
        self.b = nn.Parameter(torch.zeros(self.length))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        # Hook_normalized is on the LN output
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        x = x / scale  # [batch, pos, length]
        return self.hook_normalized(x * self.w + self.b)


class RMSNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        """RMSNormPre - LayerNormPre without the centering and bias (RMS = Root Mean Square)"""
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x):
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        return self.hook_normalized(x / scale)  # [batch, pos, length]


class RMSNorm(nn.Module):
    def __init__(
        self, cfg: Union[Dict, EasyTransformerConfig], length: Optional[int] = None
    ):

        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = nn.Parameter(torch.ones(self.length))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x):
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        x = self.hook_normalized(x / scale)  # [batch, pos, length]
        return x * self.w


# Attention
class Attention(nn.Module):
    def __init__(
        self, cfg: Union[Dict, EasyTransformerConfig], attn_type="global", layer_id=None
    ):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [batch, head_index, query_pos, key_pos]

        Args:
            cfg (Union[Dict, EasyTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistal models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_Q = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)
        )
        self.W_K = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)
        )
        self.W_V = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)
        )
        self.W_O = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
        )
        self.b_Q = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head))
        self.b_K = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head))
        self.b_V = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head))
        self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model))

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

        self.register_buffer("IGNORE", torch.tensor(-1e5))

        self.layer_id = layer_id

        # attn_scale is a constant that we divide the attention scores by pre-softmax. I'm not entirely sure why it matters, but it's probably a mix of softmax not being scale invariant and numerical stability?
        if self.cfg.use_attn_scale:
            self.attn_scale = np.sqrt(self.cfg.d_head)
        else:
            self.attn_scale = 1.0
        if self.cfg.scale_attn_by_inverse_layer_idx:
            self.attn_scale *= self.layer_id + 1

        self.ln1 = LayerNormPre(cfg)  # moved here by Arthur

        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]

        # Added by Arthur; all are ResidPre, but we want finer access
        assert self.cfg.positional_embedding_type in [
            "standard",
            "rotary",
        ], f"q_input hooks only support standard and rotary positional embeddings, not {self.cfg.positional_embedding_type}"
        self.hook_q_input = HookPoint()  # [batch, pos, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, d_model]

        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, head_index, head_index, d_model]

        # See EasyTransformerConfig for more details.
        if self.cfg.positional_embedding_type == "shortformer":
            # This tracks the input to the keys and queries, which is resid_pre + pos_embeds
            self.hook_attn_input = HookPoint()  # [batch, pos, d_model]
        elif self.cfg.positional_embedding_type == "rotary":
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position. See EasyTransformerConfig for details
            self.hook_rot_k = HookPoint()
            self.hook_rot_q = HookPoint()
            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim, self.cfg.n_ctx
            )
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)

    def forward(
        self,
        resid_pre: torch.Tensor,  # goddamn normalized thing
        shortformer_pos_embed: Optional[torch.Tensor] = None,
        past_kv_cache_entry: Optional[EasyTransformerKeyValueCacheEntry] = None,
    ):
        """
        shortformer_pos_embed is only used if self.cfg.positional_embedding_type == "shortformer", else defaults to None and is irrelevant. See EasyTransformerConfig for more details
        past_kv_cache_entry is an optional entry of past keys and values for this layer, only relevant if generating text. Defaults to None

        """

        if self.cfg.use_headwise_qkv_input:
            assert self.cfg.positional_embedding_type in ["standard", "rotary"]
            warnings.warn("Using the new way of doing qkv input")
            head_input = einops.repeat(
                resid_pre, "a b c -> a b x c", x=self.cfg.n_heads
            )
            q = self.hook_q(
                einsum(
                    "batch pos head_index d_model, head_index d_model d_head \
                    -> batch pos head_index d_head",
                    self.ln1(self.hook_q_input(head_input.clone())),
                    self.W_Q,
                )
                + self.b_Q
            )  # [batch, pos, head_index, d_head]
            k = self.hook_k(
                einsum(
                    "batch pos head_index d_model, head_index d_model d_head \
                    -> batch pos head_index d_head",
                    self.ln1(self.hook_k_input(head_input.clone())),
                    self.W_K,
                )
                + self.b_K
            )  # [batch, pos, head_index, d_head]
            v = self.hook_v(
                einsum(
                    "batch pos head_index d_model, head_index d_model d_head \
                    -> batch pos head_index d_head",
                    self.ln1(self.hook_v_input(head_input.clone())),
                    self.W_V,
                )
                + self.b_V
            )  # [batch, pos, head_index, d_head]

        else:
            if self.cfg.positional_embedding_type in ["standard", "rotary"]:
                # Normal attention
                q = self.hook_q(
                    einsum(
                        "batch pos d_model, head_index d_model d_head \
                        -> batch pos head_index d_head",
                        self.ln1(resid_pre),
                        self.W_Q,
                    )
                    + self.b_Q
                )  # [batch, pos, head_index, d_head]
                k = self.hook_k(
                    einsum(
                        "batch pos d_model, head_index d_model d_head \
                        -> batch pos head_index d_head",
                        self.ln1(resid_pre),
                        self.W_K,
                    )
                    + self.b_K
                )  # [batch, pos, head_index, d_head]
            elif self.cfg.positional_embedding_type == "shortformer":
                # Weird shortformer attention see EasyTransformerConfig for details
                q, k = self.shortformer_calculate_qk(resid_pre, shortformer_pos_embed)
            v = self.hook_v(
                einsum(
                    "batch pos d_model, head_index d_model d_head \
                    -> batch pos head_index d_head",
                    self.ln1(resid_pre),
                    self.W_V,
                )
                + self.b_V
            )  # [batch, pos, head_index, d_head]

        # if past_kv_cache_entry is not None:
        assert past_kv_cache_entry is None, "past_kv_cache_entry is not None"
        # Appends the new keys and values to the cached values, and automatically updates the cache
        # kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
        # k, v = past_kv_cache_entry.append(k, v)
        # else:
        # Not using a cache
        kv_cache_pos_offset = 0

        if self.cfg.positional_embedding_type == "rotary":
            q, k = self.rotary_rotate_qk(q, k, kv_cache_pos_offset)

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
        if self.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset
            )  # [batch, head_index, query_pos, key_pos]
        attn_scores = self.hook_attn_scores(attn_scores)
        attn_matrix = self.hook_attn(
            F.softmax(attn_scores, dim=-1)
        )  # [batch, head_index, query_pos, key_pos]
        z = self.hook_z(
            einsum(
                "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
                v,
                attn_matrix,
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

    def apply_causal_mask(self, attn_scores, past_kv_pos_offset):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it's just a single token.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        assert (
            query_ctx_length + past_kv_pos_offset == key_ctx_length
        ), f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
        return torch.where(
            self.mask[
                past_kv_pos_offset : past_kv_pos_offset + query_ctx_length,
                :key_ctx_length,
            ],
            attn_scores,
            self.IGNORE,
        )

    def shortformer_calculate_qk(self, x, shortformer_pos_embed):
        # We add on the positional encodings to the residual stream JUST for the keys and queries, it's not added to the normal residual stream.
        attn_input = self.hook_attn_input(
            x + shortformer_pos_embed
        )  # [batch, pos, d_model]
        q = self.hook_q(
            einsum(
                "batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head",
                attn_input,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(
            einsum(
                "batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head",
                attn_input,
                self.W_K,
            )
            + self.b_K
        )  # [batch, pos, head_index, d_head]
        return (q, k)

    def rotary_rotate_qk(self, q, k, past_kv_pos_offset):
        # We first apply standard q and k calculation

        q = self.hook_rot_q(self.apply_rotary(q, past_kv_pos_offset))
        k = self.hook_rot_k(self.apply_rotary(k))
        return q, k

    def calculate_sin_cos_rotary(self, rotary_dim, n_ctx, base=10000):
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details
        """
        pos = torch.arange(n_ctx, dtype=torch.float32)
        dim = torch.arange(rotary_dim // 2, dtype=torch.float32)
        # A set of frequencies evenly spaced in log space
        freq = base ** (dim / (rotary_dim / 2))
        freq = einops.repeat(freq, "d -> (d 2)")
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles), torch.cos(angles)

    def rotate_every_two(self, x):
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]
        """
        rot_x = x.clone()
        rot_x[..., 0::2] = -x[..., 1::2]
        rot_x[..., 1::2] = x[..., 0::2]
        return rot_x

    def apply_rotary(self, x, past_kv_pos_offset=0):
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
        x_pos = x.size(1)
        x_rot = x[..., : self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)
        x_rotated = (
            x_rot
            * self.rotary_cos[past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
            + x_flip
            * self.rotary_sin[past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
        )
        return torch.cat([x_rotated, x_pass], dim=-1)


# MLP Layers
class MLP(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model))

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

    def forward(self, x):
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
        mlp_out = (
            einsum(
                "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
                post_act,
                self.W_out,
            )
            + self.b_out
        )  # [batch, pos, d_model]
        return mlp_out


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig], block_index):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNorm(cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            warnings.warn("Moved LN1 to the attention block")
            if not self.cfg.attn_only:
                self.ln2 = LayerNormPre(cfg)
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
            self.mlp = MLP(cfg)

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: torch.Tensor,
        shortformer_pos_embed: Optional[torch.Tensor] = None,
        past_kv_cache_entry: Optional[EasyTransformerKeyValueCacheEntry] = None,
    ):
        """A single Transformer block.

        Args:
            resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
            cache (EasyTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
            shortformer_pos_embed (torch.Tensor, optional): Only used for positional_embeddings_type == "shortformer". The positional embeddings. See EasyTransformerConfig for details. Defaults to None.

        Returns:
            _type_: _description_
        """
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]
        # normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.hook_attn_out(
            self.attn(
                resid_pre,  # edited by Arthur from normalized ... so we can go headwise
                shortformer_pos_embed=shortformer_pos_embed,
                past_kv_cache_entry=past_kv_cache_entry,
            )
        )  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            resid_mid = self.hook_resid_mid(
                resid_pre + attn_out
            )  # [batch, pos, d_model]
            normalized_resid_mid = self.ln2(resid_mid)
            mlp_out = self.hook_mlp_out(
                self.mlp(normalized_resid_mid)
            )  # [batch, pos, d_model]
            resid_post = self.hook_resid_post(
                resid_mid + mlp_out
            )  # [batch, pos, d_model]
        elif self.cfg.parallel_attn_mlp:
            # Dumb thing done by GPT-J, both MLP and Attn read from resid_pre and write to resid_post, no resid_mid used.
            # In GPT-J, LN1 and LN2 are tied, in GPT-NeoX they aren't.
            normalized_resid_pre_2 = self.ln2(resid_pre)
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
