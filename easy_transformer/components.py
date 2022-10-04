from mimetypes import init
from typing import Callable, Union, List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import logging

from functools import *

from easy_transformer.hook_points import HookPoint
from easy_transformer.utils import (
    gelu_new,
    solu,
)
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig

from fancy_einsum import einsum

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
        return self.W_E[tokens, :] # Shape [batch pos d_model]


class Unembed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_vocab))
        self.b_U = nn.Parameter(torch.zeros(self.cfg.d_vocab))

    def forward(self, residual):
        return (
            einsum("batch pos d_model, d_model vocab -> batch pos vocab", 
                   residual, self.W_U) + self.b_U
        )  # [batch, pos, d_vocab]


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg.n_ctx, self.cfg.d_model))

    def forward(self, tokens):
        # Tokens have shape [batch, pos]
        # Output shape [pos, d_model] - will be broadcast along batch dim
        tokens_length = tokens.size(-1)
        return self.W_pos[:tokens_length, :]  # [pos, d_model]


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
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale = self.hook_scale(
            (
                x.pow(2).mean(-1, keepdim=True)
                + self.eps
            ).sqrt()
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
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale = self.hook_scale(
            (
                x.pow(2).mean(-1, keepdim=True)
                + self.eps
            ).sqrt()
        )  # [batch, pos, 1]
        x = self.hook_normalized(x / scale)  # [batch, pos, length]
        return x * self.w + self.b


# Attention
class Attention(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig], attn_type="global", layer_id=None):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

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
        # Create a query_pos x key_pos mask, with True iff that query position
        # can attend to that key position
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
            self.attn_scale *= (self.layer_id + 1)

        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, head_index, head_index, d_model]

        # This parameter is pointer to model.pos_embed.W_pos and is used only with shortformer style attention. It is initializaed to None and is loaded in separately with shortformer_load_pos_embeds.
        # See shortformer_load_pos_embeds for more details.
        if self.cfg.positional_embedding_type == "shortformer":
            self.shortformer_W_pos = None
            # This tracks the input to the keys and queries, which is resid_pre + pos_embeds
            self.hook_attn_input = HookPoint() # [batch, pos, d_model]
        

    def forward(self, x):
        if self.cfg.positional_embedding_type != "shortformer":
            # Normal attention
            q = self.hook_q(
                einsum("batch pos d_model, head_index d_model d_head \
                    -> batch pos head_index d_head", 
                            x, self.W_Q) + self.b_Q
            )  # [batch, pos, head_index, d_head]
            k = self.hook_k(
                einsum("batch pos d_model, head_index d_model d_head \
                    -> batch pos head_index d_head", 
                            x, self.W_K) + self.b_K
            )  # [batch, pos, head_index, d_head]
        else:
            # Weird shortformer attention
            q, k = self.shortformer_calculate_qk(x)
        v = self.hook_v(
            einsum("batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head", 
                        x, self.W_V) + self.b_V
        )  # [batch, pos, head_index, d_head]
        attn_scores = (
            einsum("batch query_pos head_index d_head, \
                batch key_pos head_index d_head \
                -> batch head_index query_pos key_pos", 
                   q, k) / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]
        if self.cfg.attention_dir == 'causal':
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.causal_mask(attn_scores) # [batch, head_index, query_pos, key_pos]
        attn_matrix = self.hook_attn(
            F.softmax(attn_scores, dim=-1)
        )  # [batch, head_index, query_pos, key_pos]
        z = self.hook_z(
            einsum("batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head", 
                v, attn_matrix)
        )  # [batch, pos, head_index, d_head]
        if not self.cfg.use_attn_result:
            out = (
                    einsum("batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos d_model", 
                        z, 
                        self.W_O)
                ) + self.b_O  # [batch, pos, d_model]
        else:
            # Explicitly calculate the attention result so it can be accessed by a hook
            # This is off by default because it can easily eat through your GPU memory.
            result = self.hook_result(
                einsum("batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos head_index d_model", 
                       z, 
                       self.W_O)
            )  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(
                    result, "batch position index model->batch position model", "sum"
                )
                + self.b_O
            )  # [batch, pos, d_model]
        return out

    def causal_mask(self, attn_scores):
        return torch.where(
            self.mask[: attn_scores.size(-2), : attn_scores.size(-1)],
            attn_scores,
            self.IGNORE,
        )
    
    def shortformer_load_pos_embeds(self, W_pos):
        """ 
        This is a very hacky way to implement positional embeddings for shortformer style models, which do not add positional embeddings into the residual stream but instead add it in to the queries and keys immediately before multiplying by W_Q and W_K, and NOT having it around for the values or MLPs. 

        This function adds in W_pos as a bonus parameter to the attention layer, with the same W_pos shared across all layers. This is pretty hacky, since it involves adding a parameter not included at initialization, and pollutes the state dict (the model's state dict now includes a bunch of copies of W_pos alas). I chose this implementation because it avoided changing the API for the layer (either init or forward), which felt messier.

        The original intention was to use this to do more efficient caching: caching is hard with absolute positional embeddings, since you can't translate the context window without recomputing the entire thing, but easier if the prior values and residual stream terms are the same. I've implemented it because it makes it easier for models to form induction heads. I'm not entirely sure why, though hypothesise that it's because there's two ways for induction heads to form with positional embeddings in the residual stream and only one with shortformer style positional embeddings.


        https://arxiv.org/abs/2012.15832
        """
        assert self.cfg.positional_embedding_type == 'shortformer', f"This function is only for shortformer style attention, while positional embedding type is {self.cfg.positional_embedding_type}"
        assert W_pos.shape == (self.cfg.n_ctx, self.cfg.d_model), f"Shortformer position embeddings must be of shape (n_ctx, d_model). They have shape: {W_pos.shape}"

        # As far as I can tell, self.W_pos = W_pos and self.W_pos = nn.Parameter(W_pos) are equivalent? The parameters are updated in lockstep, though in the latter each parameter has a separate gradient buffer, which are combined by the optimizer? PyTorch is weird man...
        self.shortformer_W_pos = W_pos
    
    def shortformer_calculate_qk(self, x):
        ctx_length = x.size(1)
        # We add on the positional encodings to the residual stream JUST for the keys and queries, it's not added to the normal residual stream.
        attn_input = self.hook_attn_input(
            x + 
            self.shortformer_W_pos[:ctx_length, :]
            ) # [batch, pos, d_model]
        q = self.hook_q(
            einsum("batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head", 
                        attn_input, self.W_Q) + self.b_Q
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(
            einsum("batch pos d_model, head_index d_model d_head \
                -> batch pos head_index d_head", 
                        attn_input, self.W_K) + self.b_K
        )  # [batch, pos, head_index, d_head]
        return (q, k)


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
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            self.hook_post_ln = HookPoint()  # [batch, pos, d_mlp]
            self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(self, x):
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", x, self.W_in) + self.b_in
        )  # [batch, pos, d_mlp]
        post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        if self.cfg.act_fn.endswith("_ln"):
            post_act = self.hook_post_ln(self.ln(post_act))
        mlp_out = (
            einsum("batch pos d_mlp, d_mlp d_model -> batch pos d_model", post_act, self.W_out) + self.b_out
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
            self.ln2 = LayerNorm(cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln1 = LayerNormPre(cfg)
            self.ln2 = LayerNormPre(cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
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
        self.mlp = MLP(cfg)

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, x):
        resid_pre = self.hook_resid_pre(x)  # [batch, pos, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.hook_attn_out(
            self.attn(normalized_resid_pre)
        )  # [batch, pos, d_model]
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)  # [batch, pos, d_model]
        
        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.hook_mlp_out(
            self.mlp(normalized_resid_mid)
        )  # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_mid + mlp_out)  # [batch, pos, d_model]
        return resid_post

class AttnOnlyBlock(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig], block_index):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln1 = LayerNormPre(cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
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

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, x):
        resid_pre = self.hook_resid_pre(x)  # [batch, pos, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.hook_attn_out(
            self.attn(normalized_resid_pre)
        )  # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_pre + attn_out)  # [batch, pos, d_model]
        return resid_post
