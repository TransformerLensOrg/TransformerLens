from mimetypes import init
from typing import Callable, Union, List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import logging

from tqdm import tqdm
import random
import time

from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc
import collections
import copy

# import comet_ml
import itertools

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    print_gpu_mem,
    get_sample_from_dataset,
    solu,
    reglu,
    geglu,
    swiglu,
)
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig

from easy_transformer.EasyTransformerKeyValueCache import (
    EasyTransformerKeyValueCache,
    EasyTransformerKeyValueCacheEntry,
)

VALID_MODEL_NAMES = set(
    [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "facebook/opt-125m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        "facebook/opt-30b",
        "facebook/opt-66b",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "stanford-gpt2-small-A",
        "stanford-gpt2-small-B",
        "stanford-gpt2-small-C",
        "stanford-gpt2-small-D",
        "stanford-gpt2-small-E",
        "stanford-gpt2-medium-A",
        "stanford-gpt2-medium-B",
        "stanford-gpt2-medium-C",
        "stanford-gpt2-medium-D",
        "stanford-gpt2-medium-E",
    ]
)

MODEL_NAMES_DICT = {
    "stanford-gpt2-small-A": "stanford-crfm/alias-gpt2-small-x21",
    "stanford-gpt2-small-B": "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-gpt2-small-C": "stanford-crfm/caprica-gpt2-small-x81",
    "stanford-gpt2-small-D": "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-gpt2-small-E": "stanford-crfm/expanse-gpt2-small-x777",
    "stanford-gpt2-medium-A": "stanford-crfm/arwen-gpt2-medium-x21",
    "stanford-gpt2-medium-B": "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-gpt2-medium-C": "stanford-crfm/celebrimbor-gpt2-medium-x81",
    "stanford-gpt2-medium-D": "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-gpt2-medium-E": "stanford-crfm/eowyn-gpt2-medium-x777",
}
# The steps for which there are checkpoints in the stanford crfm models - provided as reference
STANFORD_CRFM_CHECKPOINTS = (
    list(range(0, 100, 10))
    + list(range(100, 2000, 50))
    + list(range(2000, 20000, 100))
    + list(range(20000, 400000 + 1, 1000))
)

# TODO: Add Bloom, GPT-J and GPT-NeoX
"""
EleutherAI/gpt-j-6B
EleutherAI/gpt-neox-20b
bloom-350m
bloom-760m
bloom-1b3
bloom-2b5
bloom-6b3
bloom (176B parameters)
https://huggingface.co/docs/transformers/model_doc/bloom
"""

# Define network architecture

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_vocab))

    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return einops.rearrange(
            self.W_E[:, tokens], "d_model batch pos -> batch pos d_model"
        )


class Unembed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(self.cfg.d_vocab, self.cfg.d_model))
        self.b_U = nn.Parameter(torch.empty(self.cfg.d_vocab))

    def forward(self, tokens):
        return (
            torch.einsum("vm,bpm->bpv", self.W_U, tokens) + self.b_U
        )  # [batch, pos, d_vocab]


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.n_ctx))

    def forward(self, x, offset=0):
        # Output shape [pos, d_model] - will be broadcast along batch dim
        # Offset accounts for generation
        return self.W_pos[:, offset : offset + x.size(-1)].T  # [pos, d_model]


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
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
        scale = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )  # [batch, pos, 1]
        x = self.hook_normalized(x / scale)  # [batch, pos, length]
        return x * self.w + self.b


# Attention
class Attention(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig], attn_type="global"):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_Q = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
        )
        self.W_K = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
        )
        self.W_V = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model)
        )
        self.W_O = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head)
        )
        self.b_Q = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_head))
        self.b_K = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_head))
        self.b_V = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_head))
        self.b_O = nn.Parameter(torch.empty(self.cfg.d_model))

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

        if self.cfg.use_attn_scale:
            self.attn_scale = np.sqrt(self.cfg.d_head)
        else:
            self.attn_scale = 1.0

        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, head_index, head_index, d_model]

    def forward(
        self, x, cache_entry: Optional[EasyTransformerKeyValueCacheEntry] = None
    ):
        B, S, D = x.shape
        q = self.hook_q(
            torch.einsum("ihm,bpm->bpih", self.W_Q, x) + self.b_Q
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(
            torch.einsum("ihm,bpm->bpih", self.W_K, x) + self.b_K
        )  # [batch, pos, head_index, d_head]
        v = self.hook_v(
            torch.einsum("ihm,bpm->bpih", self.W_V, x) + self.b_V
        )  # [batch, pos, head_index, d_head]

        if cache_entry is not None:
            CB, CS, CN, CH = cache_entry.past_keys.shape
            assert CB == B
            assert CN == self.cfg.n_heads
            assert CH == self.cfg.d_head
            if CS != 0:
                assert S == 1, "Can only pass one token at a time after cache is loaded"
            k, v = cache_entry.append(k, v)
            pos_start = CS
        else:
            pos_start = 0

        attn_scores = (
            torch.einsum("bpih,bqih->bipq", q, k) / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]
        attn_scores = self.hook_attn_scores(
            self.causal_mask(attn_scores, pos_start)
        )  # [batch, head_index, query_pos, key_pos]
        attn_matrix = self.hook_attn(
            F.softmax(attn_scores, dim=-1)
        )  # [batch, head_index, query_pos, key_pos]
        z = self.hook_z(
            torch.einsum("bpih,biqp->bqih", v, attn_matrix)
        )  # [batch, pos, head_index, d_head]
        if self.cfg.use_attn_result:
            result = self.hook_result(
                torch.einsum("imh,bqih->bqim", self.W_O, z)
            )  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(
                    result, "batch position index model->batch position model", "sum"
                )
                + self.b_O
            )  # [batch, pos, d_model]
        else:
            out = (
                torch.einsum("idh,bqih->bqd", self.W_O, z) + self.b_O
            )  # [batch, pos, d_model]
        return out

    def causal_mask(self, attn_scores, cache_pos_start):
        return torch.where(
            self.mask[
                cache_pos_start : cache_pos_start + attn_scores.size(-2),
                : attn_scores.size(-1),
            ],
            attn_scores,
            self.IGNORE,
        )


# MLP Layers
class MLP(nn.Module):
    def __init__(self, cfg: Union[Dict, EasyTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))
        self.b_in = nn.Parameter(torch.empty(self.cfg.d_mlp))
        if self.cfg.gated_act_fn:
            self.W_gate = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model))
            self.b_gate = nn.Parameter(torch.empty(self.cfg.d_mlp))
            self.hook_gate = HookPoint()
        self.W_out = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp))
        self.b_out = nn.Parameter(torch.empty(self.cfg.d_model))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "silu":
            self.act_fn = F.silu
        elif self.cfg.act_fn == "glu":
            self.act_fn = F.glu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            self.hook_post_ln = HookPoint()  # [batch, pos, d_mlp]
            self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
        elif self.cfg.act_fn == "reglu":
            self.act_fn = reglu
        elif self.cfg.act_fn == "geglu":
            self.act_fn = geglu
        elif self.cfg.act_fn == "swiglu":
            self.act_fn = swiglu
        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(self, x):
        pre_act = self.hook_pre(
            torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        )  # [batch, pos, d_mlp]
        if self.cfg.gated_act_fn:
            gate = self.hook_gate(
                torch.einsum("md,bpd->bpm", self.W_gate, x) + self.b_gate
            )
            post_act = self.hook_post(self.act_fn(pre_act, gate))  # [batch, pos, d_mlp]
        else:
            post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        if self.cfg.act_fn == "solu_ln":
            post_act = self.hook_post_ln(self.ln(post_act))
        mlp_out = (
            torch.einsum("dm,bpm->bpd", self.W_out, post_act) + self.b_out
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
            # If it's None, don't create either layer
            pass
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )

        if not self.cfg.use_local_attn:
            self.attn = Attention(cfg, "global")
        else:
            assert self.cfg.attn_types is not None
            attn_type = self.cfg.attn_types[block_index]
            self.attn = Attention(cfg, attn_type)
        self.mlp = MLP(cfg)

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self, x, cache_entry: Optional[EasyTransformerKeyValueCacheEntry] = None
    ):
        resid_pre = self.hook_resid_pre(x)  # [batch, pos, d_model]
        if self.cfg.normalization_type is not None:
            normalized_resid_pre = self.ln1(resid_pre)
        else:
            normalized_resid_pre = resid_pre
        attn_out = self.hook_attn_out(
            self.attn(normalized_resid_pre, cache_entry)
        )  # [batch, pos, d_model]
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)  # [batch, pos, d_model]

        if self.cfg.normalization_type is not None:
            normalized_resid_mid = self.ln2(resid_mid)
        else:
            normalized_resid_mid = resid_mid
        mlp_out = self.hook_mlp_out(
            self.mlp(normalized_resid_mid)
        )  # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_mid + mlp_out)  # [batch, pos, d_model]
        return resid_post


# Full transformer
class EasyTransformer(HookedRootModule):
    """
    This class implements a full Transformer using the above components, with
    HookPoints on every interesting activation. It inherits from HookedRootModule.

    It can be initialised with a model name, and then will automatically load the model weights
    for that model, loads them into this model, as well as fold in LayerNorm and center
    the weights.

    It can also be initilised with an EasyTransformerConfig or a config dictionary, which can be used to instantiate a custom model without loading pretrained weights and will instead use Pytorch's default weight initialisation.
    """

    def __init__(
        self,
        model_name,
        cfg=None,
        tokenizer=None,
        use_attn_result=False,
        model=None,
        keep_original_model=False,
        center_weights=True,
        checkpoint=None,
    ):
        """
        model_name (str: The name of the model to load, via HuggingFace. If
            "custom", then cfg must be provided.
        cfg (EasyTransformerConfig, *optional*): The config to use for the
            model. If not provided, a model name must be passed via model_name.
        tokenizer (*optional): The tokenizer to use for the model. If not
            provided, initialized to None, though the user must initialize one
            before passing strings to the model.
        use_attn_result (bool): Says whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        model: The model loaded from HuggingFace or separately initialized. If
            None, it is automatically loaded from HuggingFace if model_name is
            passed - this just saves memory if the model was already loaded into
            RAM.
        keep_original_model (bool): If False, the original model is deleted,
            otherwise it's kept as a self.model attribute
        center_weights (bool): If True, the weights are centered
        checkpoint (int, *optional): The checkpoint number of the model to load
            if it is a model with multiple possible checkpoints to load from.
        """
        super().__init__()
        if model_name == "custom":
            assert cfg is not None, "Must provide a config for custom model"
            self.cfg = cfg
            self.model_name = cfg.model_name
            self.model_type = cfg.model_type
            if self.cfg.tokenizer_name is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens directly. In this case, we don't need a tokenizer.
                self.tokenizer = None
            self.use_attn_result = use_attn_result
            self.model = None
            self.keep_original_model = False
            # We're initializing a model, no need to load weights from a checkpoint
            self.checkpoint = None
        else:
            assert (
                model_name in VALID_MODEL_NAMES
            ), f"Invalid model name: {model_name}. Valid model names are: {VALID_MODEL_NAMES}"
            self.model_name = model_name
            if self.model_name in MODEL_NAMES_DICT:
                self.full_model_name = MODEL_NAMES_DICT[self.model_name]
            else:
                self.full_model_name = self.model_name
            self.model_type = self.get_model_type(self.full_model_name)
            if model is not None:
                self.model = model
            else:
                if checkpoint is not None:
                    if "stanford" not in self.model_name:
                        logging.warning(
                            f"Loading checkpoints is not supported for the model {self.model_name}. Loading without checkpoints"
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.full_model_name
                        )
                    else:
                        assert (
                            checkpoint in STANFORD_CRFM_CHECKPOINTS
                        ), f"Checkpoint {checkpoint} is not valid. Available checkpoints are {STANFORD_CRFM_CHECKPOINTS}"
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.full_model_name, revision=f"checkpoint-{checkpoint}"
                        )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.full_model_name
                    )

            self.cfg = self.convert_hf_config(
                self.model.config, model_type=self.model_type
            )
            self.cfg.use_attn_result = use_attn_result
            self.cfg.checkpoint = checkpoint
            self.cfg.model_type = self.model_type
            self.cfg.model_name = self.model_name
            self.cfg.tokenizer_name = self.full_model_name
            self.cfg.normalization_type = "LNPre"
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.cfg.d_vocab:
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1

        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.cfg, block_index)
                for block_index in range(self.cfg.n_layers)
            ]
        )
        if self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )
        self.unembed = Unembed(self.cfg)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

        # Load model weights, and fold in layer norm weights
        if self.model_type == "gpt2":
            self.load_gpt2_weights(self.model)
        elif self.model_type == "neo":
            self.load_neo_weights(self.model)
        elif self.model_type == "gptj":
            self.load_gptj_weights(self.model)
        elif self.model_type == "neox":
            self.load_neox_weights(self.model)
        elif self.model_type == "opt":
            self.load_opt_weights(self.model)
        elif self.model_type == "custom":
            self.init_weights()

        # Set the average of each weight matrix writing to the residual stream to zero
        # (Layer Norm removes the mean anyway, so this simplifies the weights
        # without changing the computation)
        if center_weights:
            self.center_weights()

        if not keep_original_model and self.model is not None:
            # Delete the original model to save memory
            del self.model

    def forward(
        self,
        input,
        return_type: Optional[str] = "logits",
        cache: Optional[EasyTransformerKeyValueCache] = None,
    ):
        """Input is either a batch of tokens ([batch, pos]) or a text string.

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), 'logits' (return logits), 'loss' (return cross-entropy loss), 'both' (return logits and loss)
        """
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input)
        else:
            tokens = input
        assert isinstance(tokens, torch.Tensor)
        B, S = tokens.shape
        if cache is None:
            pos_start = 0
        else:
            CB, CS, CN, CH = cache[0].past_keys.shape
            assert CB == B
            assert CN == self.cfg.n_heads
            assert CH == self.cfg.d_head
            assert CS == 0 or S == 1, "Pass in one token at a time after loading cache"
            pos_start = CS
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(
            self.pos_embed(tokens, pos_start)
        )  # [batch, pos, d_model]
        residual = embed + pos_embed  # [batch, pos, d_model]
        for i, block in enumerate(self.blocks):
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(residual)  # [batch, pos, d_model]
        if return_type is None:
            return None
        else:
            if self.cfg.normalization_type is not None:
                residual = self.ln_final(residual)  # [batch, pos, d_vocab]
            logits = self.unembed(residual)  # [batch, pos, d_vocab]
            if return_type == "logits":
                return logits
            else:
                loss = self.cross_entropy_loss(logits, tokens)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return {"logits": logits, "loss": loss}
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None

    def set_tokenizer(self, tokenizer):
        """
        Sets the tokenizer to use for this model.
        tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer
        """
        assert isinstance(tokenizer, PreTrainedTokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate(
        self,
        input: Union[str, list, torch.Tensor],
        max_new_tokens: int,
        stop_at_eos: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        use_cache: bool = True,
    ):
        """
        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            input (int): Either a batch of tokens ([batch, pos]) or a text string
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            pad_token_id (int, *optional*): The token ID to use for padding. If None, use the tokenizer's pad_token_id - required if using stop_at_eos
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use beam or greedy search.
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens
            temperature (float): Temperature for sampling. Higher values will make the model more random
            freq_penalty (float): Frequency penalty for sampling. Higher values will make the model more random
            num_beams (int): Number of beams to use for beam search. If 1, use greedy search
            num_return_sequences (int): Number of sequences to return for beam search
            use_cache (bool): If True, create and use cache to speed up generation
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
        """
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input)
        else:
            tokens = input
        assert isinstance(tokens, torch.Tensor)
        B, S = tokens.shape
        if use_cache:
            cache = EasyTransformerKeyValueCache.init_cache(self.cfg, B)
        else:
            cache = None
        if stop_at_eos and pad_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.pad_token_id is not None
            ), "Must pass a pad_token_id if stop_at_eos is True and tokenizer is None or has no pad_token_id"
            pad_token_id = self.tokenizer.pad_token_id
        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id
        self.eval()
        if not do_sample and num_beams == 1:
            return self.greedy_search(
                tokens, max_new_tokens, stop_at_eos, pad_token_id, eos_token_id, cache
            )
        elif not do_sample and num_beams > 1:
            raise NotImplementedError("Beam search not implemented yet")
            return self.beam_search(
                tokens, max_new_tokens, num_beams, num_return_sequences, cache
            )
        elif do_sample and num_beams > 1:
            raise NotImplementedError("Beam sampling not implemented yet")
            return self.beam_sample(
                tokens,
                max_new_tokens,
                num_beams,
                num_return_sequences,
                top_k,
                top_p,
                temperature,
                freq_penalty,
                cache,
            )
        else:
            return self.sample(
                tokens,
                max_new_tokens,
                stop_at_eos,
                pad_token_id,
                eos_token_id,
                top_k,
                top_p,
                temperature,
                freq_penalty,
                cache,
            )

    def greedy_search(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        stop_at_eos: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        cache: Optional[EasyTransformerKeyValueCache] = None,
    ):
        """
        Greedily sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            pad_token_id (int, *optional*): The token ID to use for padding. If None, use the tokenizer's pad_token_id - required if using stop_at_eos
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
        """
        B, S = tokens.shape
        outputs = tokens
        unfinished_sequences = tokens.new(tokens.shape[0]).fill_(1)

        if stop_at_eos and pad_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.pad_token_id is not None
            ), "Must pass a pad_token_id if stop_at_eos is True and tokenizer is None or has no pad_token_id"
            pad_token_id = self.tokenizer.pad_token_id
        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id

        for _ in tqdm(range(max_new_tokens)):
            logits = self(tokens, cache)
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            if stop_at_eos:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
                unfinished_sequences.mul_((next_tokens != eos_token_id).long())
            outputs = torch.cat([outputs, next_tokens.unsqueeze(-1)], dim=-1)
            if cache is not None:
                tokens = next_tokens.unsqueeze(-1)
            else:
                tokens = outputs

        return outputs

    def beam_search(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences: int,
        cache: Optional[EasyTransformerKeyValueCache] = None,
    ):
        """
        Beam search for tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            num_beams (int): Number of beams to use for beam search.
            num_return_sequences (int): Number of sequences to return for beam search
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
        Returns:
            outputs (torch.Tensor): [batch * num_return_sequences, pos + max_new_tokens], generated sequence of new tokens
        """
        assert num_return_sequences <= num_beams
        raise NotImplementedError("Beam search not implemented yet")

    def beam_sample(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences: int,
        top_k: int,
        top_p: float,
        temperature: float,
        freq_penalty: float,
        cache: Optional[EasyTransformerKeyValueCache] = None,
    ):
        """
        Beam sampling for tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            num_beams (int): Number of beams to use for beam search.
            num_return_sequences (int): Number of sequences to return for beam search
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens
            temperature (float): Temperature for sampling. Higher values will make the model more random
            freq_penalty (float): Frequency penalty for sampling. Higher values will make the model more random
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
        Returns:
            outputs (torch.Tensor): [batch * num_return_sequences, pos + max_new_tokens], generated sequence of new tokens
        """
        assert num_return_sequences <= num_beams
        raise NotImplementedError("Beam sampling not implemented yet")

    def sample(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        stop_at_eos: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        cache: Optional[EasyTransformerKeyValueCache] = None,
    ):
        """
        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            pad_token_id (int, *optional*): The token ID to use for padding. If None, use the tokenizer's pad_token_id - required if using stop_at_eos
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens
            temperature (float): Temperature for sampling. Higher values will make the model more random
            freq_penalty (float): Frequency penalty for sampling. Higher values will make the model more random
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
        """
        B, S = tokens.shape
        outputs = tokens
        unfinished_sequences = tokens.new(tokens.shape[0]).fill_(1)

        if stop_at_eos and pad_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.pad_token_id is not None
            ), "Must pass a pad_token_id if stop_at_eos is True and tokenizer is None or has no pad_token_id"
            pad_token_id = self.tokenizer.pad_token_id
        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id

        def sample_logits(input_ids, logits, top_k, top_p, temperature, freq_penalty):
            assert temperature > 0, "temperature has to be greater than 0"
            logits = logits / temperature
            if freq_penalty > 0:
                for b in range(logits.shape[0]):
                    logits[b] = logits[b] - freq_penalty * torch.bincount(
                        input_ids[b], minlength=logits.shape[-1]
                    )
            if top_k is not None:
                assert top_k > 0, "top_k has to be greater than 0"
                top_logits, top_idx = logits.topk(top_k)
                indices_to_remove = logits < top_logits[..., -1].unsqueeze(-1)
                logits = logits.masked_fill(indices_to_remove, -float("inf"))
            if top_p < 1.0:
                assert top_p > 0.0, "top_p has to be greater than 0"
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, -float("inf"))
            return torch.distributions.categorical.Categorical(logits=logits).sample()

        for _ in tqdm(range(max_new_tokens)):
            logits = self(tokens, cache)
            next_tokens = sample_logits(
                outputs,
                logits[:, -1, :],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
            )
            if stop_at_eos:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
                unfinished_sequences.mul_((next_tokens != eos_token_id).long())
            outputs = torch.cat([outputs, next_tokens.unsqueeze(-1)], dim=-1)
            if cache is not None:
                tokens = next_tokens.unsqueeze(-1)
            else:
                tokens = outputs

        return outputs

    def to_tokens(self, text):
        assert self.tokenizer is not None
        return self.tokenizer(text, return_tensors="pt", padding=True)["input_ids"]

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls(model_name, **kwargs)

    @classmethod
    def from_config(cls, cfg):
        """Used to generate a model from a config object to train from

        Args:
            cfg (EasyTransformerConfig): Config for the model

        Returns:
            EasyTransformer: An initialised EasyTransformer model
        """
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig(**cfg)
        return cls(
            "custom",
            cfg,
            use_attn_result=cfg.use_attn_result,
        )

    def get_model_type(self, model_name):
        if "gpt2" in model_name or "stanford" in model_name:
            return "gpt2"
        elif "opt" in model_name:
            return "opt"
        elif model_name == "EleutherAI/gpt-neox-20b":
            return "neox"
        elif model_name == "EleutherAI/gpt-j-6B":
            return "gptj"
        elif "neo" in model_name:
            return "neo"
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def convert_hf_config(self, hf_config, model_type):
        cfg_dict = {}
        if model_type == "neo":
            cfg_dict = {
                "d_model": hf_config.hidden_size,
                "d_head": hf_config.hidden_size // hf_config.num_heads,
                "n_heads": hf_config.num_heads,
                "d_mlp": hf_config.hidden_size * 4,
                "n_layers": hf_config.num_layers,
                "n_ctx": hf_config.max_position_embeddings,
                "eps": hf_config.layer_norm_epsilon,
                "d_vocab": hf_config.vocab_size,
                "attn_types": hf_config.attention_layers,
                "act_fn": hf_config.activation_function,
                "use_attn_scale": False,
                "use_local_attn": True,
                "window_size": hf_config.window_size,
            }
        elif model_type == "gpt2":
            cfg_dict = {
                "d_model": hf_config.n_embd,
                "d_head": hf_config.n_embd // hf_config.n_head,
                "n_heads": hf_config.n_head,
                "d_mlp": hf_config.n_embd * 4,
                "n_layers": hf_config.n_layer,
                "n_ctx": hf_config.n_ctx,
                "eps": hf_config.layer_norm_epsilon,
                "d_vocab": hf_config.vocab_size,
                "act_fn": hf_config.activation_function,
                "use_attn_scale": True,
                "use_local_attn": False,
            }
        elif model_type == "opt":
            cfg_dict = {
                "d_model": hf_config.hidden_size,
                "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
                "n_heads": hf_config.num_attention_heads,
                "d_mlp": hf_config.ffn_dim,
                "n_layers": hf_config.num_hidden_layers,
                "n_ctx": hf_config.max_position_embeddings,
                "eps": 1e-5,
                "d_vocab": hf_config.vocab_size,
                "act_fn": hf_config.activation_function,
                "use_attn_scale": True,
                "use_local_attn": False,
            }
        elif model_type == "gptj":
            raise NotImplementedError
        elif model_type == "neox":
            raise NotImplementedError
        else:
            raise NotImplementedError
        cfg_dict["model_name"] = self.model_name
        cfg_dict["model_type"] = model_type
        cfg = EasyTransformerConfig.from_dict(cfg_dict)
        return cfg

    def center_weights(self):
        # Sets the average of each row of each weight matrix writing to the
        # residual stream to zero
        # LayerNorm subtracts the mean of the residual stream, and it's always
        # applied when reading from the residual stream, so this dimension is
        # purely noise
        # Also does the same for W_U, since translating the logits doesn't affect
        # the log_probs or loss
        self.embed.W_E.data -= self.embed.W_E.mean(0, keepdim=True)
        self.pos_embed.W_pos.data -= self.pos_embed.W_pos.mean(0, keepdim=True)
        self.unembed.W_U.data -= self.unembed.W_U.mean(0, keepdim=True)
        for block in self.blocks:
            block.attn.W_O.data -= einops.reduce(
                block.attn.W_O, "index d_model d_head -> index 1 d_head", "mean"
            )
            block.mlp.W_out.data -= block.mlp.W_out.mean(0, keepdim=True)

    def load_gpt2_weights(self, gpt2):
        sd = self.state_dict()

        sd["embed.W_E"] = gpt2.transformer.wte.weight.T
        sd["pos_embed.W_pos"] = gpt2.transformer.wpe.weight.T

        for l in range(self.cfg.n_layers):
            # In GPT-2, q,k,v are produced by one big linear map, whose output is
            # concat([q, k, v])
            W = gpt2.transformer.h[l].attn.c_attn.weight
            w_ln_attn = gpt2.transformer.h[l].ln_1.weight
            W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
            W_Q = einops.rearrange(W_Q, "m (i h)->i h m", i=self.cfg.n_heads)
            W_K = einops.rearrange(W_K, "m (i h)->i h m", i=self.cfg.n_heads)
            W_V = einops.rearrange(W_V, "m (i h)->i h m", i=self.cfg.n_heads)

            # Fold in layer norm weights
            sd[f"blocks.{l}.attn.W_Q"] = W_Q * w_ln_attn
            sd[f"blocks.{l}.attn.W_K"] = W_K * w_ln_attn
            sd[f"blocks.{l}.attn.W_V"] = W_V * w_ln_attn

            b_ln = gpt2.transformer.h[l].ln_1.bias
            qkv_bias = gpt2.transformer.h[l].attn.c_attn.bias
            qkv_bias = einops.rearrange(
                qkv_bias,
                "(qkv index head)->qkv index head",
                qkv=3,
                index=self.cfg.n_heads,
                head=self.cfg.d_head,
            )
            # Fold in layer norm biases
            sd[f"blocks.{l}.attn.b_Q"] = W_Q @ b_ln + qkv_bias[0]
            sd[f"blocks.{l}.attn.b_K"] = W_K @ b_ln + qkv_bias[1]
            sd[f"blocks.{l}.attn.b_V"] = W_V @ b_ln + qkv_bias[2]

            W_O = gpt2.transformer.h[l].attn.c_proj.weight
            W_O = einops.rearrange(W_O, "(i h) m->i m h", i=self.cfg.n_heads)
            sd[f"blocks.{l}.attn.W_O"] = W_O
            sd[f"blocks.{l}.attn.b_O"] = gpt2.transformer.h[l].attn.c_proj.bias

            W_in = gpt2.transformer.h[l].mlp.c_fc.weight.T
            W_out = gpt2.transformer.h[l].mlp.c_proj.weight.T
            # Fold in layer norm weights
            W_in_adj = gpt2.transformer.h[l].ln_2.weight[None, :] * W_in
            sd[f"blocks.{l}.mlp.W_in"] = W_in_adj
            # Fold in layer norm biases
            sd[f"blocks.{l}.mlp.b_in"] = gpt2.transformer.h[l].mlp.c_fc.bias + (
                W_in @ gpt2.transformer.h[l].ln_2.bias
            )
            sd[f"blocks.{l}.mlp.W_out"] = W_out
            sd[f"blocks.{l}.mlp.b_out"] = gpt2.transformer.h[l].mlp.c_proj.bias
        W_U = gpt2.lm_head.weight
        # Fold in layer norm weights
        sd["unembed.W_U"] = gpt2.transformer.ln_f.weight[None, :] * W_U
        # Fold in layer norm biases
        sd["unembed.b_U"] = gpt2.lm_head.weight @ gpt2.transformer.ln_f.bias
        self.load_state_dict(sd)

    def load_neo_weights(self, neo):
        sd = self.state_dict()

        sd["embed.W_E"] = neo.transformer.wte.weight.T
        sd["pos_embed.W_pos"] = neo.transformer.wpe.weight.T

        for l in range(self.cfg.n_layers):
            w_ln_attn = neo.transformer.h[l].ln_1.weight
            W_Q = neo.transformer.h[l].attn.attention.q_proj.weight
            W_K = neo.transformer.h[l].attn.attention.k_proj.weight
            W_V = neo.transformer.h[l].attn.attention.v_proj.weight
            W_Q = einops.rearrange(W_Q, "(i h) m->i h m", i=self.cfg.n_heads)
            W_K = einops.rearrange(W_K, "(i h) m->i h m", i=self.cfg.n_heads)
            W_V = einops.rearrange(W_V, "(i h) m->i h m", i=self.cfg.n_heads)

            sd[f"blocks.{l}.attn.W_Q"] = W_Q * w_ln_attn
            sd[f"blocks.{l}.attn.W_K"] = W_K * w_ln_attn
            sd[f"blocks.{l}.attn.W_V"] = W_V * w_ln_attn

            b_ln = neo.transformer.h[l].ln_1.bias
            sd[f"blocks.{l}.attn.b_Q"] = W_Q @ b_ln
            sd[f"blocks.{l}.attn.b_K"] = W_K @ b_ln
            sd[f"blocks.{l}.attn.b_V"] = W_V @ b_ln

            W_O = neo.transformer.h[l].attn.attention.out_proj.weight
            W_O = einops.rearrange(W_O, "m (i h)->i m h", i=self.cfg.n_heads)
            sd[f"blocks.{l}.attn.W_O"] = W_O
            sd[f"blocks.{l}.attn.b_O"] = neo.transformer.h[
                l
            ].attn.attention.out_proj.bias

            W_in = neo.transformer.h[l].mlp.c_fc.weight
            W_out = neo.transformer.h[l].mlp.c_proj.weight
            W_in_adj = neo.transformer.h[l].ln_2.weight[None, :] * W_in
            sd[f"blocks.{l}.mlp.W_in"] = W_in_adj
            sd[f"blocks.{l}.mlp.b_in"] = neo.transformer.h[l].mlp.c_fc.bias + (
                W_in @ neo.transformer.h[l].ln_2.bias
            )
            sd[f"blocks.{l}.mlp.W_out"] = W_out
            sd[f"blocks.{l}.mlp.b_out"] = neo.transformer.h[l].mlp.c_proj.bias
        W_U = neo.lm_head.weight
        sd["unembed.W_U"] = neo.transformer.ln_f.weight[None, :] * W_U
        sd["unembed.b_U"] = neo.lm_head.weight @ neo.transformer.ln_f.bias
        self.load_state_dict(sd)

    def load_neox_weights(self, neox):
        raise NotImplementedError

    def load_gptj_weights(self, gptj):
        raise NotImplementedError

    def load_opt_weights(self, opt):
        sd = self.state_dict()

        sd["embed.W_E"] = opt.model.decoder.embed_tokens.weight.T
        sd["pos_embed.W_pos"] = opt.model.decoder.embed_positions.weight.T[:, 2:]

        for l in range(self.cfg.n_layers):
            w_ln_attn = opt.model.decoder.layers[l].self_attn_layer_norm.weight
            W_Q = opt.model.decoder.layers[l].self_attn.q_proj.weight
            W_K = opt.model.decoder.layers[l].self_attn.k_proj.weight
            W_V = opt.model.decoder.layers[l].self_attn.v_proj.weight
            W_Q = einops.rearrange(
                W_Q,
                "(index d_head) d_model->index d_head d_model",
                index=self.cfg.n_heads,
            )
            W_K = einops.rearrange(
                W_K,
                "(index d_head) d_model->index d_head d_model",
                index=self.cfg.n_heads,
            )
            W_V = einops.rearrange(
                W_V,
                "(index d_head) d_model->index d_head d_model",
                index=self.cfg.n_heads,
            )

            sd[f"blocks.{l}.attn.W_Q"] = W_Q * w_ln_attn
            sd[f"blocks.{l}.attn.W_K"] = W_K * w_ln_attn
            sd[f"blocks.{l}.attn.W_V"] = W_V * w_ln_attn

            b_ln = opt.model.decoder.layers[l].self_attn_layer_norm.bias
            q_bias = einops.rearrange(
                opt.model.decoder.layers[l].self_attn.q_proj.bias,
                "(head_index d_head)->head_index d_head",
                head_index=self.cfg.n_heads,
                d_head=self.cfg.d_head,
            )
            k_bias = einops.rearrange(
                opt.model.decoder.layers[l].self_attn.k_proj.bias,
                "(head_index d_head)->head_index d_head",
                head_index=self.cfg.n_heads,
                d_head=self.cfg.d_head,
            )
            v_bias = einops.rearrange(
                opt.model.decoder.layers[l].self_attn.v_proj.bias,
                "(head_index d_head)->head_index d_head",
                head_index=self.cfg.n_heads,
                d_head=self.cfg.d_head,
            )

            sd[f"blocks.{l}.attn.b_Q"] = W_Q @ b_ln + q_bias
            sd[f"blocks.{l}.attn.b_K"] = W_K @ b_ln + k_bias
            sd[f"blocks.{l}.attn.b_V"] = W_V @ b_ln + v_bias

            W_O = opt.model.decoder.layers[l].self_attn.out_proj.weight
            W_O = einops.rearrange(
                W_O,
                "d_model (index d_head)->index d_model d_head",
                index=self.cfg.n_heads,
            )
            sd[f"blocks.{l}.attn.W_O"] = W_O
            sd[f"blocks.{l}.attn.b_O"] = opt.model.decoder.layers[
                l
            ].self_attn.out_proj.bias

            W_in = opt.model.decoder.layers[l].fc1.weight
            W_out = opt.model.decoder.layers[l].fc2.weight
            W_in_adj = (
                opt.model.decoder.layers[l].final_layer_norm.weight[None, :] * W_in
            )
            sd[f"blocks.{l}.mlp.W_in"] = W_in_adj
            sd[f"blocks.{l}.mlp.b_in"] = opt.model.decoder.layers[l].fc1.bias + (
                W_in @ opt.model.decoder.layers[l].final_layer_norm.bias
            )
            sd[f"blocks.{l}.mlp.W_out"] = W_out
            sd[f"blocks.{l}.mlp.b_out"] = opt.model.decoder.layers[l].fc2.bias
        W_U = opt.lm_head.weight
        sd["unembed.W_U"] = opt.model.decoder.final_layer_norm.weight[None, :] * W_U
        sd["unembed.b_U"] = W_U @ opt.model.decoder.final_layer_norm.bias
        self.load_state_dict(sd)

    def load_bloom_weights(self, bloom):
        raise NotImplementedError

    def init_weights(self):
        """
        Initialize weights according to default Pytorch initialization.

        LayerNorm weights are already initialized to 1.0 (and biases to 0.0)
        in the constructor
        """
        # Initialize weights with std 1/sqrt(d_model) so the vector has variance 1
        nn.init.normal_(self.embed.W_E, std=self.cfg.d_model ** (-0.5))
        nn.init.normal_(self.pos_embed.W_pos, std=self.cfg.d_model ** (-0.5))

        def init_linear_weight_and_bias(weight, bias):
            nn.init.kaiming_uniform_(weight, a=np.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

        for l in range(self.cfg.n_layers):
            init_linear_weight_and_bias(
                self.blocks[l].attn.W_Q, self.blocks[l].attn.b_Q
            )
            init_linear_weight_and_bias(
                self.blocks[l].attn.W_K, self.blocks[l].attn.b_K
            )
            init_linear_weight_and_bias(
                self.blocks[l].attn.W_V, self.blocks[l].attn.b_V
            )
            init_linear_weight_and_bias(
                self.blocks[l].attn.W_O, self.blocks[l].attn.b_O
            )
            init_linear_weight_and_bias(
                self.blocks[l].mlp.W_in, self.blocks[l].mlp.b_in
            )
            init_linear_weight_and_bias(
                self.blocks[l].mlp.W_out, self.blocks[l].mlp.b_out
            )

            if self.cfg.gated_act_fn:
                init_linear_weight_and_bias(
                    self.blocks[l].mlp.W_gate, self.blocks[l].mlp.b_gate
                )

        init_linear_weight_and_bias(self.unembed.W_U, self.unembed.b_U)

    def cross_entropy_loss(
        self, logits: torch.Tensor, tokens: torch.Tensor, return_per_token: bool = False
    ):
        """Cross entropy loss for the language model.

        Args:
            logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
            tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
            return_per_token (bool, optional): Whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs). Defaults to False.

        Returns:
            _type_: _description_
        """
        log_probs = F.log_softmax(logits, dim=-1)
        # Use torch.gather to find the log probs of the correct tokens
        # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
        # None and [..., 0] needed because the tensor used in gather must have the same rank.
        predicted_log_probs = log_probs[..., :-1, :].gather(
            dim=-1, index=tokens[..., 1:, None]
        )[..., 0]
        if return_per_token:
            return -predicted_log_probs
        else:
            return -predicted_log_probs.mean()
