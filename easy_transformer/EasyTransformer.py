# Import stuff
from dataclasses import dataclass
from dataclasses import fields
from typing import Callable, Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops

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

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.utils import gelu_new, to_numpy, get_corner, print_gpu_mem, get_sample_from_dataset


VALID_MODEL_NAMES = [
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
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neox-20b",
]


# TODO: Add Bloom
"""
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(self.cfg["d_model"], self.cfg["d_vocab"]))

    def forward(self, tokens):
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        return einops.rearrange(self.W_E[:, tokens], "d_model batch pos -> batch pos d_model")


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty(self.cfg["d_vocab"], self.cfg["d_model"]))
        self.b_U = nn.Parameter(torch.empty(self.cfg["d_vocab"]))

    def forward(self, tokens):
        return torch.einsum("vm,bpm->bpv", self.W_U, tokens) + self.b_U  # [batch, pos, d_vocab]


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg["d_model"], self.cfg["n_ctx"]))

    def forward(self, x):
        # Output shape [pos, d_model] - will be broadcast along batch dim
        return self.W_pos[:, : x.size(-1)].T  # [pos, d_model]


# LayerNormPre
# I fold the LayerNorm weights and biases into later weights and biases.
# This is just the 'center and normalise' part of LayerNorm
# Centering is equivalent to just deleting one direction of residual space,
# and is equivalent to centering the weight matrices of everything writing to the residual stream
# Normalising is a funkier non-linear operation, that projects the residual stream onto the unit hypersphere
class LayerNormPre(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg["eps"]

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]

    def forward(self, x):
        x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, d_model]
        scale = self.hook_scale(
            (einops.reduce(x.pow(2), "batch pos embed -> batch pos 1", "mean") + self.eps).sqrt()
        )  # [batch, pos, 1]
        return x / scale


# Attention
class Attention(nn.Module):
    def __init__(self, cfg, attn_type="global"):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"], self.cfg["d_model"]))
        self.W_K = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"], self.cfg["d_model"]))
        self.W_V = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"], self.cfg["d_model"]))
        self.W_O = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_model"], self.cfg["d_head"]))
        self.b_Q = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"]))
        self.b_K = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"]))
        self.b_V = nn.Parameter(torch.empty(self.cfg["n_heads"], self.cfg["d_head"]))
        self.b_O = nn.Parameter(torch.empty(self.cfg["d_model"]))

        self.attn_type = attn_type
        # Create a query_pos x key_pos mask, with True iff that query position
        # can attend to that key position
        causal_mask = torch.tril(torch.ones((self.cfg["n_ctx"], self.cfg["n_ctx"])).bool())
        if self.attn_type == "global":
            # For global attention, this is a lower triangular matrix - key <= query
            self.register_buffer("mask", causal_mask)
        elif self.attn_type == "local":
            # For local, this is banded, query - window_size < key <= query
            self.register_buffer("mask", torch.triu(causal_mask, 1 - self.cfg["window_size"]))
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        self.register_buffer("IGNORE", torch.tensor(-1e5))

        if self.cfg["use_attn_scale"]:
            self.attn_scale = np.sqrt(self.cfg["d_head"])
        else:
            self.attn_scale = 1.0

        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_attn = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, head_index, head_index, d_model]

    def forward(self, x):
        q = self.hook_q(torch.einsum("ihm,bpm->bpih", self.W_Q, x) + self.b_Q)  # [batch, pos, head_index, d_head]
        k = self.hook_k(torch.einsum("ihm,bpm->bpih", self.W_K, x) + self.b_K)  # [batch, pos, head_index, d_head]
        v = self.hook_v(torch.einsum("ihm,bpm->bpih", self.W_V, x) + self.b_V)  # [batch, pos, head_index, d_head]
        attn_scores = torch.einsum("bpih,bqih->bipq", q, k) / self.attn_scale  # [batch, head_index, query_pos, key_pos]
        attn_scores = self.hook_attn_scores(self.causal_mask(attn_scores))  # [batch, head_index, query_pos, key_pos]
        attn_matrix = self.hook_attn(F.softmax(attn_scores, dim=-1))  # [batch, head_index, query_pos, key_pos]
        z = self.hook_z(torch.einsum("bpih,biqp->bqih", v, attn_matrix))  # [batch, pos, head_index, d_head]
        if self.cfg["use_attn_result"]:
            result = self.hook_result(torch.einsum("imh,bqih->bqim", self.W_O, z))  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(result, "batch position index model->batch position model", "sum") + self.b_O
            )  # [batch, pos, d_model]
        else:
            out = torch.einsum("idh,bqih->bqd", self.W_O, z) + self.b_O  # [batch, pos, d_model]
        return out

    def causal_mask(self, attn_scores):
        return torch.where(self.mask[: attn_scores.size(-2), : attn_scores.size(-1)], attn_scores, self.IGNORE)


# MLP Layers
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty(self.cfg["d_mlp"], self.cfg["d_model"]))
        self.b_in = nn.Parameter(torch.empty(self.cfg["d_mlp"]))
        self.W_out = nn.Parameter(torch.empty(self.cfg["d_model"], self.cfg["d_mlp"]))
        self.b_out = nn.Parameter(torch.empty(self.cfg["d_model"]))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg["act_fn"] == "relu":
            self.act_fn = F.relu
        elif self.cfg["act_fn"] == "gelu_new":
            self.act_fn = gelu_new
        else:
            raise ValueError(f"Invalid activation function name: {self.cfg['act_fn']}")

    def forward(self, x):
        x = self.hook_pre(torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in)  # [batch, pos, d_mlp]
        x = self.hook_post(self.act_fn(x))  # [batch, pos, d_mlp]
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out  # [batch, pos, d_model]
        return x


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg, block_index):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNormPre(cfg)
        if not self.cfg["use_local_attn"]:
            self.attn = Attention(cfg, "global")
        else:
            attn_type = self.cfg["attn_types"][block_index]
            self.attn = Attention(cfg, attn_type)
        self.ln2 = LayerNormPre(cfg)
        self.mlp = MLP(cfg)

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, x):
        resid_pre = self.hook_resid_pre(x)  # [batch, pos, d_model]
        attn_out = self.hook_attn_out(self.attn(self.ln1(resid_pre)))  # [batch, pos, d_model]
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)  # [batch, pos, d_model]
        mlp_out = self.hook_mlp_out(self.mlp(self.ln2(resid_mid)))  # [batch, pos, d_model]
        resid_post = self.hook_resid_post(resid_mid + mlp_out)  # [batch, pos, d_model]
        return resid_post


# Full transformer
class EasyTransformer(HookedRootModule):
    """
    This class implements a full Transformer using the above components, with
    HookPoints on every interesting activation. It inherits from HookedRootModule.
    It is initialised with a model_name, and automatically loads the model weights
    for that model, loads them into this model, folds in LayerNorm and centers
    the weights
    """

    def __init__(self, model_name, use_attn_result=False, model=None, keep_original_model=False, center_weights=True):
        """
        model_name (str): The name of the model to load, via HuggingFace
        use_attn_result (bool): Says whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        model: The loaded model from HuggingFace. If None, it is automatically
            loaded from HuggingFace - this just saves memory if the model was
            already loaded into RAM
        keep_original_model (bool): If False, the original HuggingFace model is
            deleted, otherwise it's kept as a self.model attribute
        """
        assert model_name in VALID_MODEL_NAMES
        super().__init__()
        self.model_name = model_name
        self.model_type = self.get_model_type(model_name)
        if model is not None:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.cfg = self.convert_config(self.model.config, model_type=self.model_type)
        self.cfg["use_attn_result"] = use_attn_result
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d_model]

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.cfg, block_index) for block_index in range(self.cfg["n_layers"])]
        )
        self.ln_final = LayerNormPre(self.cfg)
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

        # Set the average of each weight matrix writing to the residual stream to zero
        # (Layer Norm removes the mean anyway, so this simplifies the weights
        # without changing the computation)
        if center_weights:
            self.center_weights()

        if not keep_original_model:
            # Delete the original model to save memory
            del self.model

    def forward(self, x):
        # Input x is either a batch of tokens ([batch, pos]) or a text string
        if type(x) == str or type(x) == list:
            # If text, convert to tokens (batch_size=1)
            x = self.to_tokens(x)
        embed = self.hook_embed(self.embed(x))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(x))  # [batch, pos, d_model]
        residual = embed + pos_embed  # [batch, pos, d_model]
        for block in self.blocks:
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(residual)  # [batch, pos, d_model]
        x = self.unembed(self.ln_final(residual))  # [batch, pos, d_vocab]
        return x

    def to_tokens(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True)["input_ids"]

    def get_model_type(self, model_name):
        if "gpt2" in model_name:
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

    def convert_config(self, config, model_type):
        if model_type == "neo":
            cfg = {
                "d_model": config.hidden_size,
                "d_head": config.hidden_size // config.num_heads,
                "n_heads": config.num_heads,
                "d_mlp": config.hidden_size * 4,
                "n_layers": config.num_layers,
                "n_ctx": config.max_position_embeddings,
                "eps": config.layer_norm_epsilon,
                "d_vocab": config.vocab_size,
                "attn_types": config.attention_layers,
                "act_fn": config.activation_function,
                "use_attn_scale": False,
                "use_local_attn": True,
                "window_size": config.window_size,
            }
        elif model_type == "gpt2":
            cfg = {
                "d_model": config.n_embd,
                "d_head": config.n_embd // config.n_head,
                "n_heads": config.n_head,
                "d_mlp": config.n_embd * 4,
                "n_layers": config.n_layer,
                "n_ctx": config.n_ctx,
                "eps": config.layer_norm_epsilon,
                "d_vocab": config.vocab_size,
                "act_fn": config.activation_function,
                "use_attn_scale": True,
                "use_local_attn": False,
            }
        elif model_type == "opt":
            cfg = {
                "d_model": config.hidden_size,
                "d_head": config.hidden_size // config.num_attention_heads,
                "n_heads": config.num_attention_heads,
                "d_mlp": config.ffn_dim,
                "n_layers": config.num_hidden_layers,
                "n_ctx": config.max_position_embeddings,
                "eps": 1e-5,
                "d_vocab": config.vocab_size,
                "act_fn": config.activation_function,
                "use_attn_scale": True,
                "use_local_attn": False,
            }
        elif model_type == "gptj":
            raise NotImplementedError
        elif model_type == "neox":
            raise NotImplementedError

        cfg["model_name"] = self.model_name
        cfg["model_type"] = model_type
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
            block.attn.W_O.data -= einops.reduce(block.attn.W_O, "index d_model d_head -> index 1 d_head", "mean")
            block.mlp.W_out.data -= block.mlp.W_out.mean(0, keepdim=True)

    def load_gpt2_weights(self, gpt2):
        sd = self.state_dict()

        sd["embed.W_E"] = gpt2.transformer.wte.weight.T
        sd["pos_embed.W_pos"] = gpt2.transformer.wpe.weight.T

        for l in range(self.cfg["n_layers"]):
            # In GPT-2, q,k,v are produced by one big linear map, whose output is
            # concat([q, k, v])
            W = gpt2.transformer.h[l].attn.c_attn.weight
            w_ln_attn = gpt2.transformer.h[l].ln_1.weight
            W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
            W_Q = einops.rearrange(W_Q, "m (i h)->i h m", i=self.cfg["n_heads"])
            W_K = einops.rearrange(W_K, "m (i h)->i h m", i=self.cfg["n_heads"])
            W_V = einops.rearrange(W_V, "m (i h)->i h m", i=self.cfg["n_heads"])

            # Fold in layer norm weights
            sd[f"blocks.{l}.attn.W_Q"] = W_Q * w_ln_attn
            sd[f"blocks.{l}.attn.W_K"] = W_K * w_ln_attn
            sd[f"blocks.{l}.attn.W_V"] = W_V * w_ln_attn

            b_ln = gpt2.transformer.h[l].ln_1.bias
            qkv_bias = gpt2.transformer.h[l].attn.c_attn.bias
            qkv_bias = einops.rearrange(
                qkv_bias, "(qkv index head)->qkv index head", qkv=3, index=self.cfg["n_heads"], head=self.cfg["d_head"]
            )
            # Fold in layer norm biases
            sd[f"blocks.{l}.attn.b_Q"] = W_Q @ b_ln + qkv_bias[0]
            sd[f"blocks.{l}.attn.b_K"] = W_K @ b_ln + qkv_bias[1]
            sd[f"blocks.{l}.attn.b_V"] = W_V @ b_ln + qkv_bias[2]

            W_O = gpt2.transformer.h[l].attn.c_proj.weight
            W_O = einops.rearrange(W_O, "(i h) m->i m h", i=self.cfg["n_heads"])
            sd[f"blocks.{l}.attn.W_O"] = W_O
            sd[f"blocks.{l}.attn.b_O"] = gpt2.transformer.h[l].attn.c_proj.bias

            W_in = gpt2.transformer.h[l].mlp.c_fc.weight.T
            W_out = gpt2.transformer.h[l].mlp.c_proj.weight.T
            # Fold in layer norm weights
            W_in_adj = gpt2.transformer.h[l].ln_2.weight[None, :] * W_in
            sd[f"blocks.{l}.mlp.W_in"] = W_in_adj
            # Fold in layer norm biases
            sd[f"blocks.{l}.mlp.b_in"] = gpt2.transformer.h[l].mlp.c_fc.bias + (W_in @ gpt2.transformer.h[l].ln_2.bias)
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

        for l in range(self.cfg["n_layers"]):
            w_ln_attn = neo.transformer.h[l].ln_1.weight
            W_Q = neo.transformer.h[l].attn.attention.q_proj.weight
            W_K = neo.transformer.h[l].attn.attention.k_proj.weight
            W_V = neo.transformer.h[l].attn.attention.v_proj.weight
            W_Q = einops.rearrange(W_Q, "(i h) m->i h m", i=self.cfg["n_heads"])
            W_K = einops.rearrange(W_K, "(i h) m->i h m", i=self.cfg["n_heads"])
            W_V = einops.rearrange(W_V, "(i h) m->i h m", i=self.cfg["n_heads"])

            sd[f"blocks.{l}.attn.W_Q"] = W_Q * w_ln_attn
            sd[f"blocks.{l}.attn.W_K"] = W_K * w_ln_attn
            sd[f"blocks.{l}.attn.W_V"] = W_V * w_ln_attn

            b_ln = neo.transformer.h[l].ln_1.bias
            sd[f"blocks.{l}.attn.b_Q"] = W_Q @ b_ln
            sd[f"blocks.{l}.attn.b_K"] = W_K @ b_ln
            sd[f"blocks.{l}.attn.b_V"] = W_V @ b_ln

            W_O = neo.transformer.h[l].attn.attention.out_proj.weight
            W_O = einops.rearrange(W_O, "m (i h)->i m h", i=self.cfg["n_heads"])
            sd[f"blocks.{l}.attn.W_O"] = W_O
            sd[f"blocks.{l}.attn.b_O"] = neo.transformer.h[l].attn.attention.out_proj.bias

            W_in = neo.transformer.h[l].mlp.c_fc.weight
            W_out = neo.transformer.h[l].mlp.c_proj.weight
            W_in_adj = neo.transformer.h[l].ln_2.weight[None, :] * W_in
            sd[f"blocks.{l}.mlp.W_in"] = W_in_adj
            sd[f"blocks.{l}.mlp.b_in"] = neo.transformer.h[l].mlp.c_fc.bias + (W_in @ neo.transformer.h[l].ln_2.bias)
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

        for l in range(self.cfg["n_layers"]):
            w_ln_attn = opt.model.decoder.layers[l].self_attn_layer_norm.weight
            W_Q = opt.model.decoder.layers[l].self_attn.q_proj.weight
            W_K = opt.model.decoder.layers[l].self_attn.k_proj.weight
            W_V = opt.model.decoder.layers[l].self_attn.v_proj.weight
            W_Q = einops.rearrange(W_Q, "(index d_head) d_model->index d_head d_model", i=self.cfg["n_heads"])
            W_K = einops.rearrange(W_K, "(index d_head) d_model->index d_head d_model", i=self.cfg["n_heads"])
            W_V = einops.rearrange(W_V, "(index d_head) d_model->index d_head d_model", i=self.cfg["n_heads"])

            sd[f"blocks.{l}.attn.W_Q"] = W_Q * w_ln_attn
            sd[f"blocks.{l}.attn.W_K"] = W_K * w_ln_attn
            sd[f"blocks.{l}.attn.W_V"] = W_V * w_ln_attn

            b_ln = opt.model.decoder.layers[l].self_attn_layer_norm.bias
            q_bias = einops.rearrange(
                opt.model.decoder.layers[l].self_attn.q_proj.bias,
                "(head_index d_head)->head_index d_head",
                head_index=self.cfg["n_heads"],
                d_head=self.cfg["d_head"],
            )
            k_bias = einops.rearrange(
                opt.model.decoder.layers[l].self_attn.k_proj.bias,
                "(head_index d_head)->head_index d_head",
                head_index=self.cfg["n_heads"],
                d_head=self.cfg["d_head"],
            )
            v_bias = einops.rearrange(
                opt.model.decoder.layers[l].self_attn.v_proj.bias,
                "(head_index d_head)->head_index d_head",
                head_index=self.cfg["n_heads"],
                d_head=self.cfg["d_head"],
            )

            sd[f"blocks.{l}.attn.b_Q"] = W_Q @ b_ln + q_bias
            sd[f"blocks.{l}.attn.b_K"] = W_K @ b_ln + k_bias
            sd[f"blocks.{l}.attn.b_V"] = W_V @ b_ln + v_bias

            W_O = opt.model.decoder.layers[l].self_attn.out_proj.weight
            W_O = einops.rearrange(W_O, "d_model (index d_head)->index d_model d_head", i=self.cfg["n_heads"])
            sd[f"blocks.{l}.attn.W_O"] = W_O
            sd[f"blocks.{l}.attn.b_O"] = opt.model.decoder.layers[l].self_attn.out_proj.bias

            W_in = opt.model.decoder.layers[l].fc1.weight
            W_out = opt.model.decoder.layers[l].fc2.weight
            W_in_adj = opt.model.decoder.layers[l].final_layer_norm.weight[None, :] * W_in
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


# Ablation implem


class ExperimentMetric:
    def __init__(
        self,
        metric: Callable[[EasyTransformer, List[str]], torch.Tensor],
        dataset: List[str],
        scalar_metric=True,
        relative_metric=True,
    ):
        self.relative_metric = relative_metric
        self.metric = metric  # metric can return any tensor shape. Can call run_with_hook with reset_hooks_start=False
        self.scalar_metric = scalar_metric
        self.baseline = None  # metric without ablation
        self.dataset = dataset
        self.shape = None

    def set_baseline(self, model):
        model.reset_hooks()
        base_metric = self.metric(model, self.dataset)
        self.baseline = base_metric
        self.shape = base_metric.shape

    def compute_metric(self, model):
        assert (self.baseline is not None) or not (self.relative_metric), "Baseline has not been set in relative mean"
        out = self.metric(model, self.dataset)
        if self.scalar_metric:
            assert len(out.shape) == 0, "Output of scalar metric has shape of length > 0"
        self.shape = out.shape
        if self.relative_metric:
            out = (self.baseline / out) - 1
        return out


class ExperimentConfig:
    def __init__(
        self,
        target_module: str = "attn_head",
        layers: Union[Tuple[int, int], str] = "all",
        heads: Union[List[int], str] = "all",
        verbose: bool = False,
        head_circuit: str = "z",
    ):
        assert target_module in ["mlp", "attn_layer", "attn_head"]
        assert head_circuit in ["z", "q", "v", "k", "attn", "attn_scores"]

        self.target_module = target_module
        self.head_circuit = head_circuit
        self.layers = layers
        self.heads = heads
        self.dataset = None
        self.verbose = verbose

        self.beg_layer = None  # layers where the ablation begins and ends
        self.end_layer = None

    def adapt_to_model(self, model: EasyTransformer):
        """Return a new experiment config that fits the model."""
        model_cfg = self.copy()
        if self.target_module == "attn_head":
            if self.heads == "all":
                model_cfg.heads = list(range(model.cfg["n_heads"]))

        if self.layers == "all":
            model_cfg.beg_layer = 0
            model_cfg.end_layer = model.cfg["n_layers"]
        else:
            model_cfg.beg_layer, model_cfg.end_layer = self.layers
        return model_cfg

    def copy(self):
        copy = self.__class__()
        for name, attr in vars(self).items():
            if type(attr) == list:
                setattr(copy, name, attr.copy())
            else:
                setattr(copy, name, attr)
        return copy

    def __str__(self):
        str_print = f"--- {self.__class__.__name__}: ---\n"
        for name, attr in vars(self).items():
            attr = getattr(self, name)
            attr_str = f"* {name}: "

            if name == "mean_dataset" and self.compute_means and attr is not None:
                attr_str += get_sample_from_dataset(self.mean_dataset)
            elif name == "dataset" and attr is not None:
                attr_str += get_sample_from_dataset(self.dataset)
            else:
                attr_str += str(attr)
            attr_str += "\n"
            str_print += attr_str
        return str_print

    def __repr__(self):
        return self.__str__()


def zero_fn(z, hk):
    return torch.zeros(z.shape)


def cst_fn(z, cst, hook):
    return cst


def neg_fn(z, hk):
    return -z


class AblationConfig(ExperimentConfig):
    def __init__(
        self,
        abl_type: str = "zero",
        mean_dataset: List[str] = None,
        cache_means: bool = True,
        abl_fn: Callable[[torch.tensor, torch.tensor, HookPoint], torch.tensor] = None,
        **kwargs,
    ):
        assert abl_type in ["mean", "zero", "neg", "custom"]
        assert not (abl_type == "custom" and abl_fn is None), "You must specify you ablation function"
        super().__init__(**kwargs)

        self.abl_type = abl_type
        self.mean_dataset = mean_dataset
        self.dataset = None
        self.cache_means = cache_means
        self.compute_means = abl_type == "mean" or abl_type == "custom"
        self.abl_fn = abl_fn

        if abl_type == "zero":
            self.abl_fn = zero_fn
        if abl_type == "neg":
            self.abl_fn = neg_fn
        if abl_type == "mean":
            self.abl_fn = cst_fn


class PatchingConfig(ExperimentConfig):
    """Configuration for patching activations from the source dataset to the taregt dataset"""

    def __init__(
        self,
        source_dataset: List[str] = None,
        target_dataset: List[str] = None,
        patch_fn: Callable[[torch.tensor, torch.tensor, HookPoint], torch.tensor] = None,
        cache_act: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.cache_act = cache_act  # if we should cache activation. Take more GPU memory but faster to run
        self.patch_fn = patch_fn
        if patch_fn is None:  # default patch_fn
            self.patch_fn = cst_fn


# TODO : loss metric, zero ablation, mean ablation
# TODO : add different direction for the mean, add tokens, change type of
# datasets, make deterministic


class EasyExperiment:
    """A virtual class to interatively apply hooks to layers or heads. The children class only needs to define the methods
    get_hook"""

    def __init__(self, model: EasyTransformer, config: ExperimentConfig, metric: ExperimentMetric):
        self.model = model
        self.metric = metric
        self.cfg = config.adapt_to_model(model)
        self.cfg.dataset = self.metric.dataset

    def run_experiment(self):
        self.metric.set_baseline(self.model)
        results = torch.empty(self.get_result_shape())
        if self.cfg.verbose:
            print(self.cfg)
        self.metric.set_baseline(self.model)
        results = torch.empty(self.get_result_shape())
        for layer in tqdm(range(self.cfg.beg_layer, self.cfg.end_layer)):
            if self.cfg.target_module == "attn_head":
                for head in self.cfg.heads:
                    hook = self.get_hook(layer, head)
                    results[layer, head] = self.compute_metric(hook).cpu().detach()
            else:
                hook = self.get_hook(layer)
                results[layer] = self.compute_metric(hook).cpu().detach()
        self.model.reset_hooks()
        if len(results.shape) < 2:
            results = results.unsqueeze(0)  # to make sure that we can always easily plot the results
        return results

    def get_result_shape(self):
        if self.cfg.target_module == "attn_head":
            return (self.cfg.end_layer - self.cfg.beg_layer, len(self.cfg.heads)) + self.metric.shape
        else:
            return (self.cfg.end_layer - self.cfg.beg_layer,) + self.metric.shape

    def compute_metric(self, abl_hook):
        self.model.reset_hooks()
        hk_name, hk = abl_hook
        self.model.add_hook(hk_name, hk)
        return self.metric.compute_metric(self.model)

    def get_target(self, layer, head):
        if head is not None:
            hook_name = f"blocks.{layer}.attn.hook_{self.cfg.head_circuit}"
            dim = (
                1 if "hook_attn" in hook_name else 2
            )  # hook_attn and hook_attn_scores are [batch,nb_head,seq_len, seq_len] and the other activation of head (z, q, v,k) are [batch, seq_len, nb_head, head_dim]
        else:
            if self.cfg.target_module == "mlp":
                hook_name = f"blocks.{layer}.mlp.hook_post"
            else:
                hook_name = f"blocks.{layer}.hook_attn_out"
            dim = None  # all the activation dimensions are ablated
        return hook_name, dim


class EasyAblation(EasyExperiment):
    def __init__(self, model: EasyTransformer, config: AblationConfig, metric: ExperimentMetric):
        super().__init__(model, config, metric)
        assert type(config) == AblationConfig
        if self.cfg.mean_dataset is None and config.compute_means:
            self.cfg.mean_dataset = self.metric.dataset
        if self.cfg.cache_means and self.cfg.compute_means:
            self.get_all_mean()

    def run_ablation(self):
        return self.run_experiment()

    def get_hook(self, layer, head=None):
        # If the target is a layer, head is None.
        hook_name, dim = self.get_target(layer, head)
        mean = None
        if self.cfg.compute_means:
            if self.cfg.cache_means:
                mean = self.mean_cache[hook_name]
            else:
                mean = self.get_mean(hook_name)

        abl_hook = get_act_hook(self.cfg.abl_fn, mean, head, dim=dim)
        return (hook_name, abl_hook)

    def get_all_mean(self):
        cache = {}
        self.model.reset_hooks()
        self.model.cache_all(cache)
        self.model(self.cfg.mean_dataset)
        self.mean_cache = {}
        for hk in cache.keys():
            mean = torch.mean(cache[hk], dim=0, keepdim=False).clone()  # we compute the mean along the batch dim
            self.mean_cache[hk] = einops.repeat(mean, "... -> s ...", s=cache[hk].shape[0])

    def get_mean(self, hook_name):
        cache = {}

        def cache_hook(z, hook):
            cache[hook_name] = z.detach().to("cuda")

        self.model.reset_hooks()
        self.model.run_with_hooks(self.cfg.mean_dataset, fwd_hooks=[(hook_name, cache_hook)])
        mean = torch.mean(cache[hook_name], dim=0, keepdim=False)
        return einops.repeat(mean, "... -> s ...", s=cache[hook_name].shape[0])


class EasyPatching(EasyExperiment):
    def __init__(self, model: EasyTransformer, config: PatchingConfig, metric: ExperimentMetric):
        super().__init__(model, config, metric)
        assert type(config) == PatchingConfig, f"{type(config)}"
        if self.cfg.cache_act:
            self.get_all_act()

    def run_patching(self):
        return self.run_experiment()

    def get_hook(self, layer, head=None):
        # If the target is a layer, head is None.
        hook_name, dim = self.get_target(layer, head)
        if self.cfg.cache_act:
            act = self.act_cache[hook_name]  # activation on the source dataset
        else:
            act = self.get_act(hook_name)

        hook = get_act_hook(self.cfg.patch_fn, act, head, dim=dim)
        return (hook_name, hook)

    def get_all_act(self):
        self.act_cache = {}
        self.model.reset_hooks()
        self.model.cache_all(self.act_cache)
        self.model(self.cfg.source_dataset)

    def get_act(self, hook_name):
        cache = {}

        def cache_hook(z, hook):
            cache[hook_name] = z.detach().to("cuda")

        self.model.reset_hooks()
        self.model.run_with_hooks(self.cfg.mean_dataset, fwd_hooks=[(hook_name, cache_hook)])
        return cache[hook_name]


def get_act_hook(fn, alt_act=None, idx=None, dim=None):
    """Return an hook that modify the activation on the fly. alt_act (Alternative activations) is a tensor of the same shape of the z.
    E.g. It can be the mean activation or the activations on other dataset."""
    if alt_act is not None:

        def custom_hook(z, hook):
            if dim is None:  # mean and z have the same shape, the mean is constant along the batch dimension
                return fn(z, alt_act, hook)
            if dim == 0:
                z[idx] = fn(z[idx], alt_act[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], alt_act[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], alt_act[:, :, idx], hook)
            return z

    else:

        def custom_hook(z, hook):
            if dim is None:
                return fn(z, hook)
            if dim == 0:
                z[idx] = fn(z[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], hook)
            return z

    return custom_hook
