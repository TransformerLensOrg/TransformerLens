#%%
# % TODO: ablations last
# % and 2.2 improvements: do things with making more specific to transformers
# % ablations later
# % back reference equations
# % not HYPOTHESISE, do the computationally intractable
# % do completeness, minimality NOT methods first
#%%
import abc
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

assert torch.cuda.device_count() == 1
from ioi_utils import all_subsets
from ioi_circuit_extraction import CIRCUIT, RELEVANT_TOKENS
from copy import deepcopy
from time import ctime
import io
from random import randint as ri
from easy_transformer import EasyTransformer
from functools import partial
from ioi_utils import all_subsets, logit_diff, probs, show_attention_patterns
from ioi_dataset import BABA_EARLY_IOS, BABA_LATE_IOS, ABBA_EARLY_IOS, ABBA_LATE_IOS
import logging
import sys
from ioi_dataset import *
from ioi_utils import max_2d
import IPython
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from easy_transformer.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    print_gpu_mem,
)  # helper functions
from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.EasyTransformer import (
    EasyTransformer,
    TransformerBlock,
    MLP,
    Attention,
    LayerNormPre,
    PosEmbed,
    Unembed,
    Embed,
)
from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
    get_act_hook,
)
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional, Iterable
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from sklearn.linear_model import LinearRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import spacy
import re
from einops import rearrange
import einops
from pprint import pprint
import gc
from datasets import load_dataset
from IPython import get_ipython
import matplotlib.pyplot as plt
import random as rd

from ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    BABA_EARLY_IOS,
    BABA_LATE_IOS,
    ABBA_TEMPLATES,
)
from ioi_utils import (
    attention_on_token,
    # patch_positions,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    # show_attention_patterns,
    safe_del,
)

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")


def e(mess=""):
    print_gpu_mem(mess)
    torch.cuda.empty_cache()


#%%
model = EasyTransformer("gpt2", use_attn_result=True).cuda()
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts(
    ("S2", "RAND")
)  # we flip the second b for a random c
acca_dataset = ioi_dataset.gen_flipped_prompts(("S", "RAND"))
dcc_dataset = acca_dataset.gen_flipped_prompts(("IO", "RAND"))
cbb_dataset = ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
acba_dataset = ioi_dataset.gen_flipped_prompts(("S1", "RAND"))
adea_dataset = ioi_dataset.gen_flipped_prompts(("S", "RAND")).gen_flipped_prompts(
    ("S1", "RAND")
)
totally_diff_dataset = IOIDataset(N=ioi_dataset.N, prompt_type=ioi_dataset.prompt_type)
all_diff_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
bcca_dataset = ioi_dataset.gen_flipped_prompts(("IO", "RAND")).gen_flipped_prompts(
    ("S", "RAND")
)
from ioi_utils import logit_diff

ABCm_dataset = IOIDataset(prompt_type="ABC mixed", N=N, tokenizer=model.tokenizer)
ABC_dataset = IOIDataset(prompt_type="ABC", N=N, tokenizer=model.tokenizer)
BAC_dataset = IOIDataset("BAC", N, model.tokenizer)
mixed_dataset = IOIDataset("ABC mixed", N, model.tokenizer)

circuit = deepcopy(CIRCUIT)


def patch_positions(z, source_act, hook, positions=["end"]):
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z


def attention_probs(
    model, text_prompts, variation=True, scale=True
):  # we have to redefine logit differences to use the new abba dataset
    """Difference between the IO and the S logits at the "to" token"""
    cache_patched = {}
    model.cache_some(
        cache_patched, lambda x: x in hook_names
    )  # we only cache the activation we're interested
    logits = model(text_prompts).detach()
    # we want to measure Mean(Patched/baseline) and not Mean(Patched)/Mean(baseline)
    # ... but do this elsewhere as otherwise we bad
    # attn score of head HEAD at token "to" (end) to token IO
    assert variation or not scale
    attn_probs_variation_by_keys = []
    for key in ["IO", "S", "S2"]:
        attn_probs_variation = []
        for i, hook_name in enumerate(hook_names):
            layer = layers[i]
            for head in heads_by_layer[layer]:
                attn_probs_patched = cache_patched[hook_name][
                    torch.arange(len(text_prompts)),
                    head,
                    ioi_dataset.word_idx["end"],
                    ioi_dataset.word_idx[key],
                ]
                attn_probs_base = cache_baseline[hook_name][
                    torch.arange(len(text_prompts)),
                    head,
                    ioi_dataset.word_idx["end"],
                    ioi_dataset.word_idx[key],
                ]
                if variation:
                    res = attn_probs_patched - attn_probs_base
                    if scale:
                        res /= attn_probs_base
                    res = res.mean().unsqueeze(dim=0)
                    attn_probs_variation.append(res)
                else:
                    attn_probs_variation.append(
                        attn_probs_patched.mean().unsqueeze(dim=0)
                    )
        attn_probs_variation_by_keys.append(
            torch.cat(attn_probs_variation).mean(dim=0, keepdim=True)
        )

    attn_probs_variation_by_keys = torch.cat(attn_probs_variation_by_keys, dim=0)
    return attn_probs_variation_by_keys.detach().cpu()


#%% [markdown] let's try to remove MO MLPS from the circuit
circuit = deepcopy(CIRCUIT)
heads_to_keep = get_heads_circuit(ioi_dataset, circuit=circuit)
mlps_to_keep = get_mlps_circuit(ioi_dataset, mlps=[0, 1, 2, 3, 4, 5, 10, 11])
e()
model.reset_hooks()

# model, _ = do_circuit_extraction(
#     model=model,
#     heads_to_keep=heads_to_keep,
#     mlps_to_remove={},
#     # mlps_to_keep=mlps_to_keep,
#     ioi_dataset=ioi_dataset,
#     mean_dataset=abca_dataset,
# )

prefixed_dataset = IOIDataset(N=100, prompt_type="mixed")

for a_dataset in [
    prefixed_dataset,
    ioi_dataset,
    ABC_dataset,
    BAC_dataset,
    mixed_dataset,
]:
    circuit_logit_diff = logit_diff(model, a_dataset)
    circuit_probs = probs(model, a_dataset)
    print(f"{circuit_logit_diff=} {circuit_probs=}")

#%% [markdown] Add some ablation of MLP0 to try and tell what's up
model.reset_hooks()
metric = ExperimentMetric(metric=logit_diff, dataset=abca_dataset, relative_metric=True)
config = AblationConfig(
    abl_type="random",
    mean_dataset=abca_dataset.text_prompts,
    target_module="mlp",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    nb_metric_iteration=1,
    max_seq_len=ioi_dataset.max_len,
)
abl = EasyAblation(
    model, config, metric
)  # , mean_by_groups=True, groups=ioi_dataset.groups)
e()

ablate_these = [1, 2, 3]  # single numbers for MLPs, tuples for heads
# ablate_these = max_2d(-result, 10)[0]
ablate_these += [
    (5, 9),
    (5, 8),
    (0, 10),
    (4, 6),
    (3, 10),
    (4, 0),
    (3, 8),
    (3, 7),
    (5, 2),
    (6, 5),
]
ablate_these = []
# run some below cell to see the max impactful heads

for this in ablate_these:
    if isinstance(this, int):
        layer = this
        head_idx = None
        cur_tensor_name = f"blocks.{layer}.hook_mlp_out"
    elif isinstance(this, tuple):
        layer, head_idx = this
        cur_tensor_name = f"blocks.{layer}.attn.hook_result"
    else:
        raise ValueError(this)

    def ablation_hook(z, act, hook):
        # batch, seq, head dim, because get_act_hook hides scary things from us
        # TODO probably change this to random ablation when that arrives
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        # assert hook.name == cur_tensor_name, (hook.name, cur_tensor_name)
        # sad, that only works when cur_tensor_name doesn't change (?!)
        assert len(list(z.shape)) == 3, z.shape
        assert list(z.shape) == list(act.shape), (z.shape, act.shape)

        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S2"]] = act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S2"]
        ]  # hope that we don't see changing values of mean_cached_values...
        return z

    cur_hook = get_act_hook(
        ablation_hook,
        alt_act=abl.mean_cache[cur_tensor_name],
        idx=head_idx,
        dim=2 if head_idx is not None else None,
    )
    model.add_hook(cur_tensor_name, cur_hook)

# [markdown] After adding some hooks we see that yeah MLP0 ablations destroy S2 probs -> this one is at end
#%%
my_toks = [
    2215,
    5335,
    290,
    1757,
    1816,
    284,
    262,
    3650,
    11,
    1757,
    2921,
    257,
    4144,
    284,
    5335,
]  # this is the John and Mary one

mary_tok = 5335
john_tok = 1757

model.reset_hooks()
logits = model(torch.Tensor([my_toks]).long()).detach().cpu()
assert mary_tok in torch.topk(logits[0, -2], 1).indices, (
    torch.topk(logits[0, -2], 5),
    logits[0, -2, john_tok],
)
# mary_res = logits[0, -2, mary_tok]
# john_res = logits[0, -2, john_tok]


def replace(my_list, a, b):
    return [b if x == a else x for x in my_list]


from random import randint as ri
from copy import deepcopy

cnt = 0
bet = 0
bet_sub = 0
bet_sub_2 = 0
for it in tqdm(range(1000)):
    cur_list = deepcopy(my_toks)
    new_mary_tok = my_list[ri(0, -1 + len(my_list))]  # ri(0, 50_000)
    new_john_tok = my_list[ri(0, -1 + len(my_list))]  # ri(0, 50_000)
    cur_list = replace(cur_list, mary_tok, new_mary_tok)
    cur_list = replace(cur_list, john_tok, new_john_tok)
    logits = model(torch.Tensor([cur_list]).long()).detach().cpu()[0, -2]
    top_100 = torch.topk(logits, 100).indices

    if logits[new_mary_tok] > logits[new_john_tok]:
        bet += 1

    if new_mary_tok in top_100 or new_john_tok in top_100:
        cnt += 1

        if logits[new_mary_tok] > logits[new_john_tok]:
            bet_sub += 1
        else:
            bet_sub_2 += 1
#%%
NEW_CIRCUIT = {
    # old name mover
    (9, 6): ["S2", "end"],
    (9, 9): ["S+1", "end"],
    (10, 0): ["end"],
    # old s2 inhibition
    (7, 3): ["S2", "end"],
    (7, 9): ["S+1", "end"],
    (10, 7): [],
    (11, 10): [],
    # old induction
    (5, 5): ["end"],
    (5, 8): ["S"],
    (5, 9): [],
    (6, 9): [],
    # old duplicate
    (0, 1): ["IO"],
    (0, 10): ["end"],
    (3, 0): [],
    # old previous token
    (2, 2): [],
    (2, 9): ["S", "end"],
    (4, 11): ["S2"],
}

NEGS = {
    (10, 7): ["end"],
    (11, 10): ["end"],
}

model.reset_hooks()
e()
whole_circuit_base, whole_std = logit_diff(model, ioi_dataset.text_prompts, std=True)
print(f"{whole_circuit_base=} {whole_std=}")

heads_to_keep_new = {}
for head in NEW_CIRCUIT.keys():
    heads_to_keep_new[head] = get_extracted_idx(NEW_CIRCUIT[head], ioi_dataset)
e()
new_model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep_new,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)
e()
new_circuit_base, new_std = logit_diff(new_model, ioi_dataset.text_prompts, std=True)
print(f"{new_circuit_base=} {new_std=}")
model.reset_hooks()
heads_to_keep_neg_new = heads_to_keep_new.copy()
heads_to_keep = {}
for circuit_class in CIRCUIT.keys():
    if circuit_class == "negative":
        continue
    for head in CIRCUIT[circuit_class]:
        heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)
heads_to_keep_neg = heads_to_keep.copy()
for head in NEGS.keys():
    heads_to_keep_neg_new[head] = get_extracted_idx(NEGS[head], ioi_dataset)
    heads_to_keep_neg[head] = get_extracted_idx(NEGS[head], ioi_dataset)
e()
calib_model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep_neg,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)
e()
calib_model_base, calib_std = logit_diff(
    calib_model, ioi_dataset.text_prompts, std=True
)
print(f"{calib_model_base=} {calib_std=}")
# %%
seq_len = ioi_dataset.toks.shape[1]

for mlp in range(12):
    model.reset_hooks()
    calib_model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep_neg,
        mlps_to_remove={mlp: [list(range(seq_len)) for _ in range(N)]},
        ioi_dataset=ioi_dataset,
    )
    e()
    mlp_base, mlp_std = logit_diff(calib_model, ioi_dataset.text_prompts, std=True)
    print(f"{mlp} {mlp_base=} {mlp_std=}")
#%% # quick S2 experiment
from ioi_circuit_extraction import ARTHUR_CIRCUIT

heads_to_keep = {}

for circuit_class in ARTHUR_CIRCUIT.keys():
    for head in ARTHUR_CIRCUIT[circuit_class]:
        heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)

model, abl = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)

for layer, head_idx in [(7, 9), (8, 6), (7, 3), (8, 10)]:
    # use abl.mean_cache
    cur_tensor_name = f"blocks.{layer}.attn.hook_v"
    s2_token_idxs = get_extracted_idx(["S2"], ioi_dataset)
    mean_cached_values = (
        abl.mean_cache[cur_tensor_name][:, :, head_idx, :].cpu().detach()
    )

    def s2_v_ablation_hook(
        z, act, hook
    ):  # batch, seq, head dim, because get_act_hook hides scary things from us
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_v", hook.name
        assert len(list(z.shape)) == 3, z.shape
        assert list(z.shape) == list(act.shape), (z.shape, act.shape)

        true_s2_values = z[:, s2_token_idxs, :].clone()
        z = (
            mean_cached_values.cuda()
        )  # hope that we don't see chaning values of mean_cached_values...
        z[:, s2_token_idxs, :] = true_s2_values

        return z

    cur_hook = get_act_hook(
        s2_v_ablation_hook,
        alt_act=abl.mean_cache[cur_tensor_name],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(cur_tensor_name, cur_hook)

new_ld, new_ld_std = logit_diff(model, ioi_dataset.text_prompts, std=True)
new_ld, new_ld_std
#%% # look at what's affecting the V stuff
from ioi_circuit_extraction import ARTHUR_CIRCUIT
from ioi_utils import probs

heads_to_keep = {}
for circuit_class in CIRCUIT.keys():
    for head in CIRCUIT[circuit_class]:
        # if head[0] <= 6: continue # let's think about the effect before S2 ..
        heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)

# early_heads = [(layer, head_idx) for layer in list(range(6[])) for head_idx in list(range(12))]

model, abl = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    # exclude_heads=early_heads,
)
init_probs = probs(model, ioi_dataset)
print(f"{init_probs=}")

vprobs = torch.zeros(7, 12)

for layer in range(7):
    for head_idx in range(12):
        print(layer, head_idx)
        heads_to_keep = {}
        for circuit_class in ARTHUR_CIRCUIT.keys():
            for head in ARTHUR_CIRCUIT[circuit_class]:
                if head[0] <= 6:
                    continue
                heads_to_keep[head] = get_extracted_idx(
                    RELEVANT_TOKENS[head], ioi_dataset
                )
        new_early_heads = early_heads.copy()
        new_early_heads.remove((layer, head_idx))
        model, abl = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            exclude_heads=new_early_heads,
        )
        vprobs[layer][head_idx] = probs(model, ioi_dataset)
#%% # wait, do even the very first plots work for the IO probs metric??
vals = torch.zeros(12, 12)
from ioi_circuit_extraction import (
    ARTHUR_CIRCUIT,
    get_heads_circuit,
    get_mlps_circuit,
    CIRCUIT,
    do_circuit_extraction,
)
from ioi_utils import probs

old_probs = probs(model, ioi_dataset)

for layer in range(12):
    print(layer)
    for head in range(12):
        heads_to_keep = get_heads_circuit(ioi_dataset, excluded=[], circuit=CIRCUIT)
        torch.cuda.empty_cache()

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_remove={
                (layer, head): [
                    list(range(ioi_dataset.word_idx["end"][i] + 1)) for i in range(N)
                ]
            },
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            # exclude_heads=[(layer, head)],
        )
        torch.cuda.empty_cache()
        new_probs = probs(model, ioi_dataset)
        vals[layer, head] = new_probs - old_probs

show_pp(vals)
#%%
vals2 = torch.zeros(12)
for layer in range(12):
    print(layer)

    heads_to_keep = get_heads_circuit(ioi_dataset, excluded=[], circuit=CIRCUIT)
    torch.cuda.empty_cache()

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        mlps_to_remove={
            (layer): [list(range(ioi_dataset.word_idx["end"][i] + 1)) for i in range(N)]
        },
        heads_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    new_probs = probs(model, ioi_dataset)
    vals2[layer] = new_probs - old_probs
#%%
show_pp(vals2.unsqueeze(0), title="MLP removal")
#%%
# %%
show_scatter = True
circuit_perf_scatter = []
eps = 1.2

# by points
if show_scatter:
    fig = go.Figure()
    all_xs = []
    all_ys = []

    for i, circuit_class in enumerate(set(circuit_perf.removed_group)):
        xs = list(
            circuit_perf[circuit_perf["removed_group"] == circuit_class][
                "cur_metric_broken"
            ]
        )
        ys = list(
            circuit_perf[circuit_perf["removed_group"] == circuit_class][
                "cur_metric_cobble"
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                # hover_data=["sentence", "template"], # TODO get this working
                mode="markers",
                marker=dict(color=CLASS_COLORS[circuit_class], size=3),
                # name=circuit_vlass,
                showlegend=False,
                # color=CLASS_COLORS[circuit_class],
                # opacity=1.0,
            )
        )

        all_xs += xs
        all_ys += ys
        plot_ellipse(
            fig,
            xs,
            ys,
            color=CLASS_COLORS[circuit_class],
            name=circuit_class,
        )

    minx = min(min(all_xs), min(all_ys))
    maxx = max(max(all_xs), max(all_ys))
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="x",
                x0=minx,
                x1=maxx,
                yref="y",
                y0=minx,
                y1=maxx,
            )
        ]
    )

    xs = np.linspace(minx, maxx, 100)
    ys_max = xs + eps
    ys_min = xs - eps

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys_min,
            mode="lines",
            name="THIS ONE IS HIDDEN",
            showlegend=False,
            line=dict(color="grey"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys_max,
            mode="lines",
            name=f"Completeness region, epsilon={eps}",
            fill="tonexty",
            line=dict(color="grey"),
        )
    )

    fig.update_xaxes(gridcolor="black", gridwidth=0.1)
    fig.update_yaxes(gridcolor="black", gridwidth=0.1)
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    fig.write_image(f"svgs/circuit_completeness_at_{ctime()}.svg")
    fig.show()
#%%
MODEL_CFG = model.cfg
MODEL_EPS = model.cfg.eps


def layer_norm(x, cfg=MODEL_CFG):
    return LayerNormPre(cfg)(x)


def get_layer_norm_div(x, eps=MODEL_EPS):
    mean = x.mean(dim=-1, keepdim=True)
    new_x = (x - mean).detach().clone()
    return (new_x.var(dim=-1, keepdim=True).mean() + eps).sqrt()


#%%
# def qk_logit_lens(
#     model,
#     ioi_dataset,
#     heads,
#     show=["attn"],  # can add "mlp" to this
#     return_vals=False,
#     dir_mode="IO - S1",
#     title="",
#     return_figs=False,
# ):

heads = [(9, 9)]
return_vals = True
dir_mode = "IO - S1"
show = ["attn", "mlp"]
title = "asdof"

if True:
    """
    Jacob's QK logit lens
    ... this requires annoying think-about-contribution-through-layer-norm
    """

    if len(heads) != 1:
        raise NotImplementedError("only works for one head")
    nm_layer, nm_idx = heads[0]

    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    attn_vals = torch.zeros(size=(n_heads, n_layers))
    mlp_vals = torch.zeros(size=(n_layers,))

    wq = model.blocks[nm_layer].attn.W_Q[nm_idx]
    bias_q = model.blocks[nm_layer].attn.b_Q
    wk = model.blocks[nm_layer].attn.W_K[nm_idx]
    bias_k = model.blocks[nm_layer].attn.b_K
    assert wq.shape == wk.shape
    assert wq.shape == (64, 768)

    for i in tqdm(range(ioi_dataset.N)):
        cache = {}
        k_hook_name = f"blocks.{nm_layer}.attn.hook_k"
        pre_name = f"blocks.{nm_layer}.hook_resid_pre"
        ln_hook_name = f"blocks.{nm_layer}.ln1.hook_scale"
        model.cache_all(
            cache,
            lambda name: name in [pre_name, k_hook_name, ln_hook_name],
            device="cuda",
        )
        logits = model(ioi_dataset.text_prompts[i])
        iok = cache[k_hook_name][0, ioi_dataset.word_idx["IO"][i].item(), nm_idx, :]
        s1k = cache[k_hook_name][
            0, ioi_dataset.word_idx["S"][i].item(), nm_idx, :
        ]  # these already have the bias_K added
        if dir_mode == "IO - S1":
            k = iok - s1k
        else:
            raise NotImplementedError(dir_mode)
        assert len(k.shape) == 1

        neels_ln_scale = cache[ln_hook_name][:, -2, :]
        our_maunal_ln_scale = get_layer_norm_div(cache[pre_name][:, -2, :])
        assert torch.allclose(neels_ln_scale, our_maunal_ln_scale, atol=1e-3, rtol=1e-3)

        res_stream_sum = torch.zeros(size=(d_model,), device="cuda")
        res_stream_sum += cache["blocks.0.hook_resid_pre"][0, -2, :]
        # at the Q position

        for lay in range(0, 9):
            cur_attn = cache[f"blocks.{lay}.attn.hook_result"][0, -2, :, :]
            cur_mlp = cache[f"blocks.{lay}.hook_mlp_out"][:, -2, :][0]

            # TODO check we have res_stream_sum working
            res_stream_sum += torch.sum(cur_attn, dim=0)
            res_stream_sum += model.blocks[lay].attn.b_O  # .detach().cpu()
            res_stream_sum += cur_mlp
            assert torch.allclose(
                res_stream_sum,
                cache[f"blocks.{lay}.hook_resid_post"][0, -2, :].detach(),
                rtol=1e-3,
                atol=1e-3,
            ), lay

            cur_mlp -= cur_mlp.mean()
            for j in range(n_heads):
                cur_attn[j] -= cur_attn[j].mean()
                # we layer norm the end result of the residual stream,
                # (which firstly centres the residual stream)
                # so to estimate "amount written in the IO-S direction"
                # we centre each head's output
            cur_attn /= neels_ln_scale.item()
            cur_mlp /= neels_ln_scale.item()

            # print(cur_attn.shape) # (12, 768)
            att_q = torch.einsum("ab,cb->ac", cur_attn, wq)  # + bias_q
            # this is now 12 * 64
            mlp_q = torch.einsum("b,cb->c", cur_mlp, wq)  # + bias_q
            # this is now 64

            attn_vals[:n_heads, lay] += torch.einsum("ha,a->h", att_q.cpu(), k.cpu())
            mlp_vals[lay] = torch.einsum("a,a->", mlp_q.cpu(), k.cpu())

        res_stream_sum -= res_stream_sum.mean()
        res_stream_sum = (
            layer_norm(res_stream_sum.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        )
        cur_writing = torch.einsum("a,ca->c", res_stream_sum, wq) + bias_q[nm_idx]
        calculated_k = torch.einsum("a,a->", cur_writing.cpu(), k.cpu())
        calculated_k /= model.blocks[nm_layer].attn.attn_scale

        cached_score = (
            cache[f"blocks.{nm_layer}.attn.hook_attn_scores"][
                0, nm_idx, -2, ioi_dataset.word_idx["IO"][i].item()
            ]
            - cache[f"blocks.{nm_layer}.attn.hook_attn_scores"][
                0, nm_idx, -2, ioi_dataset.word_idx["S"][i].item()
            ]
        )
        assert torch.allclose(
            calculated_k,
            cached_score,
            rtol=1e-2,
            atol=1e-2,
        ), f"{i=} {calculated_k.shape=} {get_corner(cached_score, n=4)=} {get_corner(calculated_k, n=4)=}"

    attn_vals /= ioi_dataset.N
    mlp_vals /= ioi_dataset.N

    all_figs = []
    if "attn" in show:
        all_figs.append(
            show_pp(
                attn_vals.detach().cpu().numpy(),
                xlabel="head no",
                ylabel="layer no",
                title=title,
                return_fig=True,
            )
        )
    if "mlp" in show:
        all_figs.append(
            show_pp(
                mlp_vals.detach().numpy().unsqueeze(0).T.detach(),
                xlabel="",
                ylabel="layer no",
                title=title,
                return_fig=True,
            )
        )
    # if return_figs and return_vals:
    #     return all_figs, attn_vals, mlp_vals
    # if return_vals:
    #     return attn_vals, mlp_vals
    # if return_figs:
    #     return all_figs
#%%
def writing_direction_heatmap(
    model,
    ioi_dataset,
    show=["attn"],  # can add "mlp" to this
    return_vals=False,
    dir_mode="IO - S",
    unembed_mode="normal",
    title="",
    verbose=False,
    return_figs=False,
):
    """
    Plot the dot product between how much each attention head
    output with `IO-S`, the difference between the unembeds between
    the (correct) IO token and the incorrect S token
    """

    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    model_unembed = (
        model.unembed.W_U.detach().cpu()
    )  # note that for GPT2 embeddings and unembeddings are tides such that W_E = Transpose(W_U)

    unembed_bias = model.unembed.b_U.detach().cpu()

    attn_vals = torch.zeros(size=(n_heads, n_layers))
    mlp_vals = torch.zeros(size=(n_layers,))
    logit_diffs = logit_diff(model, ioi_dataset, all=True).cpu()

    for i in tqdm(range(ioi_dataset.N)):
        io_tok = ioi_dataset.toks[i][ioi_dataset.word_idx["IO"][i].item()]
        s_tok = ioi_dataset.toks[i][ioi_dataset.word_idx["S"][i].item()]
        io_dir = model_unembed[io_tok]
        s_dir = model_unembed[s_tok]
        unembed_bias_io = unembed_bias[io_tok]
        unembed_bias_s = unembed_bias[s_tok]
        if dir_mode == "IO - S":
            dire = io_dir - s_dir
        elif dir_mode == "IO":
            dire = io_dir
        elif dir_mode == "S":
            dire = s_dir
        else:
            raise NotImplementedError()
        dire.to("cuda")
        cache = {}
        model.cache_all(
            cache, device="cuda"
        )  # TODO maybe speed up by only caching relevant things
        logits = model(ioi_dataset.text_prompts[i])

        res_stream_sum = torch.zeros(
            size=(d_model,), device="cuda"
        )  # cuda implem to speed up things
        res_stream_sum += cache["blocks.0.hook_resid_pre"][0, -2, :]  # .detach().cpu()
        # the pos and token embeddings

        layer_norm_div = get_layer_norm_div(
            cache["blocks.11.hook_resid_post"][0, -2, :]
        )

        for lay in range(n_layers):
            cur_attn = (
                cache[f"blocks.{lay}.attn.hook_result"][0, -2, :, :]
                # + model.blocks[lay].attn.b_O.detach()  # / n_heads
            )
            cur_mlp = cache[f"blocks.{lay}.hook_mlp_out"][:, -2, :][0]

            # check that we're really extracting the right thing
            res_stream_sum += torch.sum(cur_attn, dim=0)
            res_stream_sum += model.blocks[lay].attn.b_O  # .detach().cpu()
            res_stream_sum += cur_mlp
            assert torch.allclose(
                res_stream_sum,
                cache[f"blocks.{lay}.hook_resid_post"][0, -2, :].detach(),
                rtol=1e-3,
                atol=1e-3,
            ), lay

            cur_mlp -= cur_mlp.mean()
            for i in range(n_heads):
                cur_attn[i] -= cur_attn[i].mean()
                # we layer norm the end result of the residual stream,
                # (which firstly centres the residual stream)
                # so to estimate "amount written in the IO-S direction"
                # we centre each head's output
            cur_attn /= layer_norm_div  # ... and then apply the layer norm division
            cur_mlp /= layer_norm_div

            attn_vals[:n_heads, lay] += torch.einsum(
                "ha,a->h", cur_attn.cpu(), dire.cpu()
            )
            mlp_vals[lay] = torch.einsum("a,a->", cur_mlp.cpu(), dire.cpu())

        res_stream_sum -= res_stream_sum.mean()
        res_stream_sum = (
            layer_norm(res_stream_sum.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        )
        cur_writing = (
            torch.einsum("a,a->", res_stream_sum, dire.to("cuda"))
            + unembed_bias_io
            - unembed_bias_s
        )

        assert i == 11 or torch.allclose(  # ??? and it's way off, too
            cur_writing,
            logit_diffs[i],
            rtol=1e-2,
            atol=1e-2,
        ), f"{i=} {cur_writing=} {logit_diffs[i]}"

    attn_vals /= ioi_dataset.N
    mlp_vals /= ioi_dataset.N
    all_figs = []
    if "attn" in show:
        all_figs.append(
            show_pp(
                attn_vals,
                xlabel="head no",
                ylabel="layer no",
                title=title,
                return_fig=True,
            )
        )
    if "mlp" in show:
        all_figs.append(
            show_pp(
                mlp_vals.unsqueeze(0).T,
                xlabel="",
                ylabel="layer no",
                title=title,
                return_fig=True,
            )
        )
    if return_figs and return_vals:
        return all_figs, attn_vals, mlp_vals
    if return_vals:
        return attn_vals, mlp_vals
    if return_figs:
        return all_figs


torch.cuda.empty_cache()

all_figs, attn_vals, mlp_vals = writing_direction_heatmap(
    model,
    ioi_dataset,
    return_vals=True,
    show=["attn", "mlp"],
    dir_mode="IO - S",
    title="Output into IO - S token unembedding direction",
    verbose=True,
    return_figs=True,
)
modules = ["attn", "mlp"]

for i, fig in enumerate(all_figs):
    fig.write_image(f"svgs/writing_direction_heatmap_module_{modules[i]}.svg")
#%% # Q: can we just replace S2 Inhibition Heads with 1.0 attention to S2?
# A: pretty much yes

from ioi_circuit_extraction import CIRCUIT
from ioi_utils import probs, logit_diff

circuit = CIRCUIT.copy()

heads_to_keep = get_heads_circuit(
    ioi_dataset,
    circuit=circuit,
)

heads = circuit["s2 inhibition"].copy()

for change in [False, True]:
    model.reset_hooks()

    model, abl = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )

    if change:
        for layer, head in heads:
            hook_name = f"blocks.{layer}.attn.hook_attn"

            def s2_ablation_hook(
                z, act, hook
            ):  # batch, seq, head dim, because get_act_hook hides scary things from us
                assert z.shape == act.shape, (z.shape, act.shape)
                z = act
                return z

            act = torch.zeros(
                size=(
                    ioi_dataset.N,
                    model.cfg.n_heads,
                    ioi_dataset.max_len,
                    ioi_dataset.max_len,
                )
            )
            act[
                torch.arange(ioi_dataset.N),
                :,
                ioi_dataset.word_idx["end"][: ioi_dataset.N],
                ioi_dataset.word_idx["S2"][: ioi_dataset.N],
            ] = 0.5
            cur_hook = get_act_hook(
                s2_ablation_hook,
                alt_act=act,
                idx=head,
                dim=1,
            )
            model.add_hook(hook_name, cur_hook)

    io_probs = probs(model, ioi_dataset)
    print(f" {logit_diff(model, ioi_dataset)}, {io_probs=}")
#%% evidence for the S2 story
# ablating V for everywhere except S2 barely affects LD. But ablating all V has LD go to almost 0

model.reset_hooks()
model, abl = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)

for layer, head_idx in [(7, 9), (8, 6), (7, 3), (8, 10)]:
    # break
    cur_tensor_name = f"blocks.{layer}.attn.hook_q"
    s2_token_idxs = get_extracted_idx(["S2"], ioi_dataset)
    mean_cached_values = (
        abl.mean_cache[cur_tensor_name][:, :, head_idx, :].cpu().detach()
    )

    def s2_v_ablation_hook(
        z, act, hook
    ):  # batch, seq, head dim, because get_act_hook hides scary things from us
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_q", hook.name
        assert len(list(z.shape)) == 3, z.shape
        assert list(z.shape) == list(act.shape), (z.shape, act.shape)

        z = (
            mean_cached_values.cuda()
        )  # hope that we don't see chaning values of mean_cached_values...
        return z

    cur_hook = get_act_hook(
        s2_v_ablation_hook,
        alt_act=abl.mean_cache[cur_tensor_name],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(cur_tensor_name, cur_hook)

new_ld = logit_diff(model, ioi_dataset)
new_probs = probs(model, ioi_dataset)
print(f"{new_ld=}, {new_probs=}")
#%% # new shit: attention probs on S2 is the score
heads_to_patch = circuit["s2 inhibition"].copy()
attn_circuit_template = "blocks.{patch_layer}.attn.hook_v"
cache_names = set(
    [
        attn_circuit_template.format(patch_layer=patch_layer)
        for patch_layer, _ in heads_to_patch
    ]
)

logit_diffs = torch.zeros(size=(12, 12))
mlps = torch.zeros(size=(12,))
model.reset_hooks()

S2_LAYER = 7
S2_HEAD = 9
metric = partial(attention_on_token, head_idx=S2_HEAD, layer=S2_LAYER, token="S2")
metric = partial(attention_on_token, head_idx=9, layer=9, token="IO")

model.reset_hooks()
base_metric = metric(model, ioi_dataset)
print(f"{base_metric=}")

experiment_metric = ExperimentMetric(
    metric=metric, dataset=ioi_dataset, relative_metric=False
)
config = AblationConfig(
    abl_type="random",
    mean_dataset=abca_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    nb_metric_iteration=1,
    max_seq_len=ioi_dataset.max_len,
)
abl = EasyAblation(
    model, config, experiment_metric
)  # , mean_by_groups=True, groups=ioi_dataset.groups)
e()
result = abl.run_experiment()
show_pp(
    (result - base_metric).T,
    xlabel="head",
    ylabel="layer",
    title="Attention on S2 change when ablated",
)
#%%
for layer in range(12):
    for head_idx in [None] + list(range(12)):
        # do a run where we ablate (head, layer)

        if head_idx is None:
            cur_tensor_name = f"blocks.{layer}.hook_mlp_out"
            mean_cached_values = abl.mean_cache[cur_tensor_name].cpu().detach()

        else:
            cur_tensor_name = f"blocks.{layer}.attn.hook_result"
            mean_cached_values = (
                abl.mean_cache[cur_tensor_name][:, :, head_idx, :].cpu().detach()
            )

        def ablation_hook(z, act, hook):
            # batch, seq, head dim, because get_act_hook hides scary things from us
            cur_layer = int(hook.name.split(".")[1])
            cur_head_idx = hook.ctx["idx"]

            assert hook.name == cur_tensor_name, hook.name
            assert len(list(z.shape)) == 3, z.shape
            assert list(z.shape) == list(act.shape), (z.shape, act.shape)

            z = (
                mean_cached_values.cuda()
            )  # hope that we don't see changing values of mean_cached_values...
            return z

        cur_hook = get_act_hook(
            ablation_hook,
            alt_act=abl.mean_cache[cur_tensor_name],
            idx=head_idx,  # nice deals with
            dim=2 if head_idx is not None else None,  # None for MLPs
        )
        model.reset_hooks()
        model.add_hook(cur_tensor_name, cur_hook)
        cache = {}

        model.cache_some(cache, lambda x: x in cache_names)
        torch.cuda.empty_cache()
        metric(model, ioi_dataset)
        # all_cached[(layer, head_idx)] = cache[f"blocks.{S2_HEAD}.attn.hook_q"].cpu().detach()

        model.reset_hooks()
        for patch_layer, patch_head in heads_to_patch:

            def patch_in_q(z, act, hook):
                # assert hook.name == f"blocks.{patch_head}.attn.hook_q", hook.name # OOH ERR, is commenting this out ok???
                assert len(list(z.shape)) == 3, z.shape
                assert list(z.shape) == list(act.shape), (z.shape, act.shape)
                z = act.cuda()
                return z

            s2_hook = get_act_hook(
                patch_in_q,
                alt_act=cache[attn_circuit_template.format(patch_layer=patch_layer)],
                idx=patch_head,
                dim=2,
            )
            model.add_hook(
                attn_circuit_template.format(patch_layer=patch_layer), s2_hook
            )

        ld = metric(model, ioi_dataset)

        if head_idx is None:
            mlps[layer] = ld.detach().cpu()
        else:
            logit_diffs[layer, head_idx] = ld.detach().cpu()

        print(f"{layer=}, {head_idx=}, {ld=}")

att_heads_mean_diff = logit_diffs - base_metric
show_pp(
    att_heads_mean_diff.T,
    ylabel="layer",
    xlabel="head",
    title=f"Change in logit diff: {torch.sum(att_heads_mean_diff)}",
)
mlps_mean_diff = mlps - base_ld
show_pp(
    mlps_mean_diff.T.unsqueeze(0),
    ylabel="layer",
    xlabel="head",
    title=f"Change in logit diff, MLPs: {torch.sum(mlps_mean_diff)}",
)
#%% # new experiment idea: the duplicators and induction heads shouldn't care where their attention is going, provided that
# it goes to either S or S+1.

for j in range(2, 5):
    # [batch, head_index, query_pos, key_pos] # so pass dim=1 to ignore the head
    def attention_pattern_modifier(z, hook):
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_attn", hook.name
        assert len(list(z.shape)) == 3, z.shape
        # batch, seq (attending_query), attending_key

        # cur = z[torch.arange(ioi_dataset.N), s2_positions, s_positions+1]
        # print(cur)
        # print(f"{cur.shape=}")
        # some_atts = torch.argmax(cur, dim=1)
        # for i in range(20):
        # print(i, model.tokenizer.decode(ioi_dataset.toks[i][some_atts[i]]), ":", model.tokenizer.decode(ioi_dataset.toks[i][:6]))
        # print(some_atts.shape)

        # prior_stuff = []
        # for i in range(0, 2):
        #     prior_stuff.append(z[torch.arange(ioi_dataset.N), s2_positions, s_positions + i].clone())
        # for i in range(0, 2):
        #     z[torch.arange(ioi_dataset.N), s2_positions, s_positions + i] =  prior_stuff[(i + j) % 2] # +1 is the do nothing one # ([0, 1][(i+j)%2]) is way beyond scope

        # z[torch.arange(ioi_dataset.N), s2_positions, 0] = prior_stuff[(0 + j) % 2]
        # z[torch.arange(ioi_dataset.N), s2_positions, s_positions] = prior_stuff[(1 + j) % 2]

        z[torch.arange(ioi_dataset.N), s2_positions, :] = 0

        for key in ioi_dataset.word_idx.keys():
            z[
                torch.arange(ioi_dataset.N), s2_positions, ioi_dataset.word_idx[key]
            ] = average_attention[(cur_layer, cur_head_idx)][key]

        return z

    F = logit_diff  # or logit diff
    model.reset_hooks()
    ld = F(model, ioi_dataset)

    circuit_classes = ["s2 inhibition"]

    for circuit_class in circuit_classes:
        for layer, head_idx in circuit[circuit_class]:
            cur_hook = get_act_hook(
                attention_pattern_modifier,
                alt_act=None,
                idx=head_idx,
                dim=1,
            )
            model.add_hook(f"blocks.{layer}.attn.hook_attn", cur_hook)

    ld2 = F(model, ioi_dataset)
    print(
        f"Initially there's a logit difference of {ld}, and after permuting by {j-1}, the new logit difference is {ld2=}"
    )
#%%
heads_to_patch = (
    circuit["duplicate token"] + circuit["induction"]
)  # circuit["s2 inhibition"]
layers = list(set([layer for layer, head_idx in heads_to_patch]))
hook_names = [f"blocks.{l}.attn.hook_result" for l in layers]
model.reset_hooks()
cache_s2 = {}
model.cache_some(cache_s2, lambda x: x in hook_names)
logits = model(adea_dataset.text_prompts).detach()

patch_last_tokens = partial(patch_positions, positions=["end"])
patch_s2_token = partial(patch_positions, positions=["S2"])

for layer, head_idx in heads_to_patch:
    cur_hook = get_act_hook(
        patch_s2_token,
        alt_act=cache_s2["blocks.{}.attn.hook_result".format(layer)],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(f"blocks.{layer}.attn.hook_result", cur_hook)

l = logit_diff(model, ioi_dataset)
print(f"{l=}")
model.reset_hooks()
#%%
ys = []
average_attention = {}

for idx, dataset in enumerate([ioi_dataset, abca_dataset]):
    fig = go.Figure()
    print(idx, ["ioi", "abca"][idx])
    for heads_raw in circuit["name mover"][
        :3
    ]:  # heads_to_patch: # [(9, 9), (9, 6), (10, 0)]:
        heads = [heads_raw]
        average_attention[heads_raw] = {}
        cur_ys = []
        cur_stds = []
        att = torch.zeros(size=(dataset.N, dataset.max_len, dataset.max_len))
        for head in tqdm(heads):
            att += show_attention_patterns(
                model, [head], dataset, return_mtx=True, mode="scores"
            )
        att /= len(heads)

        vals = att[torch.arange(dataset.N), ioi_dataset.word_idx["end"][: dataset.N], :]
        evals = torch.exp(vals)
        val_sum = torch.sum(evals, dim=1)
        assert val_sum.shape == (dataset.N,), val_sum.shape
        print(f"{heads=} {val_sum.mean()=}")

        for key in ioi_dataset.word_idx.keys():
            end_to_s2 = att[
                torch.arange(dataset.N),
                ioi_dataset.word_idx["end"][: dataset.N],
                ioi_dataset.word_idx[key][: dataset.N],
            ]
            # ABCA dataset calculates S2 in trash way... so we use the IOI dataset indices
            cur_ys.append(end_to_s2.mean().item())
            cur_stds.append(end_to_s2.std().item())
            average_attention[heads_raw][key] = end_to_s2.mean().item()
        fig.add_trace(
            go.Bar(
                x=list(ioi_dataset.word_idx.keys()),
                y=cur_ys,
                error_y=dict(type="data", array=cur_stds),
                name=str(heads_raw),
            )
        )  # ["IOI", "ABCA"][idx]))

        fig.update_layout(title_text="Attention scores; from END to S2")
    fig.show()
#%% [markdown] implement Jacob's QK logit lens

# save K_{IO} and K_{S}
# look at what writes in the

#%%
heads_to_measure = [(9, 6), (9, 9), (10, 0)]  # name movers
heads_by_layer = {9: [6, 9], 10: [0]}
layers = [9, 10]
hook_names = [f"blocks.{l}.attn.hook_attn" for l in layers]

# warnings.warn("Actually doing S Inhib stuff")
# heads_to_measure = circuit["s2 inhibition"]
# layers = [7, 8]
# heads_by_layer = {7: [3, 9], 8: [6, 10]}
# hook_names = [f"blocks.{l}.attn.hook_attn" for l in layers]

model.reset_hooks()
cache_baseline = {}
model.cache_some(
    cache_baseline, lambda x: x in hook_names
)  # we only cache the activation we're interested
logits = model(ioi_dataset.text_prompts).detach()
# %% [markdown] Q: is it possible to patch in ACCA sentences to make things work? A: Yes!
def patch_positions(
    z, source_act, hook, positions=["END"]
):  # we patch at the "to" token
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z


patch_last_tokens = partial(patch_positions, positions=["end"])
patch_s2 = partial(patch_positions, positions=["S2"])
#%%  actually answered here...
config = PatchingConfig(
    source_dataset=acca_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",  # we patch "result", the result of the attention head
    cache_act=True,
    verbose=False,
    patch_fn=patch_s2,
    layers=(0, max(layers) - 1),
)  # we stop at layer "LAYER" because it's useless to patch after layer 9 if what we measure is attention of a head at layer 9.
metric = ExperimentMetric(
    partial(attention_probs, scale=False),
    config.target_dataset,
    relative_metric=False,
    scalar_metric=False,
)
patching = EasyPatching(model, config, metric)
add_these_hooks = []  # actually, let's not add the fixed S2 Inhib thing
patching.other_hooks = add_these_hooks
result = patching.run_patching()

if config.target_module == "mlp":
    assert result.shape[1] == 3
    result[0, :] = 0

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :] if config.target_module == "mlp" else result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Variation in attention probs of Head {str(heads_to_measure)} from token "to" to {key} after Patching ABC->ABB on "to"',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )
    fig.write_image(
        f"svgs/variation_average_nm_attn_prob_key_{key}_patching_ABC_END.svg"
    )
    fig.show()
#%% [markdown]
# This was: okay, so is ACCA identical for induction heads??? A: yes, and for dupes too
# Now is: Try the ACBA dataset and see what happens
e()
relevant_heads = {}

for head in circuit["duplicate token"] + circuit["induction"]:
    relevant_heads[head] = "S2"

circuit = deepcopy(CIRCUIT)
relevant_hook_names = set(
    [f"blocks.{layer}.attn.hook_result" for layer, _ in relevant_heads.keys()]
)

if "alt_cache" not in dir() and False:
    alt_cache = {}
    model.reset_hooks()
    model.cache_some(alt_cache, names=lambda name: name in relevant_hook_names)
    logits = model(bcca_dataset.text_prompts)
    del logits
    elayers = ()

ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
oii_dataset = ioi_dataset.gen_flipped_prompts(("IO", "S1"))
arthur_alex_clash = ioi_dataset.gen_flipped_prompts(("S2", "IO")).gen_flipped_prompts(
    ("IO", "S1")
)
total_reversal_dataset = ioi_dataset.gen_flipped_prompts(("S2", "IO"))

early_dataset = IOIDataset(prompt_type=BABA_EARLY_IOS, N=100)
late_dataset = IOIDataset.construct_from_ioi_prompts_metadata(
    templates=BABA_LATE_IOS, ioi_prompts_data=dearly.ioi_prompts, N=100
)

config = PatchingConfig(
    source_dataset=late_dataset.text_prompts,
    target_dataset=early_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=patch_s2,
    layers=(0, max(layers) - 1),
)  # we stop at layer "LAYER" because it's useless to patch after layer 9 if what we measure is attention of a head at layer 9.
metric = ExperimentMetric(
    partial(attention_probs, scale=False),
    config.target_dataset,
    relative_metric=False,
    scalar_metric=False,
)
patching = EasyPatching(model, config, metric)
#%%
model.reset_hooks()
# new_heads_to_keep = get_heads_circuit(ioi_dataset, circuit=circuit)
# model, _ = do_circuit_extraction(
#     model=model,
#     heads_to_keep=new_heads_to_keep,
#     mlps_to_remove={},
#     ioi_dataset=ioi_dataset,
#     mean_dataset=abca_dataset,
# )

for idx, head_set in enumerate(
    [
        [],
        ["duplicate token"],
        ["induction"],
        ["s2 inhibition"],
        ["induction", "duplicate token"],
        ["s2 inhibition"] + ["induction"] + ["duplicate token"],
    ]
):
    model.reset_hooks()
    heads = []
    for circuit_class in head_set:
        heads += circuit[circuit_class]
    for layer, head_idx in heads:
        hook = patching.get_hook(
            layer,
            head_idx,
            manual_patch_fn=partial(
                patch_positions, positions=RELEVANT_TOKENS[(layer, head_idx)]
            ),
        )
        model.add_hook(*hook)

    att_probs = attention_probs(
        model, ioi_dataset.text_prompts, variation=False, scale=False
    )
    print(f"{head_set=}, IO S S2, {att_probs=}")  # print("IO S S2")
    cur_logit_diff = logit_diff(model, ioi_dataset)
    cur_io_probs = probs(model, ioi_dataset)
    print(f"{idx=} {cur_logit_diff=} {cur_io_probs=}")
#%%
# some [logit difference, IO probs] for the different modes

model_results = {"x": 3.8212, "y": 0.5281, "name": "model"}  # saved from prompts2.py
duplicate_results = {"x": 3.0485, "y": 0.4755, "name": "duplicate"}
duplicate_and_induction_results = {
    "x": -0.66,
    "y": 0.1517,
    "name": "duplicate and induction",
}
induction_results = {"x": 0.459, "y": 0.2498, "name": "induction"}
inhibition_results = {"x": -0.82, "y": 0.1367, "name": "inhibition"}

# model_results = {"x":3.5492, "y":0.4955, "name":"Model"} # hopefully saved at prompts.py
# circuit_results = {"x":3.3414, "y":0.2854, "name":"Circuit", "textposition":"top left"}
# hooked_induction_dupe_inhibition_results = {"x":2.8315, "y":0.2901, "name":"Induction, Duplication, Inhibition Heads hooked on ACC", "textposition":"bottom left"}
# hooked_induction_dupe_results = {"x":3.3488, "y":0.2779, "name":"Induction and Duplication Heads hooked on ACC", "textposition":"bottom right"}
# previous_token_results = {"x":1.2908, "y":0.1528, "name":"Previous Token Heads hooked on ACC"}
# abc_duplicate_induction_results = {"x":0.0675, "y":0.0708, "name":"Duplicate and Induction Heads hooked on ABC (S2)", "textposition":"bottom right"}
# acb_duplicate_induction_results = {"x":0.1685, "y":0.0727, "name":"Duplicate and Induction Heads hooked on ABC (S1)", "textposition":"top right"}

xs = []
ys = []
names = []
textpositions = []

for var in dir():
    if var.endswith("_results"):
        xs.append(globals()[var]["x"])
        ys.append(globals()[var]["y"])
        names.append(globals()[var]["name"])
        textpositions.append(globals()[var].get("textposition", "top center"))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=names,
        textposition=textpositions,
        # textposition=[["top left", "bottom left"][i%2] for i in range(len(xs))],
    )
)

# set x scaling
fig.update_xaxes(range=[-1, 5])

fig.update_layout(
    title="Effects of various patching experiments on circuit behaviour",
    xaxis_title="Logit Difference",
    yaxis_title="IO Probability",
)
fig.write_image(f"svgs/new_signal_plots_at_{ctime()}.svg")
fig.show()
#%% # test the hypothesis that MLP5 implements "if copy signal present then write S2 real hard"
# get logits we unembed right after Layer5 RESULTS: in 86/100 examples, yeah MLP5 increases S2 probs


def zero_hook(z, hook):
    z[:] = 0.0
    return z


ls = []

for zero_mlp in [True, False]:
    e()
    model.reset_hooks()

    for layer in range(5, 12):
        if layer > 5:
            cur_hook = get_act_hook(
                zero_hook,
                alt_act=None,
                idx=None,
                dim=None,
            )
            model.add_hook("blocks.{}.attn.hook_result".format(layer), cur_hook)

        if layer > 5 or not zero_mlp:
            cur_hook = get_act_hook(
                zero_hook,
                alt_act=None,
                idx=None,
                dim=None,
            )
            model.add_hook("blocks.{}.hook_mlp_out".format(layer), cur_hook)

    logits = model(ioi_dataset.text_prompts)
    ls.append(logits[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S2"], :])
#%% [markdown] let's try to remove MO MLPS from the circuit
circuit = deepcopy(CIRCUIT)
heads_to_keep = get_heads_circuit(ioi_dataset, circuit=circuit)
mlps_to_keep = get_mlps_circuit(ioi_dataset, mlps=[0, 1, 2, 3, 4, 5, 10, 11])
e()
model.reset_hooks()
# model, _ = do_circuit_extraction(
#     model=model,
#     heads_to_keep=heads_to_keep,
#     mlps_to_remove={},
#     # mlps_to_keep=mlps_to_keep,
#     ioi_dataset=ioi_dataset,
#     mean_dataset=abca_dataset,
# )

BAC_dataset = IOIDataset("BAC", N, model.tokenizer)
mixed_dataset = IOIDataset("ABC mixed", N, model.tokenizer)

for dataset in [ioi_dataset, ABC_dataset, BAC_dataset, mixed_dataset]:
    circuit_logit_diff = logit_diff(model, dataset)
    circuit_probs = probs(model, dataset)
    print(f"{circuit_logit_diff=} {circuit_probs=}")

logit_diff_min = 2.8
io_probs_min = 0.23
#%%
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
from time import ctime

study = optuna.create_study(
    study_name=f"Check by heads and index @ {ctime()}", storage=storage_name
)  # ADD!
#%%
e()
heads_to_keep = get_heads_circuit(ioi_dataset, circuit=circuit)
relevant_stuff = [(10, "end"), (11, "end")]
for layer in range(0, 6):
    for token in ["IO", "S2", "S", "end", "S+1"]:
        relevant_stuff.append((layer, token))


def objective(trial, manual=None):
    """
    BLAH
    """

    e()

    if manual is None:
        total_things = trial.suggest_int("total_things", 0, 6 * 5 + 2)
        cur_stuff = []
        for i in range(total_things):
            cur_stuff.append(
                trial.suggest_categorical("idx_{}".format(i), relevant_stuff)
            )
    else:
        cur_stuff = manual

    mlps_to_keep = {i: [] for i in range(12)}
    for i, pos in relevant_stuff:
        if (i, pos) in cur_stuff and pos not in mlps_to_keep[i]:
            mlps_to_keep[i].append(pos)
    mlps_to_keep = get_mlps_circuit(ioi_dataset, mlps=mlps_to_keep)
    new_model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_keep=mlps_to_keep,
        ioi_dataset=ioi_dataset,
        mean_dataset=abca_dataset,
    )
    circuit_logit_diff = logit_diff(new_model, ioi_dataset)
    circuit_probs = probs(new_model, ioi_dataset)
    print(f"{circuit_logit_diff=} {circuit_probs=}")
    ans = -circuit_logit_diff
    # if circuit_logit_diff > logit_diff_min and circuit_probs > io_probs_min:
    #     ans += 100000 - 100 * total_things
    return ans


# study.optimize(objective, n_trials=1e8)
#%%
tuples = [
    (0, "IO"),
    (0, "S2"),
    (0, "S"),
    (0, "S+1"),
    (0, "end"),
    (1, "IO"),
    (1, "S"),
    (1, "S2"),
    (2, "IO"),
    (2, "S2"),
    (3, "IO"),
    (3, "S"),
    (3, "S+1"),
    (3, "S2"),
    (4, "IO"),
    (4, "S"),
    (4, "S+1"),
    (4, "S2"),
    (5, "S"),
    (5, "S+1"),
    (5, "S2"),
    (10, "end"),
    (11, "end"),
]

ans = objective(None, tuples)
print(f"{ans=}")
#%% [markdown] Patch and freeze !!!
# do a run where we mean ablate IO position and S position
# do a run where we patch at S2 and patch the name movers' K with these things

# Activation at hook blocks.1.attn.hook_q has shape:
# torch.Size([4, 50, 12, 64])

safe_del("logits")
safe_del("embed_cache")
safe_del("k_cache")
safe_del("_")
e()

use_circuit = False
modes = ["IO"]
warnings.warn(f"{use_circuit=}")


def patch_all(z, source_act, hook):
    z[:] = source_act.clone()
    return z


def patch_positions(z, source_act, hook, positions=["end"], all_same=False):
    for pos in positions:
        if all_same:
            z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act
        else:
            z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
                torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
            ]
    return z


heads_to_keep = get_heads_circuit(ioi_dataset, circuit=circuit)

# embeddings mean
model.reset_hooks()
if use_circuit:
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=abca_dataset,
    )
embed_cache = {}
model.cache_some(embed_cache, lambda name: name == "hook_embed")
logits = model(ioi_dataset.text_prompts)
# hook_embed has shape 4x50x768
io_embed_mean = embed_cache["hook_embed"][
    torch.arange(ioi_dataset.N), ioi_dataset.word_idx["IO"]
].mean(dim=0, keepdim=True)
s_embed_mean = embed_cache["hook_embed"][
    torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S"]
].mean(dim=0, keepdim=True)

# middle thing
model.reset_hooks()
if use_circuit:
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=abca_dataset,
    )
safe_del("_")
e()
k_cache = {}
ks_names = [f"blocks.{layer}.attn.hook_k" for layer, _ in circuit["name mover"]]
model.cache_some(k_cache, lambda name: name in ks_names)
for mode in modes:
    cur_hook = get_act_hook(
        partial(patch_positions, positions=[mode], all_same=True),
        alt_act=s_embed_mean if mode == "S" else io_embed_mean,
        idx=None,
        dim=None,
    )
    model.add_hook("hook_embed", cur_hook)
logits = model(ioi_dataset.text_prompts)

# Activation at hook blocks.0.attn.hook_k has shape:
# torch.Size([4, 50, 12, 64])
model.reset_hooks()
if use_circuit:
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=abca_dataset,
    )
safe_del("_")
e()
for layer, head_idx in circuit["name mover"]:
    for mode in modes:
        cur_hook = get_act_hook(
            partial(patch_positions, positions=["S" if mode == "S" else "IO"]),
            alt_act=k_cache[f"blocks.{layer}.attn.hook_k"],
            idx=head_idx,
            dim=2,
        )
        model.add_hook(f"blocks.{layer}.attn.hook_k", cur_hook)

# model.reset_hooks()
io_probs = probs(model, ioi_dataset)
att_probs = attention_probs(
    model, ioi_dataset.text_prompts, variation=False, scale=False
)
print(f"IO S S2, {att_probs=}")
print(f" {logit_diff(model, ioi_dataset)}, {io_probs=}")
#%%
ds = []
all_templates = list(set(BABA_EARLY_IOS + BABA_LATE_IOS + BABA_TEMPLATES))
# THIS IS BABA ONLY!

for i, template in enumerate(all_templates):
    print(f"{i=} {template=}")
    d = IOIDataset(N=1, prompt_type=[template])
    ds.append(d)
#%%
templates_by_dis = [[] for _ in range(20)]
templates_by_sidx = [[] for _ in range(20)]

for i in range(len(ds)):
    dis = ds[i].word_idx["S2"].item() - ds[i].word_idx["S"].item()
    templates_by_dis[dis].append(all_templates[i])
#%%
distances = []

all_head_sets = all_subsets(
    ["s2 inhibition", "induction", "duplicate token", "previous token"]
)
results = {}

for source_dataset in [totally_diff_dataset]:
    layers = [7, 8]
    config = PatchingConfig(
        source_dataset=source_dataset.text_prompts,
        target_dataset=ioi_dataset.text_prompts,
        target_module="attn_head",
        head_circuit="result",
        cache_act=True,
        verbose=False,
        patch_fn=partial(patch_positions, positions=["S+1"]),
        layers=(0, max(layers) - 1),
    )
    metric = ExperimentMetric(
        partial(attention_probs, scale=False),
        config.target_dataset,
        relative_metric=False,
        scalar_metric=False,
    )
    patching = EasyPatching(model, config, metric)
    model.reset_hooks()

    def patch_positions(z, source_act, hook, positions=["end"], all_same=False):
        for pos in positions:
            if all_same:
                z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act
            else:
                z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
                    torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
                ]
        return z

    for head_set in tqdm(all_head_sets):
        model.reset_hooks()
        heads = []
        for circuit_class in head_set:
            heads += circuit[circuit_class]
        for layer, head_idx in heads:
            hook = patching.get_hook(
                layer,
                head_idx,
                manual_patch_fn=partial(
                    patch_positions, positions=RELEVANT_TOKENS[(layer, head_idx)]
                ),
            )
            model.add_hook(*hook)

        cur_logit_diff = logit_diff(model, ioi_dataset)
        cur_io_probs = probs(model, ioi_dataset)
        print(f"{head_set=} {cur_logit_diff}, {cur_io_probs=}")
        # results[(i, j, idx)] = cur_logit_diff, cur_io_probs
#%%
for all_head_sets_idx, head_set in enumerate(all_head_sets):
    initials = []
    DCCs = []
    CDCs = []

    title = str(head_set)

    for i in range(len(distances)):
        idx = distances[i]
        initials.append(results[(idx, 0, 0)][0])
        DCCs.append(results[(idx, 0, all_head_sets_idx)][0])
        CDCs.append(results[(idx, 1, all_head_sets_idx)][0])
    x1s = torch.tensor([1, 2, 3])
    x2s = x1s - 0.5
    x3s = -x1s

    # plot a bar chart of the results
    fig = go.Figure()
    fig.add_trace(go.Bar(x=distances, y=initials, name="initials"))
    fig.add_trace(go.Bar(x=distances, y=DCCs, name="DCCs"))
    fig.add_trace(go.Bar(x=distances, y=CDCs, name="CDCs"))
    fig.update_layout(barmode="group")

    # update x axis
    fig.update_xaxes(title_text="Distance between S and S2")
    # update title
    fig.update_layout(title_text=f"IOI Logit Diff by Distance, {title}")

    fig.show()
#%% [markdown] Hmm, so do the induction head outputs care about what's written into S+1 by previous token heads, or not?

# ablate all of the Previous Token Heads
# save the activations for the V matrix of the induction heads when we remove the previous token heads
# (AAAAAA) this could retain performance
# I guess the comparison is with not ablating the Previous Token Heads
# YEE, borked for now

# save the activations of induction V matrix, when previous tokens are ablated
# ablated = from RANDOM other thing or just 0??? NO ZERO ABLATION IS VERY BAD!!!

layers = [7, 8]
config = PatchingConfig(
    source_dataset=totally_diff_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=partial(patch_positions, positions=["S+1"]),
    layers=(0, max(layers) - 1),
)
metric = ExperimentMetric(
    partial(attention_probs, scale=False),
    config.target_dataset,
    relative_metric=False,
    scalar_metric=False,
)
patching = EasyPatching(model, config, metric)
model.reset_hooks()


def patch_positions(z, source_act, hook, positions=["end"], all_same=False):
    for pos in positions:
        if all_same:
            z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act
        else:
            z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
                torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
            ]
    return z


def turn_to_zero(z, source_act, hook):
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S+1"]] = 0
    return z


model.reset_hooks()
for layer, head_idx in circuit["previous token"]:
    hook = patching.get_hook(
        layer,
        head_idx,
        # manual_patch_fn=turn_to_zero,
        manual_patch_fn=partial(
            patch_positions,
            positions=["S+1"],  # RELEVANT_TOKENS[(layer, head_idx)]
        ),
    )
    model.add_hook(*hook)

# Activation at hook blocks.3.attn.hook_v has shape:
# torch.Size([4, 50, 12, 64])
cache = {}
hook_template = "blocks.{}.attn.hook_k"
hook_names = list(set(hook_template.format(layer) for layer, _ in circuit["induction"]))
model.cache_some(cache, lambda name: name in hook_names)
cur_logit_diff = logit_diff(model, ioi_dataset)
print(f"{cur_logit_diff=}")

# def patch_new_v(z, source_act, hook):
#     z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S+1"]] = source_act[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S+1"]]
#     return z

model.reset_hooks()
for layer, head_idx in circuit["induction"]:
    cur_tensor_name = hook_template.format(layer)
    cur_hook = get_act_hook(
        partial(patch_positions, positions=["S"]),
        alt_act=cache[cur_tensor_name],
        idx=head_idx,
        dim=2 if head_idx is not None else None,
    )
    model.add_hook(cur_tensor_name, cur_hook)

# model.reset_hooks()
cur_logit_diff = logit_diff(model, ioi_dataset)
cur_io_probs = probs(model, ioi_dataset)
print(f"{cur_logit_diff}, {cur_io_probs=}")
#%% [markdown] -> S2 Inhibition: Q versus K composition
# save the Inhibition K scores, when we ablate all the Induction and Duplicate token heads
names = [
    "ioi_dataset",
    "abca_dataset",
    "dcc_dataset",
    "acca_dataset",
    "acba_dataset",
    "all_diff_dataset",
    "totally_diff_dataset",
]
results = [[] for _ in names]

for i, hook_templates in enumerate(
    [
        ["blocks.{}.attn.hook_k"],
        ["blocks.{}.attn.hook_v"],
        ["blocks.{}.attn.hook_v", "blocks.{}.attn.hook_k"],
    ]
):
    for dataset_name in names:
        model.reset_hooks()
        dataset = eval(dataset_name)
        config = PatchingConfig(
            source_dataset=dataset.text_prompts,
            target_dataset=ioi_dataset.text_prompts,
            target_module="attn_head",
            head_circuit="result",
            cache_act=True,
            verbose=False,
            patch_fn=partial(
                patch_positions, positions=["S2" if hook_template[-1] != "q" else "end"]
            ),
            layers=(0, max(layers) - 1),
        )
        metric = ExperimentMetric(
            partial(attention_probs, scale=False),
            config.target_dataset,
            relative_metric=False,
            scalar_metric=False,
        )
        patching = EasyPatching(model, config, metric)
        model.reset_hooks()

        def patch_positions(z, source_act, hook, positions=["end"], all_same=False):
            for pos in positions:
                if all_same:
                    z[
                        torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
                    ] = source_act
                else:
                    z[
                        torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
                    ] = source_act[
                        torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
                    ]
            return z

        model.reset_hooks()
        for layer, head_idx in circuit["induction"] + circuit["duplicate token"]:
            hook = patching.get_hook(
                layer,
                head_idx,
                manual_patch_fn=partial(
                    patch_positions,
                    positions=["S2" if hook_template[-1] != "q" else "end"],
                ),
            )
            model.add_hook(*hook)
        hook_names = list(
            set(
                hook_template.format(layer)
                for layer, _ in circuit["s2 inhibition"]
                for hook_template in hook_templates
            )
        )
        raw_cache = {}
        model.cache_some(raw_cache, lambda name: name in hook_names)
        cur_logit_diff = logit_diff(model, dataset)
        print(f"{cur_logit_diff=}")
        cache = deepcopy(raw_cache)
        cur_logit_diff = logit_diff(model, ioi_dataset)
        print(f"Actually on IOI: {cur_logit_diff=}")

        model.reset_hooks()
        for layer, head_idx in circuit["s2 inhibition"]:
            for hook_template in hook_templates:
                cur_tensor_name = hook_template.format(layer)
                cur_hook = get_act_hook(
                    partial(patch_positions, positions=["S2"]),
                    alt_act=cache[cur_tensor_name],
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                )
                model.add_hook(cur_tensor_name, cur_hook)

        # model.reset_hooks()
        cur_logit_diff = logit_diff(model, ioi_dataset)
        cur_io_probs = probs(model, ioi_dataset)
        print(f"{cur_logit_diff}, {cur_io_probs=}")
        results[i].append(cur_logit_diff)

fig = go.Figure()
for i in range(3):
    fig.add_trace(
        go.Bar(
            x=names,
            y=results[i],
            name=["K", "V", "V+K"][i],
        )
    )

fig.update_layout(
    title="S2 Inhibition: Q versus K composition",
    xaxis_title="Dataset",
    yaxis_title="Logit Difference",
)

fig.show()
