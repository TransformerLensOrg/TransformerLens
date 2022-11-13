#%% [markdown]
# Setup
from copy import deepcopy
import torch

assert torch.cuda.device_count() == 1
from tqdm import tqdm
from easy_transformer.experiments import get_act_hook
import plotly.graph_objects as go
from numpy import efrom numpy import e
import pandas as pd
import torch
import torch as t
from easy_transformer.EasyTransformer import (
    EasyTransformer,
)
from time import ctime
from functools import partial

import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import random
import einops
from IPython import get_ipython
from copy import deepcopy
from ioi_dataset import (
    IOIDataset,
)
from ioi_utils import (
    path_patching,
    max_2d,
    CLASS_COLORS,
    show_pp,
    show_attention_patterns,
    scatter_attention_and_contribution,
)
from random import randint as ri
from ioi_circuit_extraction import (
    do_circuit_extraction,
    get_heads_circuit,
    CIRCUIT,
)
from ioi_utils import logit_diff, probs
from ioi_utils import get_top_tokens_and_probs as g

from random import randint
from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from functools import partial
import itertools
import pathlib
from pathlib import PurePath as PP
from copy import copy

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
os.environ["RR_CIRCUITS_REPR_NAME"] = "true"

RRFS_DIR = os.path.expanduser("~/rrfs")
RRFS_INTERP_MODELS_DIR = f"{RRFS_DIR}/interpretability_models_jax/"
os.environ["INTERPRETABILITY_MODELS_DIR"] = os.environ.get(
    "INTERPRETABILITY_MODELS_DIR",
    os.path.expanduser("~/interp_models_jax/")
    if os.path.exists(os.path.expanduser("~/interp_models_jax/"))
    else RRFS_INTERP_MODELS_DIR,
)

from tqdm import tqdm
from tabulate import tabulate
import numpy as np

np.random.seed(1726)
import attrs
from attrs import frozen
import torch
import torch as t
import einops
import jax
import plotly.express as px
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
import os

os.chdir("/home/arthur/unity")
print("User Arthur")
print(os.getcwd())

from interp.circuit.get_update_node import (
    FalseMatcher,
    NodeMatcher,
    TrueMatcher,
    TypeMatcher,
    EqMatcher,
    AnyMatcher,
    NameMatcher as NM,
    NodeUpdater as NU,
    IterativeNodeMatcher as INM,
    FunctionIterativeNodeMatcher as F,
    BasicFilterIterativeNodeMatcher as BF,
    Rename,
    Replace,
    RegexMatcher as RE,
    replace_circuit,
    sub_name,
)
from pathlib import PurePath as PP

#%%

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)

N = 100
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
)

# model2 = EasyTransformer.from_pretrained("gpt2-xl").cuda() # eek this is too big
# model2.set_use_attn_result(True)

#%%
# DATA SHIT
print("WARN: not using Ryan stuff")
data_rrfs = os.path.expanduser(f"~/rrfs/pretraining_datasets/owt_tokens_int16_val/0.pt")
data_suffix = "name_data/data-2022-07-30.pt"
data_local = os.path.expanduser(f"~/{data_suffix}")
try:
    data_full = torch.load(data_local)
except FileNotFoundError:
    data_full = torch.load(data_rrfs)
toks = data_full["tokens"].long() + 32768
lens = data_full["lens"].long()


def d(tokens, tokenizer=model.tokenizer):
    return tokenizer.decode(tokens)


if False:
    SEQ_LEN = 10
    print(f"WARN: SEQ_LEN = {SEQ_LEN}")
    MODEL_ID = "attention_only_four_layers_untied"
    MODEL_ID = "attention_only_two_layers_untied"
    # MODEL_ID = "jan5_attn_only_two_layers"
    DATASET_SIZE = 8000  # total data points is twice this...
    DATASET_DIR = PP("/home/arthur/rrfs/arthur/induction/data7/")
    MODIFY_DATASETS = False
    TRIM_TO_SIZE = False
    FIND_SAME_TOKEN = False

    DATASET_PATH = DATASET_DIR / "ind.pt"
    MADE_DATA = os.path.exists(DATASET_PATH)
    VOCAB_SIZE = 50259
    if os.path.exists(DATASET_PATH):
        smol = torch.load(str(DATASET_PATH))
        print("Trying to decode ...")
        print(d(smol[0, :]))
        print("... done.")
    else:
        print("Rip, no smol found")
        if not os.path.exists(DATASET_DIR):
            print(f"Made {str(DATASET_DIR)}")
            os.mkdir(DATASET_DIR)

#%% [markdown]
# Check that the model BPBs roughly agree with https://arxiv.org/pdf/2101.00027v1.pdf page 8


def perplexity(losses):
    return torch.exp(torch.mean(losses))


def bpb(losses):
    """Cursed EleutherAI value"""
    return (0.29335 / np.log(2)) * losses


def get_loss(model, tokens, return_per_token=False):
    losses = model(
        tokens,
        return_type="loss",
        return_per_token=return_per_token,
    )
    return losses.item()


model_name_list = [
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "gpt2-large",
    "EleutherAI/gpt-neo-1.3B",
    "gpt2-xl",
    "EleutherAI/gpt-neo-2.7B",
]


def get_bpb(model_name, toks, lens, samples=100, manual_eos=None):
    model = EasyTransformer.from_pretrained(model_name).cuda()
    model.set_use_attn_result(True)
    loss_list = []
    for idx in tqdm(range(samples)):
        cur = torch.cat(
            (
                torch.tensor([model.tokenizer.pad_token_id])
                if manual_eos is None
                else torch.tensor([manual_eos]),
                toks[torch.sum(lens[:idx]) : torch.sum(lens[: idx + 1])],
            )
        )
        cur_tokens = cur.unsqueeze(0)[:, :1024]
        cur_tokens[:, 0] = model.tokenizer.pad_token_id

        losses = get_loss(
            model, cur_tokens
        ).mean().item()  # this is the average over a input sequence
        loss_list.append(losses)

    bs = [bpb(t) for t in loss_list]
    return torch.tensor(bs)


#%% [markdown]
# takes about 2 mins to run but checks the models are getting legit scores

bpb_results = {}
for model_name in tqdm(model_name_list):
    bpb_results["model_name"] = get_bpb(model_name, toks, lens, samples=100)

#%%
fig = go.Figure()

for idx, model_name in tqdm(enumerate(model_name_list)):
    fig.add_trace(
        go.Box(
            y=bpb_results[model_name],
            name=model_name,
        )
    )
fig.show()

#%%
expected_ranges = {  # note these overlap
    "gpt2": (1.25, 1.35),  # 1.22 in the pile paper
    "EleutherAI/gpt-neo-125M": (1.25, 1.35),
    "gpt2-large": (1.05, 1.20),  # 1.08 in the pile paper
    "EleutherAI/gpt-neo-1.3B": (1.05, 1.20),
    "gpt2-xl": (1.0, 1.1),  # 1.04 in the pile paper
    "EleutherAI/gpt-neo-2.7B": (1.0, 1.1),
}

for model_name in model_name_list:
    mean = bpb_results[model_name].mean()
    assert (
        expected_ranges[model_name][0] < mean < expected_ranges[model_name][1]
    ), f"Model {model_name} has mean BPB {mean} which is outside the expected range {expected_ranges[model_name]}"

for model_name_mid in model_name_list[2:4]:
    # check that there is a sensible model performance improvement
    for model_name_low in model_name_list[:2]:
        assert (
            bpb_results[model_name_mid].mean() < bpb_results[model_name_low].mean()
        ), f"Model {model_name_mid} has mean BPB {bpb_results[model_name_mid].mean()} which is not less than {model_name_low} which has mean BPB {bpb_results[model_name_low].mean()}"
    for model_name_high in model_name_list[4:]:
        assert (
            bpb_results[model_name_mid].mean() > bpb_results[model_name_high].mean()
        ), f"Model {model_name_mid} has mean BPB {bpb_results[model_name_mid].mean()} which is not greater than {model_name_high} which has mean BPB {bpb_results[model_name_high].mean()}"

if "all_results" not in globals():
    all_results = {model_name: [] for model_name in model_name_list}

#%%
tot = []


def patch_all(z, source_act, hook):
    assert z.shape == source_act.shape, f"{z.shape} != {source_act.shape}"
    z[:] = source_act
    return z


def get_losses(
    model,
    manual_eos=None,
    heads=[],  # (random) ablate these heads
    samples=100,
    acts=None,
    batch_size=1,
    return_per_token=False,
):
    list_losses = []

    assert (
        samples % batch_size == 0
    ), f"Samples {samples} must be divisible by batch size {batch_size}"

    idx = 0
    for batch_idx in tqdm(range(samples // batch_size)):  # range(len(lens)):
        for in_batch_idx in range(batch_size):
            cur = torch.cat(
                (
                    torch.tensor([model.tokenizer.pad_token_id])
                    if manual_eos is None
                    else torch.tensor([manual_eos]),
                    toks[
                        torch.sum(
                            lens[: in_batch_idx + batch_size * batch_idx]
                        ) : torch.sum(lens[: in_batch_idx + batch_size * batch_idx + 1])
                    ],
                )
            )
            cur_tokens = cur.unsqueeze(0)[:, :1024]
            cur_tokens[:, 0] = model.tokenizer.bos_token_id

        model.reset_hooks()
        for head in heads:
            hook_name = f"blocks.{head[0]}.attn.hook_result"
            alt_act = acts[randint(0, -1 + len(acts))][hook_name][
                :, : cur_tokens.shape[1]
            ]

            hook = get_act_hook(
                patch_all,
                alt_act=alt_act,
                idx=head[1],
                dim=2 if head[1] is not None else None,
                name=hook_name,
            )
            model.add_hook(hook_name, hook)
        list_losses.append(get_loss(model, cur_tokens, return_per_token=return_per_token))
    return torch.tensor(list_losses)


#%%

head_nos = [0, 1, 2, 5, 10, 15, 20, 100]
head_nos += [40, 50, 60]

# sort head_nos
head_nos = sorted(head_nos)

def do_thing(model_name):
    model = EasyTransformer.from_pretrained(model_name).cuda()
    model.set_use_attn_result(True)

    def cached(model, tokens):
        model.reset_hooks()
        cache = {}
        model.cache_all(cache)
        model(tokens)
        for key in list(cache.keys()):
            cache[key] = cache[key].detach().clone().cpu()
        return cache

    acts = []
    assert len(the_1024s) > 10, f"Not enough samples, only {len(the_1024s)}"
    the_1024 = the_1024s[randint(0, len(the_1024s) - 1)][
        :, :1024
    ]  # this is something that is a global variable ... I don't THINK anything else is
    the_1024[:, 0] = model.tokenizer.bos_token_id

    print("Caching random things")
    for i in tqdm(range(10)):
        acts.append(cached(model, the_1024))
    print("...done")

    all_heads = [(i, j) for i in range(12) for j in range(12)]

    for num_heads in head_nos:
        cur_heads = random.sample(all_heads, num_heads)
        losses = get_losses(model, heads=cur_heads, acts=acts)
        print(
            f"Model {model_name} loss: {losses.mean()} +- {losses.std()}, {num_heads} heads"
        )
        all_results[model_name].append(
            {"num_heads": num_heads, "losses": losses, "ctime": ctime()}
        )


for model_name in model_name_list[2:4]: # ["EleutherAI/gpt-neo-125M", "gpt2"]:
    if model_name not in all_results:
        all_results[model_name] = []
    do_thing(model_name)

#%%

def line_with_error(
    xs, ys, errs, show=True, fig=None, color="royalblue", show_err=True, name="mean", yaxis="",
):
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    errs = torch.tensor(errs)
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=2),
            name=name,
        )
    )
    if show_err:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys + errs,
                mode="lines",
                line=dict(color=color, width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys - errs,
                fill="tonexty",
                fillcolor="rgba(68, 68, 68, 0.3)",
                mode="lines",
                line=dict(color=color, width=0),
                showlegend=False,
            )
        )
    fig.update_layout(
        xaxis_title="Number of heads",
        yaxis_title=yaxis,
    )
    if show:
        fig.show()
    return fig


cutoff = 20
colors = ["royalblue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

fig = go.Figure()

for model_idx, model_name in enumerate(
    model_name_list[2:4]
):  #  ["gpt2", "EleutherAI/gpt-neo-125M"]:
    ys = all_results[model_name]

    # sort ys by the head_no key
    ys.sort(key=lambda x: x["num_heads"])

    # only keep one entry per head_no
    ys = [
        ys[i]
        for i in range(len(ys))
        if i == 0 or ys[i]["num_heads"] != ys[i - 1]["num_heads"]
    ]

    ys = ys[:cutoff]

    line_with_error(
        head_nos[:cutoff],
        # torch.exp(-torch.tensor([y["losses"].mean() for y in ys])),
        # torch.exp(-torch.tensor([y["losses"].std() for y in ys])),
        torch.tensor([y["losses"].mean() for y in ys]),
        torch.tensor([y["losses"].std() for y in ys]),
        show=False,
        fig=fig,
        color=colors[model_idx],
        show_err=False,
        name=model_name,
        yaxis="loss",
    )

fig.show()

#%% [markdown]
# create some things to random patch in
# WARNING I don't think the Eleuther models have 50256...

lengths = []
the_1024s = []
for idx in tqdm(range(len(lens))):  # range(len(lens)):
    cur = torch.cat(
        (
            torch.tensor([model.tokenizer.pad_token_id]),
            toks[torch.sum(lens[:idx]) : torch.sum(lens[: idx + 1])],
        )
    )
    cur_tokens = cur.unsqueeze(0)

    if cur_tokens.shape[1] >= 1024:
        the_1024s.append(cur_tokens)

# lengths.append(min(1024, cur_tokens.shape[1]))
# # plot a histogram of the lengths
# if idx % 500 == 100:
#     plt.hist(lengths, bins=100)
#     # set the x axis max to 1024
#     plt.xlim(0, 1024)
#     plt.show()

#%% [markdown]
# Do some positive writing direction and negative writing direction stuff

def pos_neg(
    model_name,
):
    # SAME AS DO THING
    model = EasyTransformer.from_pretrained(model_name).cuda()
    model.set_use_attn_result(True)

    def cached(model, tokens):
        model.reset_hooks()
        cache = {}
        model.cache_all(cache)
        model(tokens)
        for key in list(cache.keys()):
            cache[key] = cache[key].detach().clone().cpu()
        return cache

    acts = []
    assert len(the_1024s) > 10, f"Not enough samples, only {len(the_1024s)}"
    the_1024 = the_1024s[randint(0, len(the_1024s) - 1)][
        :, :1024
    ]  # this is something that is a global variable ... I don't THINK anything else is
    the_1024[:, 0] = model.tokenizer.bos_token_id

    print("Caching random things")
    for i in tqdm(range(10)):
        acts.append(cached(model, the_1024))
    print("...done")
    # END SAME 

    all_heads = [(i, j) for i in range(12) for j in range(12)]

    for head in all_heads:
        if model_name not in all_losses:
            all_losses[model_name] = []
        all_losses[model_name].append({
            "head": head,
            "losses": get_losses(model, heads=[head], acts=acts),
            "ctime": ctime(),
        })
        break

if "all_losses" not in globals():
    all_losses = {}
pos_neg("gpt2")