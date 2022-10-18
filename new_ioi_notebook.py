#%% [markdown]
# # GPT-2 small Indirect Object Identification
# <h1><b>Intro</b></h1>
# This notebook is an implementation of the IOI experiments (all from the <a href="https://docs.google.com/presentation/d/19H__CYCBL5F3M-UaBB-685J-AuJZNsXqIXZR-O4j9J8/edit#slide=id.g14659e4d87a_0_290">paper</a>.
# It should be able to be run as by just git cloning this repo (+ some easy installs).
#
# ### Task
# We're interested in GPT-2 ability to complete sentences like "Alice and Bob went to the store, Alice gave a bottle of milk to"...
# GPT-2 knows that it have to output a name that is not the subject (Alice) and that was present in the context: Bob.
# The first apparition of Alice is called "S" (or sometimes "S1") for "Subject", and Bob is the indirect object ("IO"). Even if the sentences we generate contains the last word of the sentence "Bob", we'll never look at the transformer output here. What's matter is the next-token prediction on the token "to", sometime called the "end" token.
#
# ### Tools
# In this notebook, we define a class `IOIDataset` to handle the generation and utils for this particular dataset.
#
# Refer to the demo of the [`easy_transformer` library](https://github.com/neelnanda-io/Easy-Transformer) here: <a href="https://colab.research.google.com/drive/1MLwJ7P94cizVs2LD8Qwi-vLGSoH-cHxq?usp=sharing">demo with ablation & patching</a>.
#
# Reminder of the circuit:
# <img src="https://i.imgur.com/PPtTQRh.png">

# ## Imports
import abc
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

assert torch.cuda.device_count() == 1
from ioi_utils import all_subsets
from ioi_circuit_extraction import (
    CIRCUIT,
    RELEVANT_TOKENS,
    do_circuit_extraction,
    get_heads_circuit,
)
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


#%% [markdown] The model, and loads and loads of datasets
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
    model,
    text_prompts,
    variation=True,
    scale=True,
    # hook_names=[],
    # att_from=[],
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


#%% [markdown] general setup for freeze_and_patch
# %% [markdown] Alex's experiment on DCC patching LITERALLY EVERYTHING at position S2
layers = list(range(12))

config = PatchingConfig(
    source_dataset=dcc_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=partial(patch_positions, positions=["S2"]),
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


model.reset_hooks()
for layer in range(12):
    for head_idx in [None] + list(range(12)):

        # for layer, head_idx in circuit["induction"] + circuit["duplicate token"]:
        hook = patching.get_hook(
            layer,
            head_idx,
            manual_patch_fn=partial(
                patch_positions,
                positions=["S2"],
            ),
        )
        model.add_hook(*hook)

# model.reset_hooks()
cur_logit_diff = logit_diff(model, ioi_dataset)
cur_io_probs = probs(model, ioi_dataset)
print(f"{cur_logit_diff}, {cur_io_probs=}")
#%% [markdown] wait, so it's the case that positional clues from S2 from S Inhibition heads are quite weak. Are they useful for this on ABC, too?
# see the change from ABB to ACC (patching S2 Inhibition bois)
heads = circuit["s2 inhibition"] + [(10, 0)]

config = PatchingConfig(
    source_dataset=all_diff_dataset.text_prompts,
    target_dataset=acba_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=partial(patch_positions, positions=["end"]),
    layers=(0, max([layer for layer, _ in heads]) - 1),
)
metric = ExperimentMetric(
    attention_probs,
    acba_dataset,
    relative_metric=True,
    scalar_metric=False,
)
patching = EasyPatching(model, config, metric)
res = patching.run_experiment()
show_pp(res.T, title="Change in logit diff (patching XYZ->ACB)")
#%%
all_combs = list(itertools.product([False, True], [False, True]))

for use_circuit, extra_hooks in all_combs:
    model.reset_hooks()

    heads_to_keep = get_heads_circuit(
        ioi_dataset,
        circuit=circuit,
    )

    if use_circuit:
        model, abl = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
        )

    if extra_hooks:
        # for layer in range(9):
        #     for head_idx in [None] + list(range(12)):
        for layer, head_idx in circuit["s2 inhibition"]:
            hook = patching.get_hook(
                layer,
                head_idx,
                manual_patch_fn=partial(
                    patch_positions,
                    positions=["end"],
                ),
            )
            model.add_hook(*hook)

    print(f"{use_circuit=} {extra_hooks=} {logit_diff(model, acba_dataset)=}")
#%% [markdown] look at the attention scores of NMs to S2. Is this affected much by S2's output? What about if all the duplicate token heads and induction heads are patched?

# def attention_on_token(model, ioi_dataset, layer, head_idx, token, all=False, std=False, scores=False):

for pos in ["S2", "S", "IO"]:
    config = PatchingConfig(
        source_dataset=acba_dataset.text_prompts,
        target_dataset=ioi_dataset.text_prompts,
        target_module="attn_head",
        head_circuit="result",
        cache_act=True,
        verbose=False,
        patch_fn=partial(patch_positions, positions=["end"]),
        layers=(0, 8),
    )

    att_func = partial(attention_on_token, layer=9, head_idx=9, token=pos, scores=True)

    metric = ExperimentMetric(
        att_func,
        ioi_dataset,
        relative_metric=False,
        scalar_metric=False,
    )
    patching = EasyPatching(model, config, metric)

    # #%%
    # res = patching.run_experiment()
    # show_pp(res.T, title="Change in attention score to S2")

    print(f"{pos=}")

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=get_heads_circuit(ioi_dataset, circuit=circuit),
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=all_diff_dataset,
    )
    print(f"{att_func(model, ioi_dataset)=}")
    for layer, head_idx in circuit["s2 inhibition"]:
        for thing in [None]:
            # for layer in range(9):
            #     for head_idx in [None] + list(range(12)):
            hook = patching.get_hook(
                layer,
                head=head_idx,
                target_module="mlp" if head_idx is None else "attn_head",
                manual_patch_fn=partial(
                    patch_positions,
                    positions=["end"],
                ),
            )
            model.add_hook(*hook)

    print(f"{att_func(model, ioi_dataset)=}")
