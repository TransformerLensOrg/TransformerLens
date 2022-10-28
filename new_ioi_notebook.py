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
from csv import excel
import os
import time

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
from ioi_utils import path_patching
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
model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts(
    ("S2", "RAND")
)  # we flip the second b for a random c
abca_dataset.word_idx = ioi_dataset.word_idx
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

baba_dataset = IOIDataset(N=100, prompt_type="BABA")
abba_dataset = IOIDataset(N=100, prompt_type="ABBA")

baba_all_diff = (
    baba_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"), manual_word_idx=baba_dataset.word_idx)
)
abba_all_diff = (
    abba_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"), manual_word_idx=abba_dataset.word_idx)
)
circuit = deepcopy(CIRCUIT)

#%%
def patch_positions(z, source_act, hook, positions=["end"]):
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z


def patch_all(z, source_act, hook):
    return source_act


#%% [markdown] define the patch and freeze function


def direct_patch_and_freeze(
    model,
    source_dataset,
    target_dataset,
    ioi_dataset,
    sender_heads,
    receiver_hooks,
    max_layer,
    positions=["end"],
    verbose=False,
    return_hooks=False,
    extra_hooks=[],  # when we call reset hooks, we may want to add some extra hooks after this, add these here
):
    """
    Patch in the effect of `sender_heads` on `receiver_hooks` only
    (though MLPs are "ignored", so are slight confounders)
    """

    sender_hooks = []

    for layer, head_idx in sender_heads:
        if head_idx is None:
            raise NotImplementedError()

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]

    sender_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_some(sender_cache, lambda x: x in sender_hook_names)
    # print(f"{sender_hook_names=}")
    source_logits = model(source_dataset.text_prompts)

    target_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_all(target_cache)
    target_logits = model(target_dataset.text_prompts)

    # for all the Q, K, V things
    model.reset_hooks()
    for layer in range(max_layer):
        for head_idx in range(12):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)

                if hook_name in receiver_hook_names:
                    continue

                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                    name=hook_name,
                )
                model.add_hook(hook_name, hook)
    for (
        hook
    ) in (
        extra_hooks
    ):  # ughhh, think that this is what we want, this should override the QKV above
        model.add_hook(*hook)

    # we can override the hooks above for the sender heads, though
    for hook_name, head_idx in sender_hooks:
        assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name]), (
            hook_name,
            head_idx,
        )
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=sender_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        model.add_hook(hook_name, hook)

    # measure the receiver heads' values
    receiver_cache = {}
    model.cache_some(receiver_cache, lambda x: x in receiver_hook_names)
    receiver_logits = model(target_dataset.text_prompts)

    # patch these values in
    model.reset_hooks()
    model = model_fn(model)
    hooks = []
    for hook_name, head_idx in receiver_hooks:
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=receiver_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        hooks.append((hook_name, hook))

    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model


#%% [markdown] first patch-and-freeze experiments

dataset_names = [
    # "ioi_dataset",
    # "abca_dataset",
    # "dcc_dataset",
    # "acca_dataset",
    # "acba_dataset",
    "all_diff_dataset",
    # "totally_diff_dataset",
]

results = torch.zeros(size=(12, 12))
mlp_results = torch.zeros(size=(12, 1))

model.reset_hooks()
default_logit_diff = logit_diff(model, ioi_dataset)
print(default_logit_diff)

top_name_movers = [(9, 9), (9, 6), (10, 0)]
exclude_heads = [(layer, head_idx) for layer in range(12) for head_idx in range(12)]
for head in top_name_movers:
    exclude_heads.remove(head)

the_extra_hooks = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_circuit(
        ioi_dataset=ioi_dataset,
        circuit={"name mover": top_name_movers},
    ),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=all_diff_dataset,
    return_hooks=True,
    excluded=exclude_heads,
)

both_results = []
pos = "end"

for idx, extra_hooks in enumerate([[], the_extra_hooks]):
    results = torch.zeros(size=(12, 12))
    mlp_results = torch.zeros(size=(12, 1))

    model.reset_hooks()
    for hook in the_extra_hooks:
        model.add_hook(*hook)
    hooked_logit_diff = logit_diff(model, ioi_dataset)

    print(f"{hooked_logit_diff=}")
    model.reset_hooks()

    for source_layer in tqdm(range(12)):
        for source_head_idx in list(range(12)):
            model.reset_hooks()
            receiver_hooks = []
            receiver_hooks.append(("blocks.11.hook_resid_post", None))
            model = path_patching(
                model=model,
                source_dataset=all_diff_dataset,
                target_dataset=ioi_dataset,
                ioi_dataset=ioi_dataset,
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=receiver_hooks,
                max_layer=12,
                positions=[pos],
                verbose=False,
                return_hooks=False,
                extra_hooks=extra_hooks,
            )
            cur_logit_diff = logit_diff(model, ioi_dataset)

            if source_head_idx is None:
                mlp_results[source_layer] = cur_logit_diff - hooked_logit_diff
            else:
                results[source_layer][source_head_idx] = (
                    cur_logit_diff - hooked_logit_diff
                )

            if source_layer == 11 and source_head_idx == 11:
                fname = f"svgs/patch_and_freeze_{pos}_{ctime()}_{ri(2134, 123759)}"
                fig = show_pp(
                    results.T,
                    title=f"Direct effect of removing heads on logit diff" + ""
                    if idx == 0
                    else " (with top 3 name movers knocked out)",
                    return_fig=True,
                    show_fig=False,
                )

                both_results.append(results.clone())
                fig.show()
#%% [markdown] plotting (your downfalls!)

from ioi_utils import CLASS_COLORS

cc = deepcopy(CLASS_COLORS)
no_112 = deepcopy(CIRCUIT)
no_112["name mover"].remove((11, 2))
circuit = deepcopy(CIRCUIT)


def what_class(layer, head, circuit):
    for circuit_class in circuit:
        if (layer, head) in circuit[circuit_class]:
            return circuit_class
    return "duplicate token"
    raise ValueError((layer, head), circuit)


# plot the most important heads by

for idx, results in enumerate(both_results):
    k = 15
    top_heads = max_2d(torch.abs(results), k=k)[  # backup results or initial results
        0
    ]  # initial results is the patch with no KOs; direct effect on logits

    exclude_heads = []
    exclude_heads = [
        (layer_idx, head)
        for layer_idx in range(12)
        for head in range(12)
        if what_class(layer_idx, head, circuit=circuit)
        not in ["name mover", "negative", "s2 inhibition"]
    ]

    fig = go.Figure()
    heights = [
        results[layer][head]
        for layer, head in top_heads
        if (layer, head) not in exclude_heads
    ]
    colors = [
        cc[what_class(layer, head_idx, circuit=circuit)]
        for layer, head_idx in top_heads
        if (layer, head_idx) not in exclude_heads
    ]

    # plot a bar chart
    fig.add_trace(
        go.Bar(
            x=[str(x) for x in top_heads if x not in exclude_heads],
            y=heights,
            orientation="v",
            marker_color=colors,
        )
    )

    # set y axis range to [-1, 1]
    fig.update_yaxes(range=[-3, 3])

    # update title
    fig.update_layout(
        "Most important heads by direct effect on logits" + ""
        if idx == 0
        else " (with top 3 name movers knocked out)"
    )
    fig.show()
