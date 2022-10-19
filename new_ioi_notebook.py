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
from ioi_utils import patch_and_freeze
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


#%% [markdown] first patch-and-freeze experiments
# TODO why are there effects that come AFTER the patching?? it's before 36 mins in voko I think

dataset_names = [
    # "ioi_dataset",
    # "abca_dataset",
    # "dcc_dataset",
    # "acca_dataset",
    # "acba_dataset",
    "all_diff_dataset",
    # "totally_diff_dataset",
]

results = [torch.zeros(size=(12, 12)) for _ in range(len(dataset_names))]
mlp_results = [torch.zeros(size=(12, 1)) for _ in range(len(dataset_names))]

# patch all heads into the name mover input (hopefully find S2 Inhibition)

model.reset_hooks()
default_logit_diff = logit_diff(model, ioi_dataset)

# for pos in ["IO", "S", "S2"]:

for pos in ["end"]:
    results = [torch.zeros(size=(12, 12)) for _ in range(len(dataset_names))]
    mlp_results = [torch.zeros(size=(12, 1)) for _ in range(len(dataset_names))]
    for source_layer in tqdm(range(12)):
        for source_head_idx in list(range(12)):
            for dataset_idx, dataset_name in enumerate(dataset_names):
                dataset = eval(dataset_name)

                model.reset_hooks()

                receiver_hooks = [
                    (f"blocks.{layer}.attn.hook_q", head_idx)
                    for layer, head_idx in circuit["name mover"]
                ]

                # model = direct_patch_and_freeze(
                #     model=model,
                #     source_dataset=ioi_dataset,
                #     target_dataset=abca_dataset,  # if diff totally fall apart
                #     ioi_dataset=ioi_dataset,
                #     sender_heads=[(source_layer, source_head_idx)],
                #     receiver_hooks=receiver_hooks,
                #     max_layer=12,  # at this point, we just let the model run
                #     positions=["end"],
                #     verbose=False,
                # )

                max_layer = 12
                positions = ["end"]
                source_dataset = all_diff_dataset
                target_dataset = ioi_dataset  # if diff totally fall apart
                ioi_dataset = ioi_dataset
                sender_heads = [(source_layer, source_head_idx)]

                # BEGIN
                sender_hooks = [
                    (f"blocks.{layer}.attn.hook_result", head_idx)
                    for layer, head_idx in sender_heads
                ]
                sender_hook_names = [x[0] for x in sender_hooks]
                receiver_hook_names = [x[0] for x in receiver_hooks]

                sender_cache = {}
                model.reset_hooks()
                model.cache_some(sender_cache, lambda x: x in sender_hook_names)
                # print(f"{sender_hook_names=}")
                source_logits = model(source_dataset.text_prompts)

                target_cache = {}
                model.reset_hooks()
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

                # we can override the hooks above for the sender heads, though
                for hook_name, head_idx in sender_hooks:
                    assert not torch.allclose(
                        sender_cache[hook_name], target_cache[hook_name]
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
                for hook_name, head_idx in receiver_hooks:
                    hook = get_act_hook(
                        partial(patch_positions, positions=positions),
                        alt_act=receiver_cache[hook_name],
                        idx=head_idx,
                        dim=2 if head_idx is not None else None,
                        name=hook_name,
                    )
                    model.add_hook(hook_name, hook)
                # END OF FUNCTION CTRL C+V

                cur_logit_diff = logit_diff(model, ioi_dataset)

                if source_head_idx is None:
                    mlp_results[dataset_idx][source_layer] = (
                        cur_logit_diff - default_logit_diff
                    )
                else:
                    results[dataset_idx][source_layer][source_head_idx] = (
                        cur_logit_diff - default_logit_diff
                    )

                if source_layer == 11 and source_head_idx == 11:
                    # show attention head results
                    fname = f"svgs/patch_and_freeze_{dataset_name}_{pos}_{ctime()}_{ri(2134, 123759)}"
                    fig = show_pp(
                        results[dataset_idx].T,
                        title=f"{fname=} {dataset_name=} {pos=} patching NMs",
                        return_fig=True,
                        show_fig=False,
                    )
                    fig.write_image(fname + ".png")
                    fig.write_image(fname + ".svg")
                    fig.show()

                    # show mlp results
                    fig = show_pp(
                        mlp_results[dataset_idx].T,
                        title=f"{dataset_name=} {pos=} patching NMs",
                        return_fig=True,
                        show_fig=False,
                    )
                    fname = f"svgs/patch_and_freeze_{dataset_name}_{pos}_mlp_{ctime()}_{ri(2134, 123759)}"
                    fig.write_image(fname + ".png")
                    fig.write_image(fname + ".svg")
                    fig.show()
