#%%
import os
import torch

if os.environ["USER"] in ["exx", "arthur"]:  # so Arthur can safely use octobox
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
assert torch.cuda.device_count() == 1

import warnings
from time import ctime
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from easy_transformer.ioi_utils import probs

# from interp.circuit.projects.ioi.ioi_methods import ablate_layers, get_logit_diff
from easy_transformer.ioi_utils import probs, logit_diff
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

from easy_transformer.ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    ABBA_TEMPLATES,
)
from easy_transformer.ioi_utils import (
    ALL_COLORS,
    CLASS_COLORS,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
)
from easy_transformer.ioi_circuit_extraction import (
    NAIVE,
    join_lists,
    CIRCUIT,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
)

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%%
model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")
print_gpu_mem("About to load model")
model = EasyTransformer.from_pretrained(model_name)
model.set_use_attn_result(True)
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)

abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)

mean_dataset = abc_dataset
#%% # do some initial experiments with the naive circuit

circuits = [None, CIRCUIT.copy(), NAIVE.copy()]
circuit = circuits[1]

metric = logit_diff

naive_heads = []
for heads in circuits[2].values():
    naive_heads += heads

model.reset_hooks()
model_baseline_metric = metric(model, ioi_dataset)

model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep={},
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=mean_dataset,
    excluded=naive_heads,
)

circuit_baseline_metric = metric(model, ioi_dataset)
print(f"{model_baseline_metric} {circuit_baseline_metric}")
#%%
def get_basic_extracted_model(
    model, ioi_dataset, mean_dataset=None, circuit=circuits[1]
):
    if mean_dataset is None:
        mean_dataset = ioi_dataset
    heads_to_keep = get_heads_circuit(
        ioi_dataset,
        excluded=[],
        circuit=circuit,
    )
    torch.cuda.empty_cache()

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )
    return model


model = get_basic_extracted_model(
    model,
    ioi_dataset,
    mean_dataset=mean_dataset,
    circuit=circuits[1],
)
torch.cuda.empty_cache()

circuit_baseline_diff, circuit_baseline_diff_std = logit_diff(
    model, ioi_dataset, std=True
)
torch.cuda.empty_cache()
circuit_baseline_prob, circuit_baseline_prob_std = probs(model, ioi_dataset, std=True)
torch.cuda.empty_cache()
model.reset_hooks()
baseline_ldiff, baseline_ldiff_std = logit_diff(model, ioi_dataset, std=True)
torch.cuda.empty_cache()
baseline_prob, baseline_prob_std = probs(model, ioi_dataset, std=True)

print(f"{circuit_baseline_diff}, {circuit_baseline_diff_std}")
print(f"{circuit_baseline_prob}, {circuit_baseline_prob_std}")
print(f"{baseline_ldiff}, {baseline_ldiff_std}")
print(f"{baseline_prob}, {baseline_prob_std}")

if metric == logit_diff:
    circuit_baseline_metric = circuit_baseline_diff
else:
    circuit_baseline_metric = circuit_baseline_prob

circuit_baseline_metric = [None, circuit_baseline_metric, circuit_baseline_metric]

#%% [markdown]
# define the sets K for every vertex in the graph

K = {}

for circuit_class in circuit.keys():
    for head in circuit[circuit_class]:
        K[head] = [circuit_class]

# rebuild J
for head in K.keys():
    new_j_entry = []
    for entry in K[head]:
        if isinstance(entry, str):
            for head2 in circuits[1][entry]:
                new_j_entry.append(head2)
        elif isinstance(entry, tuple):
            new_j_entry.append(entry)
        else:
            raise NotImplementedError(head, entry)
    assert head in new_j_entry, (head, new_j_entry)
    K[head] = list(set(new_j_entry))
# name mover shit
for i, head in enumerate(circuit["name mover"]):
    old_entry = K[head]
    K[head] = deepcopy(circuit["name mover"][: i + 1])  # turn into the previous things

for head in [(9, 0), (11, 9)]:
    K[head] = circuit["name mover"] + circuit["negative"]

K[(5, 8)] = [(11, 10), (10, 7), (5, 8)]
K[(5, 9)] = [(11, 10), (10, 7), (5, 9)]

#%% [markdown]
# Run the experiment

results = {}

if "results_cache" not in dir():
    results_cache = {}  # massively speeds up future runs

for circuit_class in circuit.keys():
    for head in circuits[1][circuit_class]:
        results[head] = [None, None]
        base = frozenset(K[head])
        summit_list = deepcopy(K[head])
        summit_list.remove(head)  # and this will error if you don't have a head in J!!!
        summit = frozenset(summit_list)

        for idx, ablated_stuff in enumerate([base, summit]):
            if ablated_stuff not in results_cache:  # see the if False line
                new_heads_to_keep = get_heads_circuit(
                    ioi_dataset, excluded=ablated_stuff, circuit=circuit
                )
                model.reset_hooks()
                model, _ = do_circuit_extraction(
                    model=model,
                    heads_to_keep=new_heads_to_keep,
                    mlps_to_remove={},
                    ioi_dataset=ioi_dataset,
                    mean_dataset=mean_dataset,
                )
                torch.cuda.empty_cache()
                metric_calc = metric(model, ioi_dataset, std=False)
                results_cache[ablated_stuff] = metric_calc
                print("Do sad thing")
            results[head][idx] = results_cache[ablated_stuff]

        print(
            f"{head} with {K[head]}: progress from {results[head][0]} to {results[head][1]}"
        )

ac = ALL_COLORS
cc = CLASS_COLORS.copy()

relevant_classes = list(circuit.keys())
fig = go.Figure()

initial_y_cache = {}
final_y_cache = {}

the_xs = {}

for j, G in enumerate(relevant_classes + ["backup name mover"]):
    xs = []
    initial_ys = []
    final_ys = []
    colors = []
    names = []
    widths = []
    if G == "backup name mover":
        curvys = list(circuit["name mover"])
        for head in [(9, 6), (9, 9), (10, 0)]:
            curvys.remove(head)
    elif G == "name mover":
        curvys = [(9, 6), (9, 9), (10, 0)]
    else:
        curvys = list(circuit[G])
    curvys = sorted(curvys, key=lambda x: -abs(results[x][1] - results[x][0]))

    for v in curvys:
        colors.append(cc[G])
        xs.append(str(v))
        initial_y = results[v][0]
        final_y = results[v][1]

        initial_ys.append(initial_y)
        final_ys.append(final_y)

    the_xs[G] = xs
    initial_ys = torch.Tensor(initial_ys)
    final_ys = torch.Tensor(final_ys)
    initial_y_cache[G] = initial_ys
    final_y_cache[G] = final_ys

    y = final_ys - initial_ys

    if True:
        base = [0.0 for _ in range(len(xs))]
        warnings.warn("Base is 0")
        y = abs(y)
    else:
        base = initial_ys

    fig.add_trace(
        go.Bar(
            x=xs,
            y=y,
            base=base,
            marker_color=colors,
            width=[
                1.0 for _ in range(len(xs))
            ],  ## if G != "dis" else [0.2 for _ in range(len(xs))],
            name=G,
        )
    )


fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

fig.update_layout(
    # title="Change in logit diff when ablating all of a circuit node class when adding back one attention head",
    xaxis_title="Attention head",
    yaxis_title="Change in logit difference",
)

fig.update_xaxes(
    gridcolor="black",
    gridwidth=0.1,
    # minor=dict(showgrid=False),  # please plotly, just do what I want
)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
fig.update_yaxes(gridcolor="black", gridwidth=0.1)
fig.write_image(f"svgs/circuit_minimality_at_{ctime()}.svg")
fig.show()
