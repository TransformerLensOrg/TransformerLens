#%%
import warnings
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from interp.circuit.projects.ioi.ioi_methods import ablate_layers, get_logit_diff
import torch
import torch as t
from easy_transformer.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    print_gpu_mem,
)  # helper functions
from ioi_utils import *
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
    ABBA_TEMPLATES,
)
from ioi_utils import (
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
)


ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%%
model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")

print_gpu_mem("About to load model")
model = EasyTransformer(model_name, use_attn_result=True)
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts("S2")

from ioi_circuit_extraction import (
    ARTHUR_CIRCUIT,
    join_lists,
    CIRCUIT,
    SMALL_CIRCUIT,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
)

circuit = CIRCUIT.copy()


def get_basic_extracted_model(model, ioi_dataset, mean_dataset=None, circuit=circuit):
    if mean_dataset is None:
        mean_dataset = ioi_dataset
    heads_to_keep = get_heads_circuit(
        ioi_dataset,
        excluded_classes=[],
        circuit=circuit,
    )
    torch.cuda.empty_cache()

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=abca_dataset,
    )
    return model


model = get_basic_extracted_model(
    model,
    ioi_dataset,
    mean_dataset=abca_dataset,
    circuit=circuit,
)
torch.cuda.empty_cache()

metric = logit_diff

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

print(f"{circuit_baseline_diff=}, {circuit_baseline_diff_std=}")
print(f"{circuit_baseline_prob=}, {circuit_baseline_prob_std=}")
print(f"{baseline_ldiff=}, {baseline_ldiff_std=}")
print(f"{baseline_prob=}, {baseline_prob_std=}")
#%% TODO Explain the way we're doing the minimal circuit experiment here
results = {}
results["ldiff_circuit"] = circuit_baseline_diff
vertices = []

extra_ablate_classes = list(circuit.keys())

xs = [baseline_ldiff, circuit_baseline_diff]
ys = [baseline_prob, circuit_baseline_prob]
both = [xs, ys]
labels = ["baseline", "circuit"]

for extra_ablate in tqdm(extra_ablate_classes):
    extra_ablate_subset = [extra_ablate]
    for circuit_class in list(circuit.keys()):
        if circuit_class not in extra_ablate_subset:
            continue
        for layer, idx in circuit[circuit_class]:
            vertices.append((layer, idx))

    # compute METRIC(C \ W)
    heads_to_keep = get_heads_circuit(
        ioi_dataset, excluded_classes=extra_ablate_subset, circuit=circuit
    )
    torch.cuda.empty_cache()

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=abca_dataset,
    )
    labels.append(str(extra_ablate_subset))
    torch.cuda.empty_cache()
    for i, a_metric in enumerate([logit_diff, probs]):
        ans, std = a_metric(model, ioi_dataset, std=True)
        torch.cuda.empty_cache()
        both[i].append(ans)

        if len(extra_ablate_subset) == 1 and metric == a_metric:
            if extra_ablate_subset[0] not in results:
                results[extra_ablate_subset[0]] = {}
            results[extra_ablate_subset[0]]["metric_calc"] = ans

fig = px.scatter(x=xs, y=ys, text=labels)
fig.update_traces(textposition="top center")
fig.show()
#%%
# METRIC((C \ W) \cup \{ v \})

for i, circuit_class in enumerate(
    [key for key in circuit.keys() if key in extra_ablate_classes]
):
    results[circuit_class]["vs"] = {}
    for v in tqdm(list(circuit[circuit_class])):
        new_heads_to_keep = get_heads_circuit(
            ioi_dataset, excluded_classes=[circuit_class], circuit=circuit
        )
        v_indices = get_extracted_idx(RELEVANT_TOKENS[v], ioi_dataset)
        assert v not in new_heads_to_keep.keys()
        new_heads_to_keep[v] = v_indices

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_keep=new_heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=abca_dataset,
        )
        torch.cuda.empty_cache()
        metric_calc = metric(model, ioi_dataset, std=True)
        results[circuit_class]["vs"][v] = metric_calc
        torch.cuda.empty_cache()
#%%
fig = go.Figure()

for G in list(extra_ablate_classes):
    if len(circuit[G]) > 4:
        warnings.warn("just plotting first 4 vertices per class")
    for i, v in enumerate(list(circuit[G])[:4]):
        fig.add_trace(
            go.Bar(
                x=[G],
                y=[results[G]["vs"][v][0] - results[G]["metric_calc"]],
                base=results[G]["metric_calc"],
                width=1 / (len(CIRCUIT[G]) + 1),
                offset=(i - 3 / 2) / (len(CIRCUIT[G]) + 1),
                marker_color=["crimson", "royalblue", "darkorange", "limegreen"][i],
                text=f"{v}",
                name=f"{v}",
                textposition="outside",
            )
        )

fig.add_shape(
    type="line",
    xref="x",
    x0=-2,
    x1=8,
    yref="y",
    y0=baseline_prob if metric == probs else baseline_ldiff,
    y1=baseline_prob if metric == probs else baseline_ldiff,
    line_width=1,
    line=dict(
        color="blue",
        width=4,
        dash="dashdot",
    ),
)

fig.add_trace(
    go.Scatter(
        x=["induction"],
        y=[3.3],
        text=["Logit difference of M"],
        mode="text",
        textfont=dict(
            color="blue",
            size=10,
        ),
    )
)

fig.add_shape(
    type="line",
    xref="x",
    x0=-2,
    x1=8,
    yref="y",
    y0=circuit_baseline_prob if metric == probs else circuit_baseline_diff,
    y1=circuit_baseline_prob if metric == probs else circuit_baseline_diff,
    line_width=1,
    line=dict(
        color="black",
        width=4,
        dash="dashdot",
    ),
)

fig.add_trace(
    go.Scatter(
        x=["induction"],
        y=[3.8],
        text=["Logit difference of C"],
        mode="text",
        textfont=dict(
            color="black",
            size=10,
        ),
    )
)

fig.update_layout(showlegend=False)

fig.update_layout(
    title="Change in logit diff when ablating all of a circuit node class when adding back one attention head",
    xaxis_title="Circuit node class",
    yaxis_title="Average logit diff",
)

fig.show()
#%%

cur_stuff = []
for circuit_class in CIRCUIT.keys():
    if circuit_class == "negative":
        continue
    for head in CIRCUIT[circuit_class]:
        for relevant_token in RELEVANT_TOKENS[head]:
            cur_stuff.append((head, relevant_token))
heads = {head: [] for head, _ in cur_stuff}
for head, val in cur_stuff:
    heads[head].append(val)
heads_to_keep = {}
for head in heads.keys():
    heads_to_keep[head] = get_extracted_idx(heads[head], ioi_dataset)
model.reset_hooks()

model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)

scatter_attention_and_contribution(model, 9, 9, ioi_dataset.ioi_prompts)
# %%
