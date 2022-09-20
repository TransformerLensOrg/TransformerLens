#%%
import warnings
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from ioi_utils import probs
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
    NAIVE_CIRCUIT,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
)

#%% # do some initial experiments with the naive circuit
circuits = [None, CIRCUIT.copy(), NAIVE_CIRCUIT.copy()]

naive_heads = []
for heads in circuits[2].values():
    naive_heads += heads

model.reset_hooks()
ld0 = logit_diff(model, ioi_dataset)

model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep={},
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=ioi_dataset,
    exclude_heads=naive_heads,
)

ld = logit_diff(model, ioi_dataset)
print(f"{ld0=} {ld=}")
#%%
def get_basic_extracted_model(
    model, ioi_dataset, mean_dataset=None, circuit=circuits[1]
):
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
        mean_dataset=mean_dataset,
    )
    return model


model = get_basic_extracted_model(
    model,
    ioi_dataset,
    mean_dataset=ioi_dataset,
    circuit=circuits[1],
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

circuit_baseline_diffs = [None, circuit_baseline_diff, ld]
#%% TODO Explain the way we're doing the minimal circuit experiment here
all_results = [{}, {}, {}]

max_ind = 3
for i in range(1, max_ind):
    print(f"Doing circuit {i} of {max_ind-1}")
    circuit = circuits[i]
    results = all_results[i]
    results["ldiff_circuit"] = circuit_baseline_diff
    vertices = []

    xs = [baseline_ldiff, circuit_baseline_diff]
    ys = [baseline_prob, circuit_baseline_prob]
    both = [xs, ys]
    labels = ["baseline", "circuit"]

    for class_to_ablate in tqdm(circuit.keys()):
        for circuit_class in list(circuit.keys()):
            if circuit_class != class_to_ablate:
                continue
            for layer, idx in circuit[circuit_class]:
                vertices.append((layer, idx))

        # compute METRIC(C \ W)

        if i == 1:
            heads_to_keep = get_heads_circuit(
                ioi_dataset, excluded_classes=[class_to_ablate], circuit=circuit
            )
            excluded_heads = []
        elif i == 2:
            heads_to_keep = {}
            excluded_heads = naive_heads.copy()
            for head in circuit[class_to_ablate]:
                excluded_heads.remove(head)
        else:
            raise NotImplementedError()
        torch.cuda.empty_cache()

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=abca_dataset,
            exclude_heads=excluded_heads,
        )
        labels.append(str(class_to_ablate))
        torch.cuda.empty_cache()
        for j, a_metric in enumerate([logit_diff, probs]):
            ans, std = a_metric(model, ioi_dataset, std=True)
            torch.cuda.empty_cache()
            both[j].append(ans)

            if metric == a_metric:
                if class_to_ablate not in results:
                    results[class_to_ablate] = {}
                results[class_to_ablate]["metric_calc"] = ans

    fig = px.scatter(x=xs, y=ys, text=labels)
    fig.update_traces(textposition="top center")
    fig.show()
#%%
# METRIC((C \ W) \cup \{ v \})

for i in range(1, max_ind):
    results = all_results[i]
    circuit = circuits[i]
    for index, circuit_class in enumerate(
        [key for key in circuit.keys() if key in list(circuit.keys())]
    ):
        results[circuit_class]["vs"] = {}
        for v in tqdm(list(circuit[circuit_class])):
            if i == 1:
                new_heads_to_keep = get_heads_circuit(
                    ioi_dataset, excluded_classes=[circuit_class], circuit=circuit
                )
                v_indices = get_extracted_idx(RELEVANT_TOKENS[v], ioi_dataset)
                assert v not in new_heads_to_keep.keys()
                new_heads_to_keep[v] = v_indices
                excluded_heads = []
            elif i == 2:
                new_heads_to_keep = {}
                excluded_heads = [v]
                for other_circuit_class in list(circuit.keys()):
                    if other_circuit_class == circuit_class:
                        continue
                    for layer, idx in circuit[other_circuit_class]:
                        excluded_heads.append((layer, idx))
            else:
                raise NotImplementedError()

            model.reset_hooks()
            model, _ = do_circuit_extraction(
                model=model,
                heads_to_keep=new_heads_to_keep,
                mlps_to_remove={},
                ioi_dataset=ioi_dataset,
                mean_dataset=abca_dataset,
                exclude_heads=excluded_heads,
            )
            torch.cuda.empty_cache()
            metric_calc = metric(model, ioi_dataset, std=True)
            results[circuit_class]["vs"][v] = metric_calc
            torch.cuda.empty_cache()
#%%
circuit_idx = 1
circuit = circuits[circuit_idx]
results = all_results[circuit_idx]

xs = []
initial_ys = []
final_ys = []

ac = px.colors.qualitative.Dark2
cc = {
    "name mover": ac[0],
    "negative": ac[1],
    "s2 inhibition": ac[2],
    "induction": ac[5],
    "duplicate token": ac[3],
    "previous token": ac[6],
}

relevant_classes = list(circuit.keys())
relevant_classes.remove("name mover")

fig = go.Figure()
colors = []
for j, G in enumerate(relevant_classes):
    for i, v in enumerate(list(circuit[G])):
        xs.append(str(v))
        initial_ys.append(results[G]["metric_calc"])
        final_ys.append(results[G]["vs"][v][0])
        colors.append(cc[G])


initial_ys = torch.Tensor(initial_ys)
final_ys = torch.Tensor(final_ys)

fig.add_trace(
    go.Bar(
        x=xs,
        y=final_ys - initial_ys,
        base=initial_ys,
        marker_color=colors,
        width=[1.0 for _ in range(len(xs))],
        name="",
    )
)

all_vs = []

for circuit_class in relevant_classes:
    vs = circuit[circuit_class]
    vs_str = [str(v) for v in vs]
    all_vs.extend(vs_str)
    fig.add_trace(
        go.Scatter(
            x=vs_str,
            y=[results[circuit_class]["metric_calc"] for _ in range(len(vs))],
            line=dict(color=cc[circuit_class]),
            name=circuit_class,  # labels=dict(x="vertex", y="metric"),
        )
    )

fig.add_trace(
    go.Scatter(
        x=all_vs,
        y=[
            (baseline_prob if metric == probs else baseline_ldiff)
            for _ in range(len(all_vs))
        ],
        name="Baseline model performance",
        line=dict(color="black"),
        fill="toself",
        mode="lines",
    )
)

fig.add_trace(
    go.Scatter(
        x=all_vs,
        y=[circuit_baseline_diffs[circuit_idx] for _ in range(len(all_vs))],
        name="Circuit performance",
        line=dict(color="blue"),
        fill="toself",
        mode="lines",
    )
)

fig.update_layout(
    title="Change in logit diff when ablating all of a circuit node class when adding back one attention head",
    xaxis_title="Attention head",
    yaxis_title="Average logit diff",
)

fig.show()
#%%
fig.add_shape(
    type="line",
    xref="x",
    x0=-0.50,  # TODO sort out axl these lines5
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

# fig.update_layout(showlegend=False)

fig.update_layout(
    title="Change in logit diff when ablating all of a circuit node class when adding back one attention head",
    xaxis_title="Circuit node class",
    yaxis_title="Average logit diff",
)

fig.show()
# %% # show some previous token head importance results (not that important)
circuit = CIRCUIT.copy()
lds = {}
prbs = {}

from ioi_utils import probs

ioi_dataset = IOIDataset(prompt_type="BABA", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts("S2")
torch.cuda.empty_cache()

for ioi_dataset in [ioi_dataset]:  # [ioi_dataset_baba, ioi_dataset_abba]:
    for S in all_subsets(["S+1", "and"]):
        heads_to_keep = get_heads_circuit(
            ioi_dataset,
            excluded_classes=["previous token"],  # , "duplicate token"],
            circuit=CIRCUIT,
        )
        torch.cuda.empty_cache()

        for layer, head_idx in circuit["previous token"]:
            heads_to_keep[(layer, head_idx)] = get_extracted_idx(S, ioi_dataset)

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            mlps_to_remove={},
            heads_to_keep=heads_to_keep,
            ioi_dataset=ioi_dataset,
            mean_dataset=abca_dataset,
        )
        torch.cuda.empty_cache()

        lds[tuple(S)] = logit_diff(model, ioi_dataset)
        prbs[tuple(S)] = probs(model, ioi_dataset)
        print(S, lds[tuple(S)], prbs[tuple(S)])
#%% # see the other NMS
vs = []
xs = []
for v, a in results["name mover"]["vs"].items():
    vs.append(v)
    xs.append(a[0].item() - 1.0138)
print(vs, xs)
#%% # run Alex's experiment
for i, circuit_class in enumerate(["name mover"]):
    for extra_v in [(11, 9), (9, 0)]:
        new_heads_to_keep = get_heads_circuit(
            ioi_dataset, excluded_classes=[circuit_class], circuit=circuit
        )
        for v in [extra_v] + [(9, 7), (11, 1)]:
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
        torch.cuda.empty_cache()
        print(extra_v, metric_calc)
#%% # new experiment idea: the duplicators and induction heads shouldn't care where their attention is going, provided that
# it goes to either S or S+1.

for j in range(2, 4):
    s_positions = ioi_dataset.word_idx["S"]

    # [batch, head_index, query_pos, key_pos] # so pass dim=1 to ignore the head
    def attention_pattern_modifier(
        z, hook
    ):  # batch, seq, head dim, because get_act_hook hides scary things from us
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_attn", hook.name
        assert (
            len(list(z.shape)) == 3
        ), z.shape  # batch, seq (attending_query), attending_key

        prior_stuff = []

        for i in range(-1, 2):
            prior_stuff.append(z[:, s_positions + i, :].clone())
            # print(prior_stuff[-1].shape, torch.sum(prior_stuff[-1]))

        for i in range(-1, 2):
            z[:, s_positions + i, :] = prior_stuff[
                (i + j) % 3
            ]  # +1 is the do nothing one

        return z

    model.reset_hooks()
    ld = logit_diff(model, ioi_dataset)
    # print(f"{ld=}")

    circuit_classes = ["induction"]
    circuit_classes = ["duplicate token"]
    circuit_classes = ["duplicate token", "induction"]

    for circuit_class in circuit_classes:
        for layer, head_idx in circuit[circuit_class]:
            cur_hook = get_act_hook(
                attention_pattern_modifier,
                alt_act=None,
                idx=head_idx,
                dim=1,
            )
            model.add_hook(f"blocks.{layer}.attn.hook_attn", cur_hook)

    ld2 = logit_diff(model, ioi_dataset)
    print(
        f"Initially there's a logit difference of {ld}, and after permuting by {j-1}, the new logit difference is {ld2=}"
    )
