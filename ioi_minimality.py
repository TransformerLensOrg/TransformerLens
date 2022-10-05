#%%
import warnings
from time import ctime
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from ioi_utils import probs
from interp.circuit.projects.ioi.ioi_methods import ablate_layers, get_logit_diff
from ioi_utils import probs, logit_diff
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
    ABBA_TEMPLATES,
)
from ioi_utils import (
    ALL_COLORS,
    CLASS_COLORS,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
)
from ioi_circuit_extraction import (
    ALEX_NAIVE,
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
model = EasyTransformer(model_name, use_attn_result=True)
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts("S2")
mean_dataset = abca_dataset

#%% # do some initial experiments with the naive circuit
<<<<<<< HEAD

CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (11, 9),
        (9, 7),
        (11, 3),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (2, 9), (4, 11)],
}

=======
# UH         IS THIS JUST NOT GOOD?
>>>>>>> 43eeb1a3ec59b30098261fb8749d97b3b6911b29
circuits = [None, CIRCUIT.copy(), ALEX_NAIVE.copy()]
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
    exclude_heads=naive_heads,
)

circuit_baseline_metric = metric(model, ioi_dataset)
print(f"{model_baseline_metric=} {circuit_baseline_metric=}")
#%%
def get_basic_extracted_model(model, ioi_dataset, mean_dataset=None, circuit=circuits[1]):
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
    circuit=circuits[2],
)
torch.cuda.empty_cache()

circuit_baseline_diff, circuit_baseline_diff_std = logit_diff(model, ioi_dataset, std=True)
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

if metric == logit_diff:
    circuit_baseline_metric = circuit_baseline_diff
else:
    circuit_baseline_metric = circuit_baseline_prob

circuit_baseline_metric = [None, circuit_baseline_metric, circuit_baseline_metric]
#%% # assemble the J
J = {}

for circuit_class in circuit.keys():
    for head in circuit[circuit_class]:
        J[head] = [circuit_class]
J[(5, 8)] = [(5, 8)]
J[(5, 9)] = [(5, 9)]  
for i, head in enumerate(circuit["induction"]):
    J[head] += [(10, 7), (11, 10)]

# rebuild J
for head in J.keys():
    new_j_entry = []
    for entry in J[head]:
        if isinstance(entry, str):
            for head2 in circuits[1][entry]:
                new_j_entry.append(head2)
        elif isinstance(entry, tuple):
            new_j_entry.append(entry)
        else:
            raise NotImplementedError(head, entry)
    assert head in new_j_entry, (head, new_j_entry)
    J[head] = list(set(new_j_entry))

# name mover shit
for i, head in enumerate(circuit["name mover"]):
    old_entry = J[head]
<<<<<<< HEAD
    J[head] = []
    for other_head in circuit["name mover"][: i + 1]:
        J[head].append(other_head)

# for i, head in enumerate(circuit["name mover"]):
# J[head] += [(10, 7), (11, 10)]

for i, head in enumerate(circuit["induction"]):
    J[head] += [(10, 7), (11, 10)]

# J[(9, 6)] = [(9, 9), (9, 6)]
# J[(10, 10)] = [(9, 9), (10, 10)]
J[(11, 3)] = [(9, 9), (10, 0), (9, 6), (10, 10), (11, 3)]  # by importance
#%%
=======
    J[head] = deepcopy(circuit["name mover"][: i + 1]) # turn into the previous things
#%% 
>>>>>>> 43eeb1a3ec59b30098261fb8749d97b3b6911b29
results = {}

if "results_cache" not in dir():
    results_cache = {}  # massively speeds up future runs

for circuit_class in circuit.keys():
    for head in circuits[1][circuit_class]:
        results[head] = [None, None]
        base = frozenset(J[head])
        summit_list = deepcopy(J[head])
        summit_list.remove(head) # and this will error if you don't have a head in J!!!
        summit = frozenset(summit_list)

        for idx, ablated_stuff in enumerate([base, summit]):
            if ablated_stuff not in results_cache:  # see the if False line
                new_heads_to_keep = get_heads_circuit(ioi_dataset, excluded=ablated_stuff, circuit=circuit)
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

        print(f"{head=} with {J[head]=}: progress from {results[head][0]} to {results[head][1]}")
#%%
ac = ALL_COLORS
cc = CLASS_COLORS.copy()

relevant_classes = list(circuit.keys())
fig = go.Figure()

initial_y_cache = {}
final_y_cache = {}

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

    for v in curvys:  # i, v in enumerate(list(circuit[G])):
        # if G == "name mover":
        #     if v in [(9, 9), (9, 6), (10, 0)]:
        #         widths.append(1)
        #         names.append("name mover")
        #         colors.append(cc[G])
        #     else:
        #         widths.append(0.2)
        #         names.append("dis")
        #         colors.append("rgb(27,100,119)")
        # else:
        #     widths.append(1)
        #     colors.append(cc[G])
        #     names.append(G)
        colors.append(cc[G])
        xs.append(str(v))
        initial_y = results[v][0]
        final_y = results[v][1]

        initial_ys.append(initial_y)
        final_ys.append(final_y)

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
            width=[1.0 for _ in range(len(xs))],  ## if G != "dis" else [0.2 for _ in range(len(xs))],
            name=G,
        )
    )


fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

all_vs = []

# for circuit_class in relevant_classes:
#     vs = circuit[circuit_class]
#     vs_str = [str(v) for v in vs]
#     all_vs.extend(vs_str)
#     fig.add_trace(
#         go.Scatter(
#             x=vs_str,
#             y=[results[circuit_class]["metric_calc"] for _ in range(len(vs))],
#             line=dict(color=cc[circuit_class]),
#             name=circuit_class,
#         )
#     )

fig.add_trace(
    go.Scatter(
        x=all_vs,
        y=[(baseline_prob if metric == probs else baseline_ldiff) for _ in range(len(all_vs))],
        name="Baseline model performance",
        line=dict(color="black"),
        fill="toself",
        mode="lines",
    )
)

fig.add_trace(
    go.Scatter(
        x=all_vs,
        y=[circuit_baseline_metric[circuit_idx] for _ in range(len(all_vs))],
        name="Circuit performance",
        line=dict(color="blue"),
        fill="toself",
        mode="lines",
    )
)

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
#%% # THIS IS JUST FOR LATEX

assert False


def capitalise(name):
    """
    turn each word into a capitalised word
    """
    return " ".join([word.capitalize() for word in name.split(" ")])


idx = 0
for j, G in enumerate(relevant_classes + ["backup name mover"]):
    initial_ys = initial_y_cache[G]
    final_ys = final_y_cache[G]
    for i in range(len(initial_ys)):  ## , v in enumerate(list(circuit[G])):
        head = circuit[G][i] if G != "backup name mover" else circuit["name mover"][i]
        group = str(G)
        start = initial_ys[i]
        end = final_ys[i]
        name = capitalise(group)
        if name == "S2 Inhibition":
            name = "S Inhibition"
        K = J[head]
        if G == "backup name mover":
            K = "All previous NMs and distributed NMs"
        print(f"{head} & {name} & {K} & {start:.2f} & {end:.2f} \\\\")
        print("\\hline")
        idx += 1
#%%
fig.add_shape(
    type="line",
    xref="x",
    x0=-0.50,
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
circuit = ALEX_NAIVE.copy()
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
            excluded=["previous token"],  # , "duplicate token"],
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
#%% # check that really ablate 9.9+9.6 is worse than just 9.9 ???
for poppers in [[(9, 9)], [(9, 9), (9, 6)], [(9, 6)]]:
    new_heads_to_keep = get_heads_circuit(ioi_dataset, excluded=[], circuit=circuit)

    for popper in poppers:
        new_heads_to_keep.pop(popper)

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=new_heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    metric_calc = metric(model, ioi_dataset, std=True)
    torch.cuda.empty_cache()
    print(metric_calc)
#%% # new experiment idea: the duplicators and induction heads shouldn't care where their attention is going, provided that
# it goes to either S or S+1.

for j in range(2, 4):
    s_positions = ioi_dataset.word_idx["S"]

    # [batch, head_index, query_pos, key_pos] # so pass dim=1 to ignore the head
    def attention_pattern_modifier(z, hook):  # batch, seq, head dim, because get_act_hook hides scary things from us
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_attn", hook.name
        assert len(list(z.shape)) == 3, z.shape  # batch, seq (attending_query), attending_key

        prior_stuff = []

        for i in range(-1, 2):
            prior_stuff.append(z[:, s_positions + i, :].clone())
            # print(prior_stuff[-1].shape, torch.sum(prior_stuff[-1]))

        for i in range(-1, 2):
            z[:, s_positions + i, :] = prior_stuff[(i + j) % 3]  # +1 is the do nothing one

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
# %%
