#%%
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

from functools import partial

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")


#%% [markdown]
# # <h1><b>Setup</b></h1>
# Import model and dataset
#%% # plot writing in the IO - S direction
model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")

print_gpu_mem("About to load model")
model = EasyTransformer(
    model_name, use_attn_result=True
)  # use_attn_result adds a hook blocks.{lay}.attn.hook_result that is before adding the biais of the attention layer
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
# %% [markdown]
# Each prompts is a dictionnary containing 'IO', 'S' and the "text", the sentence that will be given to the model.
# The prompt type can be "ABBA", "BABA" or "mixed" (half of the previous two) depending on the pattern you want to study


# %%
# IOI Dataset initialisation
N = 400
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)

# %%
# webtext = load_dataset("stas/openwebtext-10k")
# owb_seqs = [
#     "".join(show_tokens(webtext["train"]["text"][i][:2000], model, return_list=True)[: ioi_dataset.max_len])
#     for i in range(ioi_dataset.N)
# ]

#%%

from ioi_circuit_extraction import (
    join_lists,
    CIRCUIT,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
)


def logit_diff(model, ioi_dataset, all=False, std=False):
    """
    Difference between the IO and the S logits at the "to" token
    """
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach()
    L = len(text_prompts)
    IO_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"][:L],
        ioi_dataset.io_tokenIDs[:L],
    ]
    S_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"][:L],
        ioi_dataset.s_tokenIDs[:L],
    ]

    if all and not std:
        return (IO_logits - S_logits).detach().cpu()
    if std:
        if all:
            first_bit = (IO_logits - S_logits).detach().cpu()
        else:
            first_bit = (IO_logits - S_logits).mean().detach().cpu()
        return first_bit, torch.std(IO_logits - S_logits).detach().cpu()
    return (IO_logits - S_logits).mean().detach().cpu()


#%% [markdown]
# TODO Explain the way we're doing Jacob's circuit extraction experiment here
#%%
if False:
    circuit_perf = []

    for G in list(CIRCUIT.keys()) + ["none"]:
        if G == "ablation":
            continue
        print_gpu_mem(G)
        # compute METRIC( C \ G )
        # excluded_classes = ["negative"]
        excluded_classes = []
        if G != "none":
            excluded_classes.append(G)
        heads_to_keep = get_heads_circuit(ioi_dataset, excluded_classes=excluded_classes)  # TODO check the MLP stuff

        model, _ = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
        )
        ldiff_broken_circuit, std_broken_circuit = logit_diff(model, ioi_dataset, std=True, all=True)
        # metric(C\G)
        # adding back the whole model

        excl_class = list(CIRCUIT.keys())
        if G != "none":
            excl_class.remove(G)
        G_heads_to_remove = get_heads_circuit(ioi_dataset, excluded_classes=excl_class)  # TODO check the MLP stuff

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_remove=G_heads_to_remove,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
        )
        ldiff_cobble, std_cobble_circuit = logit_diff(model, ioi_dataset, std=True, all=True)
        # metric(M\G)

        for i in range(len(ldiff_cobble)):
            circuit_perf.append(
                {
                    "removed_group": G,
                    "ldiff_broken": ldiff_broken_circuit[i],
                    "ldiff_cobble": ldiff_cobble[i],
                    "sentence": ioi_dataset.text_prompts[i],
                    "template": ioi_dataset.templates_by_prompt[i],
                }
            )
    circuit_perf = pd.DataFrame(circuit_perf)

    # %%

    # by points
    fig = px.scatter(
        circuit_perf,
        x="ldiff_broken",
        y="ldiff_cobble",
        hover_data=["sentence", "template"],
        color="removed_group",
        opacity=1.0,
    )

    fig.update_layout(
        shapes=[
            # adds line at y=5
            dict(
                type="line",
                xref="x",
                x0=-2,
                x1=12,
                yref="y",
                y0=-2,
                y1=12,
            )
        ]
    )
    fig.show()

    # by sets
    perf_by_sets = []
    for i in range(len(CIRCUIT) + 1):

        perf_by_sets.append(
            {
                "removed_group": circuit_perf.iloc[i * ioi_dataset.N].removed_group,
                "mean_ldiff_broken": circuit_perf.iloc[i * ioi_dataset.N : (i + 1) * ioi_dataset.N].ldiff_broken.mean(),
                "mean_ldiff_cobble": circuit_perf.iloc[i * ioi_dataset.N : (i + 1) * ioi_dataset.N].ldiff_cobble.mean(),
                "std_ldiff_broken": circuit_perf.iloc[i * ioi_dataset.N : (i + 1) * ioi_dataset.N].ldiff_broken.std(),
                "std_ldiff_cobble": circuit_perf.iloc[i * ioi_dataset.N : (i + 1) * ioi_dataset.N].ldiff_cobble.std(),
            }
        )
    df_perf_by_sets = pd.DataFrame(perf_by_sets)
    fig = px.scatter(
        perf_by_sets,
        x="mean_ldiff_broken",
        y="mean_ldiff_cobble",
        color="removed_group",
        error_x="std_ldiff_broken",
        error_y="std_ldiff_cobble",
    )

    fig.update_layout(
        shapes=[
            # adds line at y=5
            dict(
                type="line",
                xref="x",
                x0=0,
                x1=6,
                yref="y",
                y0=0,
                y1=6,
            )
        ]
    )
#
# %% gready circuit breaking


def get_heads_from_nodes(nodes, ioi_dataset):
    heads_to_keep_tok = {}
    for h, t in nodes:
        if h not in heads_to_keep_tok:
            heads_to_keep_tok[h] = []
        if t not in heads_to_keep_tok[h]:
            heads_to_keep_tok[h].append(t)

    heads_to_keep = {}
    for h in heads_to_keep_tok:
        heads_to_keep[h] = get_extracted_idx(heads_to_keep_tok[h], ioi_dataset)

    return heads_to_keep


def circuit_from_nodes_logit_diff(model, ioi_dataset, nodes):
    """Take a list of nodes, return the logit diff of the circuit described by the nodes"""
    heads_to_keep = get_heads_from_nodes(nodes, ioi_dataset)
    # print(heads_to_keep)
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    return logit_diff(model, ioi_dataset, all=False)


def circuit_from_heads_logit_diff(model, ioi_dataset, heads_to_rmv=None, heads_to_kp=None, all=False):
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_kp,
        heads_to_remove=heads_to_rmv,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    return logit_diff(model, ioi_dataset, all=all)


def compute_cobble_broken_diff(model, ioi_dataset, nodes):  # red teaming the circuit by trying
    """ "Compute |Metric(C\ nodes) - Metric(M\ nodes)|"""
    nodes_to_keep = ALL_NODES.copy()
    for n in nodes:
        nodes_to_keep.remove(n)  # C\nodes
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=get_heads_from_nodes(nodes_to_keep, ioi_dataset),
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    ldiff_broken = logit_diff(model, ioi_dataset, all=False)  # Metric(C\nodes)

    model.reset_hooks()

    model, _ = do_circuit_extraction(
        model=model,
        heads_to_remove=get_heads_from_nodes(nodes, ioi_dataset),  # M\nodes
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    ldiff_cobble = logit_diff(model, ioi_dataset, all=False)  # Metric(C\nodes)

    return np.abs(ldiff_broken - ldiff_cobble)


def greed_search_max_broken(get_circuit_logit_diff):
    """Geed search to find G that minimizes metric(C\G). Return a list of node sets."""
    NODES_PER_STEP = 10
    NB_SETS = 5
    NB_ITER = 10
    current_nodes = ALL_NODES.copy()
    all_sets = []
    all_node_baseline = get_circuit_logit_diff(ALL_NODES)

    for step in range(NB_SETS):
        current_nodes = ALL_NODES.copy()
        nodes_removed = []
        baseline = all_node_baseline

        for iter in range(NB_ITER):
            to_test = rd.sample(current_nodes, NODES_PER_STEP)

            results = []
            for node in to_test:  # check wich heads in to_test causes the biggest drop
                circuit_minus_node = current_nodes.copy()
                circuit_minus_node.remove(node)
                results.append(get_circuit_logit_diff(circuit_minus_node))

            diff_to_baseline = [(results[i] - baseline) for i in range(len(results))]
            best_node_idx = np.argmin(diff_to_baseline)

            best_node = to_test[best_node_idx]
            current_nodes.remove(best_node)  # we remove the best node from the circuit
            nodes_removed.append(best_node)

            if iter > NB_ITER // 2 - 1:  # we begin to save the sets after half of the iterations
                all_sets.append({"circuit_nodes": current_nodes.copy(), "removed_nodes": nodes_removed.copy()})

            print(f"iter: {iter} - best node:{best_node} - drop:{min(diff_to_baseline)} - baseline:{baseline}")
            print_gpu_mem(f"iter {iter}")
            baseline = results[best_node_idx]  # new baseline for the next iteration
    return all_sets


def greed_search_max_brok_cob_diff(
    get_cob_brok_from_nodes, init_set=[], NODES_PER_STEP=10, NB_SETS=5, NB_ITER=10, verbose=True
):
    """Geed search to find G that maximize the difference between broken and cobbled circuit |metric(C\G) - metric(M\G)| . Return a list of node sets."""
    all_sets = []

    neg_head_in_G = False
    if neg_head_in_G:
        init_set = list(set(init_set) + set([((10, 7), "end"), ((11, 10), "end")]))

    all_node_baseline = get_cob_brok_from_nodes(init_set)  # |metric(C) - metric(M)|

    C_minus_G_init = ALL_NODES.copy()
    for n in init_set:
        C_minus_G_init.remove(n)

    for step in range(NB_SETS):

        C_minus_G = C_minus_G_init.copy()
        G = init_set.copy()

        old_diff = all_node_baseline

        for iter in range(NB_ITER):
            to_test = rd.sample(C_minus_G, NODES_PER_STEP)

            results = []
            for node in to_test:  # check wich heads in to_test causes the biggest drop
                G_plus_node = G.copy()
                G_plus_node.append(node)
                results.append(get_cob_brok_from_nodes(G_plus_node))

            best_node_idx = np.argmax(results)
            max_diff = results[best_node_idx]
            if max_diff > old_diff:
                best_node = to_test[best_node_idx]
                C_minus_G.remove(best_node)  # we remove the best node from the circuit
                G.append(best_node)
                old_diff = max_diff

                if iter > NB_ITER // 2 - 1:  # we begin to save the sets after half of the iterations
                    all_sets.append({"circuit_nodes": C_minus_G.copy(), "removed_nodes": G.copy()})
                if verbose:
                    print(
                        f"iter: {iter} - best node:{best_node} - max brok cob diff:{max(results)} - baseline:{all_node_baseline}"
                    )
                    print_gpu_mem(f"iter {iter}")
    return all_sets


# /!\ if the dataset is too small, the mean by template will contain name information !!! -> reduce the number of templates for small N
model.reset_hooks()
small_ioi_dataset = IOIDataset(prompt_type="mixed", N=40, tokenizer=model.tokenizer, nb_templates=2)


ALL_NODES = []  # a node is a tuple (head, token)
for h in RELEVANT_TOKENS:
    for tok in RELEVANT_TOKENS[h]:
        ALL_NODES.append((h, tok))

# find G tht minimizes metric(C\G)

# %% Run experiment

greedy_heuristic = "max_brok"

assert greedy_heuristic in ["max_brok", "max_brok_cob_diff"]

if greedy_heuristic == "max_brok":
    nodes_logit_diff_small_data = partial(circuit_from_nodes_logit_diff, model, small_ioi_dataset)
    all_sets_max_brok = greed_search_max_broken(nodes_logit_diff_small_data)
    title_suffix = "min metric(C\G) "
    all_sets = all_sets_max_brok.copy()


if greedy_heuristic == "max_brok_cob_diff":
    # find G tht maximizes |metric(C\G) - metric(M\G)|
    nodes_cob_brok_diff_small_data = partial(compute_cobble_broken_diff, model, small_ioi_dataset)
    all_set_max_brok_cob_diff = greed_search_max_brok_cob_diff(nodes_cob_brok_diff_small_data)
    title_suffix = "max |metric(C\G) - metric(M\G)| "

    ## Choose wich set to plot
    all_sets = all_set_max_brok_cob_diff.copy()

print(f"{len(all_sets)} sets found")


all_sets = [
    {"circuit_nodes": ALL_NODES.copy(), "removed_nodes": []}
] + all_sets  # we add the first set to be the empty set
nb_sets = len(all_sets)
# %% evaluate the sets on the big dataset
circuit_perf_greedy = []

for set_id, nodes_set in enumerate(all_sets):

    logit_diff_broken = circuit_from_heads_logit_diff(  # note set contains circuit_nodes (C\G) and removed_nodes (G)
        model, ioi_dataset, heads_to_kp=get_heads_from_nodes(nodes_set["circuit_nodes"], ioi_dataset), all=True
    )

    print_gpu_mem(f"first extraction {set_id}")

    logit_diff_cobble = circuit_from_heads_logit_diff(  # note set contains circuit_nodes (C\G) and removed_nodes (G)
        model, ioi_dataset, heads_to_rmv=get_heads_from_nodes(nodes_set["removed_nodes"], ioi_dataset), all=True
    )

    print_gpu_mem(f"set_id {set_id}")
    set_name = f"Set {str(set_id)}" if set_id > 0 else "Empty set"
    for i in range(len(logit_diff_cobble)):
        circuit_perf_greedy.append(
            {
                "removed_set_id": set_name,
                "ldiff_broken": logit_diff_broken[i],
                "ldiff_cobble": logit_diff_cobble[i],
                "sentence": ioi_dataset.text_prompts[i],
                "template": ioi_dataset.templates_by_prompt[i],
            }
        )

df_circuit_perf_greedy = pd.DataFrame(circuit_perf_greedy)

# %%


perf_greedy_by_sets = []
for i in range(len(all_sets)):
    set_name = f"Set {str(i)}" if i > 0 else "Empty set"

    perf_greedy_by_sets.append(
        {
            "removed_set_id": set_name,
            "mean_ldiff_broken": df_circuit_perf_greedy.iloc[
                i * ioi_dataset.N : (i + 1) * ioi_dataset.N
            ].ldiff_broken.mean(),
            "mean_ldiff_cobble": df_circuit_perf_greedy.iloc[
                i * ioi_dataset.N : (i + 1) * ioi_dataset.N
            ].ldiff_cobble.mean(),
            "std_ldiff_broken": df_circuit_perf_greedy.iloc[
                i * ioi_dataset.N : (i + 1) * ioi_dataset.N
            ].ldiff_broken.std(),
            "std_ldiff_cobble": df_circuit_perf_greedy.iloc[
                i * ioi_dataset.N : (i + 1) * ioi_dataset.N
            ].ldiff_cobble.std(),
        }
    )
df_perf_greedy_by_sets = pd.DataFrame(perf_greedy_by_sets)

# %% plot the results

## by points
fig = px.scatter(
    df_circuit_perf_greedy,
    title=f"Logit diff per sample constructed by greedy search on {title_suffix} ({nb_sets} sets)",
    x="ldiff_broken",
    y="ldiff_cobble",
    color="removed_set_id",
    hover_data=["sentence", "template"],
    opacity=0.7,
    error_x="std_on_diagonal",
    error_y="std_off_diagonal",
)

min_xy = min(df_circuit_perf_greedy.ldiff_broken.min(), df_circuit_perf_greedy.ldiff_cobble.min()) - 0.1
max_xy = max(df_circuit_perf_greedy.ldiff_broken.max(), df_circuit_perf_greedy.ldiff_cobble.max()) + 0.1


fig.update_layout(
    shapes=[
        # adds line at y=5
        dict(
            type="line",
            xref="x",
            x0=min_xy,
            x1=max_xy,
            yref="y",
            y0=min_xy,
            y1=max_xy,
        )
    ]
)

fig.show()

# %% by sets
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_perf_greedy_by_sets.mean_ldiff_broken,
        y=df_perf_greedy_by_sets.mean_ldiff_cobble,
        hovertext=df_perf_greedy_by_sets.removed_set_id,
        mode="markers",
        error_y=dict(
            type="data",
            array=df_perf_greedy_by_sets.std_ldiff_broken,
            color="rgba(0, 128, 0, 0.5)",
            thickness=0.5,
        ),
        error_x=dict(
            type="data",
            array=df_perf_greedy_by_sets.std_ldiff_cobble,
            color="rgba(0, 128, 0, 0.5)",
            thickness=0.5,
        ),
        marker=dict(size=8),
    )
)


fig.add_annotation(
    x=df_perf_greedy_by_sets.mean_ldiff_broken[0],
    y=df_perf_greedy_by_sets.mean_ldiff_cobble[0],
    text="Empty set",
    showarrow=False,
    yshift=10,
)

fig.update_layout(
    title=f"Mean logit diff by set G constructed by greedy search on {title_suffix} ({nb_sets} sets)",
    xaxis_title="logit diff broken",
    yaxis_title="logit diff cobble",
)

min_xy = min(df_perf_greedy_by_sets.mean_ldiff_broken.min(), df_perf_greedy_by_sets.mean_ldiff_cobble.min()) - 0.1
max_xy = max(df_perf_greedy_by_sets.mean_ldiff_broken.max(), df_perf_greedy_by_sets.mean_ldiff_cobble.max()) + 0.1


fig.update_layout(
    shapes=[
        # adds line at y=5
        dict(
            type="line",
            xref="x",
            x0=min_xy,
            x1=max_xy,
            yref="y",
            y0=min_xy,
            y1=max_xy,
        )
    ]
)
fig.show()


# %% greedy minimality experiments

model.reset_hooks()
small_ioi_dataset = IOIDataset(prompt_type="mixed", N=30, tokenizer=model.tokenizer, nb_templates=2)


def test_minimality(model, ioi_dataset, v, J):
    """Compute |Metric( (C\J) U {v}) - Metric(C\J)| where J is a list of nodes, v is a node"""
    C_minus_J = list(set(ALL_NODES.copy()) - set(J.copy()))

    LD_C_m_J = circuit_from_nodes_logit_diff(model, ioi_dataset, C_minus_J)  # metric(C\J)
    C_minus_J_plus_v = set(C_minus_J.copy())
    C_minus_J_plus_v.add(v)
    C_minus_J_plus_v = list(C_minus_J_plus_v)

    LD_C_m_J_plus_v = circuit_from_nodes_logit_diff(model, ioi_dataset, C_minus_J_plus_v)  # metric( (C\J) U {v})
    return np.abs(LD_C_m_J - LD_C_m_J_plus_v)


best_J = {}  # list of candidate sets for each node
best_scores = {}  # list of scores for each node
for v in tqdm(ALL_NODES):
    minimality_test_v = partial(test_minimality, model, small_ioi_dataset, v)
    best_J[v] = greed_search_max_brok_cob_diff(
        minimality_test_v, init_set=[v], NODES_PER_STEP=5, NB_SETS=1, NB_ITER=10, verbose=False
    )
    if len(best_J[v]) == 0:  # if the greedy search did not find any set, we use the set with the node itself
        all_but_v = ALL_NODES.copy()
        all_but_v.remove(v)
        best_J[v] = [{"circuit_nodes": all_but_v, "removed_nodes": [v]}]

ioi_dataset = IOIDataset(prompt_type="mixed", N=200, tokenizer=model.tokenizer)

for v in tqdm(ALL_NODES):  # validate the best sets
    minimality_scores = [test_minimality(model, ioi_dataset, v, node_set["removed_nodes"]) for node_set in best_J[v]]
    best_J[v] = best_J[v][np.argmax(minimality_scores)]
    best_scores[v] = np.max(minimality_scores)
    print(f"v={v}, J={best_J[v]}, score={best_scores[v]}")


for v, J in best_J.items():
    print(f"v={v}, score={best_scores[v]}")


# %%

head_classes = []
for h, tok in best_scores.keys():
    for group in CIRCUIT:
        if h in CIRCUIT[group]:
            head_classes.append(group)
            break


px.bar(x=list(best_scores.values()), y=[str(k) for k in best_scores.keys()], orientation="h")


# %%
