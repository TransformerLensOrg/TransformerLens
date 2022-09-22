#%%
import warnings
from time import ctime
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
    gen_flipped_prompts,
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
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts("S2")

# %%
# webtext = load_dataset("stas/openwebtext-10k")
# owb_seqs = [
#     "".join(show_tokens(webtext["train"]["text"][i][:2000], model, return_list=True)[: ioi_dataset.max_len])
#     for i in range(ioi_dataset.N)
# ]

#%%

from ioi_circuit_extraction import (
    ARTHUR_CIRCUIT,
    join_lists,
    CIRCUIT,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
)

alex_greedy_things = False
old_circuit = True

if alex_greedy_things:
    print("WARINING: USING OLD CIRCUIT")

    CIRCUIT = {
        "name mover": [
            (9, 6),  # ori
            (9, 9),  # ori
            (10, 0),  # ori
        ],  # , (10, 10), (10, 6)],  # 10, 10 and 10.6 weak nm
        "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
        "duplicate token": [(1, 11), (0, 10), (3, 0)],
    }
    RELEVANT_TOKENS = {}
    for head in CIRCUIT["name mover"] + CIRCUIT["s2 inhibition"]:
        RELEVANT_TOKENS[head] = ["end"]

    for head in CIRCUIT["duplicate token"]:
        RELEVANT_TOKENS[head] = ["S2"]

    ALL_NODES = []  # a node is a tuple (head, token)
    for h in RELEVANT_TOKENS:
        for tok in RELEVANT_TOKENS[h]:
            ALL_NODES.append((h, tok))

else:
    circuit = CIRCUIT.copy()


def logit_diff(model, ioi_dataset, logits=None, all=False, std=False):
    """
    Difference between the IO and the S logits at the "to" token
    """
    if logits is None:
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


## define useful function


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


def circuit_from_heads_logit_diff(
    model, ioi_dataset, heads_to_rmv=None, heads_to_kp=None, all=False
):
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_kp,
        heads_to_remove=heads_to_rmv,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    return logit_diff(model, ioi_dataset, all=all)


def compute_cobble_broken_diff(
    model, ioi_dataset, nodes
):  # red teaming the circuit by trying
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

            if (
                iter > NB_ITER // 2 - 1
            ):  # we begin to save the sets after half of the iterations
                all_sets.append(
                    {
                        "circuit_nodes": current_nodes.copy(),
                        "removed_nodes": nodes_removed.copy(),
                    }
                )

            print(
                f"iter: {iter} - best node:{best_node} - drop:{min(diff_to_baseline)} - baseline:{baseline}"
            )
            print_gpu_mem(f"iter {iter}")
            baseline = results[best_node_idx]  # new baseline for the next iteration
    return all_sets


def test_minimality(model, ioi_dataset, v, J, absolute=True):
    """Compute |Metric( (C\J) U {v}) - Metric(C\J)| where J is a list of nodes, v is a node"""
    C_minus_J = list(set(ALL_NODES.copy()) - set(J.copy()))

    LD_C_m_J = circuit_from_nodes_logit_diff(
        model, ioi_dataset, C_minus_J
    )  # metric(C\J)
    C_minus_J_plus_v = set(C_minus_J.copy())
    C_minus_J_plus_v.add(v)
    C_minus_J_plus_v = list(C_minus_J_plus_v)

    LD_C_m_J_plus_v = circuit_from_nodes_logit_diff(
        model, ioi_dataset, C_minus_J_plus_v
    )  # metric( (C\J) U {v})
    if absolute:
        return np.abs(LD_C_m_J - LD_C_m_J_plus_v)
    else:
        return LD_C_m_J - LD_C_m_J_plus_v


def greed_search_max_brok_cob_diff(
    get_cob_brok_from_nodes,
    init_set=[],
    NODES_PER_STEP=10,
    NB_SETS=5,
    NB_ITER=10,
    verbose=True,
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

            to_test = rd.sample(C_minus_G, min(NODES_PER_STEP, len(C_minus_G)))

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

                if (
                    iter > NB_ITER // 2 - 1
                ):  # we begin to save the sets after half of the iterations
                    all_sets.append(
                        {"circuit_nodes": C_minus_G.copy(), "removed_nodes": G.copy()}
                    )
                if verbose:
                    print(
                        f"iter: {iter} - best node:{best_node} - max brok cob diff:{max(results)} - baseline:{all_node_baseline}"
                    )
                    print_gpu_mem(f"iter {iter}")
    return all_sets


#%% [markdown]
# TODO Explain the way we're doing Jacob's circuit extraction experiment here


#%% [markdown]
# # <h1><b>Setup</b></h1>
# Import model and dataset
#%% # plot writing in the IO - S direction

if __name__ == "__main__":

    model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")

    print_gpu_mem("About to load model")
    model = EasyTransformer(
        model_name, use_attn_result=True
    )  # use_attn_result adds a hook blocks.{lay}.attn.hook_result that is before adding the biais of the attention layer
    device = "cuda"
    if torch.cuda.is_available():
        model.to(device)
    print_gpu_mem("Gpt2 loaded")

    # IOI Dataset initialisation
    N = 400
    ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
# %%
# webtext = load_dataset("stas/openwebtext-10k")
# owb_seqs = [
#     "".join(show_tokens(webtext["train"]["text"][i][:2000], model, return_list=True)[: ioi_dataset.max_len])
#     for i in range(ioi_dataset.N)
# ]

run_original = True

if __name__ != "__main__":
    run_original = False

# %%
circuit = CIRCUIT.copy()
cur_metric = logit_diff  # partial(probs, type="s")
mean_dataset = abca_dataset

run_original = True
if run_original:
    circuit_perf = []
    for G in list(circuit.keys()) + ["none"]:
        if G == "ablation":
            continue
        print_gpu_mem(G)
        # compute METRIC( C \ G )
        # excluded_classes = ["negative"]
        excluded_classes = []
        if G != "none":
            excluded_classes.append(G)
        heads_to_keep = get_heads_circuit(
            ioi_dataset, excluded_classes=excluded_classes, circuit=circuit
        )  # TODO check the MLP stuff

        model, _ = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=mean_dataset,
        )
        torch.cuda.empty_cache()
        cur_metric_broken_circuit, std_broken_circuit = cur_metric(
            model, ioi_dataset, std=True, all=True
        )
        torch.cuda.empty_cache()
        # metric(C\G)
        # adding back the whole model

        excl_class = list(circuit.keys())
        if G != "none":
            excl_class.remove(G)
        G_heads_to_remove = get_heads_circuit(
            ioi_dataset, excluded_classes=excl_class, circuit=circuit
        )  # TODO check the MLP stuff
        torch.cuda.empty_cache()

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_remove=G_heads_to_remove,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=mean_dataset,
        )
        torch.cuda.empty_cache()
        cur_metric_cobble, std_cobble_circuit = cur_metric(
            model, ioi_dataset, std=True, all=True
        )
        print(cur_metric_cobble.mean(), cur_metric_broken_circuit.mean())
        torch.cuda.empty_cache()
        # metric(M\G)

        for i in range(len(cur_metric_cobble)):
            circuit_perf.append(
                {
                    "removed_group": G,
                    "cur_metric_broken": cur_metric_broken_circuit[i],
                    "cur_metric_cobble": cur_metric_cobble[i],
                    "sentence": ioi_dataset.text_prompts[i],
                    "template": ioi_dataset.templates_by_prompt[i],
                }
            )
    circuit_perf = pd.DataFrame(circuit_perf)
# %%
show_scatter = True
circuit_perf_scatter = []

# by points
if run_original:
    if show_scatter:
        fig = px.scatter(
            circuit_perf,
            x="cur_metric_broken",
            y="cur_metric_cobble",
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

        fig.update_xaxes(gridcolor="black", gridwidth=0.1)
        fig.update_yaxes(gridcolor="black", gridwidth=0.1)
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        fig.write_image(f"svgs/circuit_completeness_at_{ctime()}.svg")

        fig.show()

    # by sets
    perf_by_sets = []
    for i in range(len(circuit) + 1):
        cur_metric_brokens = circuit_perf.iloc[
            i * ioi_dataset.N : (i + 1) * ioi_dataset.N
        ].cur_metric_broken

        cur_metric_cobbles = circuit_perf.iloc[
            i * ioi_dataset.N : (i + 1) * ioi_dataset.N
        ].cur_metric_cobble

        perf_by_sets.append(
            {
                "removed_group": circuit_perf.iloc[i * ioi_dataset.N].removed_group,
                "mean_cur_metric_broken": cur_metric_brokens.mean(),
                "mean_cur_metric_cobble": cur_metric_cobbles.mean(),
                "std_cur_metric_broken": cur_metric_brokens.std(),
                "std_cur_metric_cobble": cur_metric_cobbles.std(),
            }
        )

        perf_by_sets[-1]["mean_abs_diff"] = abs(
            cur_metric_brokens - cur_metric_cobbles
        ).mean()

    circuit_classes = sorted(perf_by_sets, key=lambda x: -x["mean_abs_diff"])
    print(
        f"The circuit class with maximum difference is {circuit_classes[0]['removed_group']} with difference {circuit_classes[0]['mean_abs_diff']}"
    )

    # plot sets
    df_perf_by_sets = pd.DataFrame(perf_by_sets)
    fig = px.scatter(
        perf_by_sets,
        x="mean_cur_metric_broken",
        y="mean_cur_metric_cobble",
        color="removed_group",
        error_x="std_cur_metric_broken",
        error_y="std_cur_metric_cobble",
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

    fig.update_xaxes(gridcolor="black", gridwidth=0.1)
    fig.update_yaxes(gridcolor="black", gridwidth=0.1)
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    fig.write_image(f"svgs/circuit_completeness_plusses_at_{ctime()}.svg")
    fig.show()
#%%
# plot the covariance ellipsoid
# as in https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

xs = {}
ys = {}
for i, G in enumerate(list(CIRCUIT.keys()) + ["none"]):
    xs[G] = circuit_perf.loc[
        circuit_perf["removed_group"] == G
    ].cur_metric_broken.values
    ys[G] = circuit_perf.loc[
        circuit_perf["removed_group"] == G
    ].cur_metric_cobble.values
    xs[G] = [float(x) for x in xs[G]]
    ys[G] = [float(y) for y in ys[G]]


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if run_original:
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.axvline(c="grey", lw=1)
    ax.axhline(c="grey", lw=1)

    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
    ]
    # the plotly colors

    for i, G in enumerate(list(CIRCUIT.keys()) + ["none"]):
        ax.scatter(list(xs[G]), list(ys[G]), s=5, label=G, c=colors[i])
        confidence_ellipse(
            np.asarray(xs[G]),
            np.asarray(ys[G]),
            ax,
            edgecolor=colors[i],
            n_std=1,
        )

    xs2 = np.asarray(list(range(-100, 700))) / 100
    ys2 = np.asarray(list(range(-100, 700))) / 100
    ax.plot(xs2, ys2)
    ax.legend()
    plt.xlabel("Logit diff of broken circuit")
    plt.ylabel("Logit diff of complement of G")

warnings.warn("Increase x lim if plotting logit diffs not probs")
plt.xlim(-0.01, 0.1)
plt.ylim(-0.01, 0.1)
plt.show()

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
    small_ioi_dataset = IOIDataset(
        prompt_type="mixed", N=40, tokenizer=model.tokenizer, nb_templates=2
    )

    circuit_to_study = "natural_circuit"

    assert circuit_to_study in ["auto_search", "natural_circuit"]

    ALL_NODES_AUTO_SEARCH = [
        ((4, 0), "IO"),
        ((1, 5), "S+1"),
        ((6, 8), "S"),
        ((10, 6), "IO"),
        ((10, 10), "end"),
        ((8, 10), "end"),
        ((9, 2), "S+1"),
        ((5, 3), "and"),
        ((2, 10), "S2"),
        ((10, 4), "S2"),
        ((0, 9), "S"),
        ((7, 8), "S"),
        ((1, 8), "and"),
        ((2, 7), "S2"),
        ((1, 5), "end"),
        ((8, 7), "end"),
        ((7, 0), "S+1"),
    ]

    if circuit_to_study == "auto_search":
        ALL_NODES = ALL_NODES_AUTO_SEARCH.copy()
    elif circuit_to_study == "natural_circuit":
        ALL_NODES = []  # a node is a tuple (head, token)
        for h in RELEVANT_TOKENS:
            for tok in RELEVANT_TOKENS[h]:
                ALL_NODES.append((h, tok))


def circuit_from_heads_logit_diff(
    model, ioi_dataset, heads_to_rmv=None, heads_to_kp=None, all=False
):
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_kp,
        heads_to_remove=heads_to_rmv,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    return logit_diff(model, ioi_dataset, all=all)

    # %% Run experiment


greedy_heuristic = "max_brok_cob_diff"


def compute_cobble_broken_diff(
    model, ioi_dataset, nodes
):  # red teaming the circuit by trying
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

    assert greedy_heuristic in ["max_brok", "max_brok_cob_diff"]

    NODES_PER_STEP = 10
    NB_SETS = 2
    NB_ITER = 5

    if greedy_heuristic == "max_brok":
        nodes_logit_diff_small_data = partial(
            circuit_from_nodes_logit_diff, model, small_ioi_dataset
        )
        all_sets_max_brok = greed_search_max_broken(
            nodes_logit_diff_small_data,
            NODES_PER_STEP=NODES_PER_STEP,
            NB_SETS=NB_SETS,
            NB_ITER=NB_ITER,
        )
        title_suffix = "min metric(C\G) "
        all_sets = all_sets_max_brok.copy()

    if greedy_heuristic == "max_brok_cob_diff":
        # find G tht maximizes |metric(C\G) - metric(M\G)|
        nodes_cob_brok_diff_small_data = partial(
            compute_cobble_broken_diff, model, small_ioi_dataset
        )
        all_set_max_brok_cob_diff = greed_search_max_brok_cob_diff(
            nodes_cob_brok_diff_small_data,
            NODES_PER_STEP=NODES_PER_STEP,
            NB_SETS=NB_SETS,
            NB_ITER=NB_ITER,
        )
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
            model,
            ioi_dataset,
            heads_to_kp=get_heads_from_nodes(nodes_set["circuit_nodes"], ioi_dataset),
            all=True,
        )

        print_gpu_mem(f"first extraction {set_id}")

        logit_diff_cobble = circuit_from_heads_logit_diff(  # note set contains circuit_nodes (C\G) and removed_nodes (G)
            model,
            ioi_dataset,
            heads_to_rmv=get_heads_from_nodes(nodes_set["removed_nodes"], ioi_dataset),
            all=True,
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
    )

    min_xy = (
        min(
            df_circuit_perf_greedy.ldiff_broken.min(),
            df_circuit_perf_greedy.ldiff_cobble.min(),
        )
        - 0.1
    )
    max_xy = (
        max(
            df_circuit_perf_greedy.ldiff_broken.max(),
            df_circuit_perf_greedy.ldiff_cobble.max(),
        )
        + 0.1
    )

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

    min_xy = (
        min(
            df_perf_greedy_by_sets.mean_ldiff_broken.min(),
            df_perf_greedy_by_sets.mean_ldiff_cobble.min(),
        )
        - 0.1
    )
    max_xy = (
        max(
            df_perf_greedy_by_sets.mean_ldiff_broken.max(),
            df_perf_greedy_by_sets.mean_ldiff_cobble.max(),
        )
        + 0.1
    )

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
    small_ioi_dataset = IOIDataset(
        prompt_type="mixed", N=30, tokenizer=model.tokenizer, nb_templates=2
    )

    best_J = {}  # list of candidate sets for each node
    best_scores = {}  # list of scores for each node
    for v in tqdm(ALL_NODES):
        minimality_test_v = partial(test_minimality, model, small_ioi_dataset, v)
        best_J[v] = greed_search_max_brok_cob_diff(
            minimality_test_v,
            init_set=[v],
            NODES_PER_STEP=5,
            NB_SETS=1,
            NB_ITER=10,
            verbose=False,
        )
        if (
            len(best_J[v]) == 0
        ):  # if the greedy search did not find any set, we use the set with the node itself
            all_but_v = ALL_NODES.copy()
            all_but_v.remove(v)
            best_J[v] = [{"circuit_nodes": all_but_v, "removed_nodes": [v]}]

    ioi_dataset = IOIDataset(prompt_type="mixed", N=200, tokenizer=model.tokenizer)

    for v in tqdm(ALL_NODES):  # validate the best sets
        minimality_scores = [
            test_minimality(
                model, ioi_dataset, v, node_set["removed_nodes"], absolute=False
            )
            for node_set in best_J[v]
        ]
        best_J[v] = best_J[v][np.argmax(minimality_scores)]
        best_scores[v] = np.max(minimality_scores)
        print(f"v={v}, J={best_J[v]}, score={best_scores[v]}")

    for v, J in best_J.items():
        print(f"v={v}, score={best_scores[v]}")

    # %%
    if circuit_to_study == "natural_circuit":
        head_classes = []
        for h, tok in best_scores.keys():
            for group in CIRCUIT:
                if h in CIRCUIT[group]:
                    head_classes.append(group)
                    break
    else:
        head_classes = ["none" for i in range(len(best_scores))]

    px.bar(
        x=list(best_scores.values()),
        y=[str(k) for k in best_scores.keys()],
        orientation="h",
        color=head_classes,
    )
