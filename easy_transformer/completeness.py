#%%
import os
import torch

from easy_transformer.ioi_circuit_extraction import NAIVE

if os.environ["USER"] in ["exx", "arthur"]:  # so Arthur can safely use octobox
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
assert torch.cuda.device_count() == 1
import json
from statistics import mean
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
import warnings
import json
from numpy import sin, cos, pi
from time import ctime
from dataclasses import dataclass
from easy_transformer.ioi_utils import logit_diff
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
import matplotlib.pyplot as plt


from easy_transformer.ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_flipped_prompts,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    ABBA_TEMPLATES,
)
from easy_transformer.ioi_utils import (
    basis_change,
    add_arrow,
    CLASS_COLORS,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
    plot_ellipse,
    probs,
)
from easy_transformer.ioi_circuit_extraction import (
    join_lists,
    CIRCUIT,
    NAIVE,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
    ALL_NODES,
)
from copy import deepcopy

plotly_colors = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
]

from functools import partial

#%% [markdown]
# # <h1><b>Completeness</b></h1>
# In this notebook, we compute the incompleteness scores for the circuit classes, and use precomputed data to plot the random and greedy incompleteness scores

model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")

print_gpu_mem("About to load model")
model = EasyTransformer.from_pretrained(
    model_name,
)
model.set_use_attn_result(True)
device = "cuda"
if torch.cuda.is_available():
    model.to(device)

print_gpu_mem("Gpt2 loaded")
#%%
# IOI Dataset initialisation

N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)

abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)

#%%


def get_all_nodes(circuit):
    nodes = []
    for circuit_class in circuit:
        for head in circuit[circuit_class]:
            nodes.append((head, RELEVANT_TOKENS[head][0]))
    return nodes


#%% [markdown]
# # <h1><b>Setup</b></h1>
# Import model and dataset

mean_dataset = abc_dataset

#%% [markdown] Do some faithfulness
model.reset_hooks()
logit_diff_M = logit_diff(model, ioi_dataset)
print(f"logit_diff_M: {logit_diff_M}")

for circuit in [CIRCUIT.copy(), NAIVE.copy()]:
    all_nodes = get_all_nodes(circuit)
    heads_to_keep = get_heads_circuit(ioi_dataset, excluded=[], circuit=circuit)
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )

    logit_diff_circuit = logit_diff(model, ioi_dataset)
    print(f"{logit_diff_circuit}")
# %% [markdown] select CIRCUIT or NAIVE in otder to choose between the two circuits studied in the paper. Look at the `perf_by_sets.append` line to see how the results are saved
circuit = deepcopy(NAIVE)
print("Working with", circuit)
cur_metric = logit_diff

run_original = True
print("Are we running the original experiment?", run_original)

if run_original:
    circuit_perf = []
    perf_by_sets = []
    for G in tqdm(list(circuit.keys()) + ["none"]):
        if G == "ablation":
            continue
        print_gpu_mem(G)
        # compute METRIC( C \ G )
        # excluded_classes = ["negative"]
        excluded_classes = []
        if G != "none":
            excluded_classes.append(G)
        heads_to_keep = get_heads_circuit(
            ioi_dataset, excluded=excluded_classes, circuit=circuit
        )  # TODO check the MLP stuff
        model.reset_hooks()
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
            ioi_dataset, excluded=excl_class, circuit=circuit
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
        on_diagonals = []
        off_diagonals = []
        for i in range(len(cur_metric_cobble)):
            circuit_perf.append(
                {
                    "removed_set_id": G,
                    "ldiff_broken": float(cur_metric_broken_circuit[i].cpu().numpy()),
                    "ldiff_cobble": float(cur_metric_cobble[i].cpu().numpy()),
                    "sentence": ioi_dataset.sentences[i],
                    "template": ioi_dataset.templates_by_prompt[i],
                }
            )

            x, y = basis_change(
                circuit_perf[-1]["ldiff_broken"],
                circuit_perf[-1]["ldiff_cobble"],
            )
            circuit_perf[-1]["on_diagonal"] = x
            circuit_perf[-1]["off_diagonal"] = y
            on_diagonals.append(x)
            off_diagonals.append(y)

        perf_by_sets.append(
            {
                "removed_group": G,
                "mean_cur_metric_broken": cur_metric_broken_circuit.mean(),
                "mean_cur_metric_cobble": cur_metric_cobble.mean(),
                "std_cur_metric_broken": cur_metric_broken_circuit.std(),
                "std_cur_metric_cobble": cur_metric_cobble.std(),
                "on_diagonal": np.mean(on_diagonals),
                "off_diagonal": np.mean(off_diagonals),
                "std_on_diagonal": np.std(on_diagonals),
                "std_off_diagonal": np.std(off_diagonals),
                "color": CLASS_COLORS[G],
                "symbol": "diamond-x",
            }
        )

        perf_by_sets[-1]["mean_abs_diff"] = abs(
            perf_by_sets[-1]["mean_cur_metric_broken"]
            - perf_by_sets[-1]["mean_cur_metric_cobble"]
        ).mean()

    df_circuit_perf = pd.DataFrame(circuit_perf)
    circuit_classes = sorted(perf_by_sets, key=lambda x: -x["mean_abs_diff"])
    df_perf_by_sets = pd.DataFrame(perf_by_sets)

with open(f"sets/perf_by_classes_{ctime()}.json", "w") as f:
    json.dump(circuit_perf, f)
#%% [markdown]
# Make the figure of the circuit classes

fig = go.Figure()

## add the grey region
# make the dotted line
minx = -2
maxx = 6
eps = 1.0
xs = np.linspace(minx - 1, maxx + 1, 100)
ys = xs

fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=f"x=y",
        line=dict(color="grey", width=2, dash="dash"),
    )
)

rd_set_added = False
for i, perf in enumerate(perf_by_sets):
    fig.add_trace(
        go.Scatter(
            x=[perf["mean_cur_metric_broken"]],
            y=[perf["mean_cur_metric_cobble"]],
            mode="markers",
            name=perf[
                "removed_group"  # change to "name" or something for the greedy sets
            ],
            marker=dict(symbol=perf["symbol"], size=10, color=perf["color"]),
            showlegend=(
                (" 1" in perf["removed_group"][-2:])
                or ("Set" not in perf["removed_group"])
            ),
        )
    )
    continue


# fig.update_layout(showlegend=False) #

fig.update_xaxes(title_text="F(C \ K)")
fig.update_yaxes(title_text="F(M \ K)")
fig.update_xaxes(showgrid=True, gridcolor="black", gridwidth=1)
fig.update_yaxes(showgrid=True, gridcolor="black", gridwidth=1)
fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

# USE THESE LINES TO SCALE SVGS PROPERLY
fig.update_xaxes(range=[minx, maxx])
fig.update_yaxes(range=[minx, maxx])
fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)
import os

circuit_to_export = "natural"
fpath = f"circuit_completeness_{circuit_to_export}_CIRCUIT_at_{ctime()}.svg"
if os.path.exists(
    "/home/ubuntu/my_env/lib/python3.9/site-packages/easy_transformer/svgs"
):
    fpath = "svgs/" + fpath

fig.write_image(fpath)
fig.show()

#%% [markdown]
# Run a greedy search. Set `skip_greedy` to `True` to skip this step.

skip_greedy = True
do_asserts = False

for doover in range(int(1e9)):
    if skip_greedy:
        break
    for raw_circuit_idx, raw_circuit in enumerate([CIRCUIT, NAIVE]):

        if doover == 0 and raw_circuit_idx == 0:
            print("Starting with the NAIVE!")
            continue

        circuit = deepcopy(raw_circuit)
        all_nodes = get_all_nodes(circuit)
        all_circuit_nodes = [head[0] for head in all_nodes]
        circuit_size = len(all_circuit_nodes)

        # NOTE THESE HOOOKS ARE TOTALLY IOI DATASET DEPENDENT
        # AND CIRCUIT DEPENDENT

        complement_hooks = (
            do_circuit_extraction(  # these are the default edit-all things
                model=model,
                heads_to_keep={},
                mlps_to_remove={},
                ioi_dataset=ioi_dataset,
                mean_dataset=mean_dataset,
                return_hooks=True,
                hooks_dict=True,
            )
        )

        assert len(complement_hooks) == 144

        heads_to_keep = get_heads_from_nodes(all_nodes, ioi_dataset)
        assert len(heads_to_keep) == circuit_size, (
            len(heads_to_keep),
            circuit_size,
        )

        circuit_hooks = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=mean_dataset,
            return_hooks=True,
            hooks_dict=True,
        )

        model_rem_hooks = do_circuit_extraction(
            model=model,
            heads_to_remove=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=mean_dataset,
            return_hooks=True,
            hooks_dict=True,
        )

        circuit_hooks_keys = list(circuit_hooks.keys())

        for layer, head_idx in circuit_hooks_keys:
            if (layer, head_idx) not in heads_to_keep.keys():
                circuit_hooks.pop((layer, head_idx))
        assert len(circuit_hooks) == circuit_size, (len(circuit_hooks), circuit_size)

        # [markdown] needed functions ...

        def cobble_eval(model, nodes):
            """Eval M\nodes"""
            model.reset_hooks()
            for head in nodes:
                model.add_hook(*model_rem_hooks[head])
            cur_logit_diff = logit_diff(model, ioi_dataset)
            model.reset_hooks()
            return cur_logit_diff

        def circuit_eval(model, nodes):
            """Eval C\nodes"""
            model.reset_hooks()
            for head in all_circuit_nodes:
                if head not in nodes:
                    model.add_hook(*circuit_hooks[head])
            for head in complement_hooks:
                if head not in all_circuit_nodes or head in nodes:
                    model.add_hook(*complement_hooks[head])
            cur_logit_diff = logit_diff(model, ioi_dataset)
            model.reset_hooks()
            return cur_logit_diff

        def difference_eval(model, nodes):
            """Eval completeness metric | F(C\nodes) - F(M\nodes) |"""
            c = circuit_eval(model, nodes)
            m = cobble_eval(model, nodes)
            return torch.abs(c - m)

        # actual experiments

        if do_asserts:
            c = circuit_eval(model, [])
            m = cobble_eval(model, [])
            print(f"{c}, {m} {torch.abs(c-m)}")

            for entry in perf_by_sets:  # check backwards compatibility
                circuit_class = entry["removed_group"]  # includes "none"
                assert circuit_class in list(circuit.keys()) + ["none"], circuit_class
                nodes = (
                    circuit[circuit_class] if circuit_class in circuit.keys() else []
                )

                c = circuit_eval(model, nodes)
                m = cobble_eval(model, nodes)

                assert torch.allclose(entry["mean_cur_metric_cobble"], m), (
                    entry["mean_cur_metric_cobble"],
                    m,
                    circuit_class,
                )
                assert torch.allclose(entry["mean_cur_metric_broken"], c), (
                    entry["mean_cur_metric_broken"],
                    c,
                    circuit_class,
                )

                print(f"{circuit_class} {c}, {m} {torch.abs(c-m)}")

        # [markdown] now do the greedy set experiments

        def add_key_to_json_dict(fname, key, value):
            """Thanks copilot"""
            with open(fname, "r") as f:
                d = json.load(f)
            d[key] = value
            with open(fname, "w") as f:
                json.dump(d, f)

        def new_greedy_search(
            no_runs,
            no_iters,
            no_samples,
            save_to_file=True,
            verbose=True,
        ):
            """
            Greedy search to find G that maximizes the difference between broken and cobbled circuit: |metric(C\G) - metric(M\G)|
            """
            all_sets = [{"circuit_nodes": [], "removed_nodes": []}]  # not mantained
            C_minus_G_init = deepcopy(all_nodes)
            C_minus_G_init = [head[0] for head in C_minus_G_init]

            c = circuit_eval(model, [])
            m = cobble_eval(model, [])
            baseline = torch.abs(c - m)

            metadata = {
                "no_runs": no_runs,
                "no_iters": no_iters,
                "no_samples": no_samples,
            }
            fname = (
                f"jsons/greedy_search_results_{raw_circuit_idx}_{doover}_{ctime()}.json"
            )
            print(fname)

            # write to JSON file
            if save_to_file:
                with open(
                    fname,
                    "w",
                ) as outfile:
                    json.dump(metadata, outfile)

            for run in tqdm(range(no_runs)):
                C_minus_G = deepcopy(C_minus_G_init)
                G = []
                old_diff = baseline.clone()

                for iter in range(no_iters):
                    print("iter", iter)
                    to_test = random.sample(C_minus_G, min(no_samples, len(C_minus_G)))
                    # sample without replacement

                    cevals = []
                    mevals = []

                    results = []
                    for (
                        node
                    ) in (
                        to_test
                    ):  # check which heads in to_test causes the biggest drop
                        G_plus_node = deepcopy(G) + [node]

                        cevals.append(circuit_eval(model, G_plus_node).item())
                        mevals.append(cobble_eval(model, G_plus_node).item())
                        results.append(abs(cevals[-1] - mevals[-1]))

                    best_node_idx = np.argmax(results)
                    max_diff = results[best_node_idx]
                    if max_diff > old_diff:
                        best_node = to_test[best_node_idx]
                        C_minus_G.remove(
                            best_node
                        )  # we remove the best node from the circuit
                        G.append(best_node)
                        old_diff = max_diff

                        all_sets.append(
                            {
                                "circuit_nodes": deepcopy(C_minus_G),
                                "removed_nodes": deepcopy(G),
                                "ceval": cevals[best_node_idx],
                                "meval": mevals[best_node_idx],
                            }
                        )
                        if verbose:
                            print(
                                f"iter: {iter} - best node:{best_node} - max brok cob diff:{max(results)} - baseline:{baseline}"
                            )
                            print_gpu_mem(f"iter {iter}")

                run_results = {
                    "result": old_diff,
                    "best set": all_sets[-1]["removed_nodes"],
                    "ceval": all_sets[-1]["ceval"],
                    "meval": all_sets[-1]["meval"],
                }

                if save_to_file:
                    add_key_to_json_dict(fname, f"run {run}", run_results)

        new_greedy_search(
            no_runs=10,
            no_iters=10,
            no_samples=10 if circuit_size == 26 else 5,
            save_to_file=True,
            verbose=True,
        )
#%% [markdown]
# Do random search too. Set `skip_random` to `True` to skip this part.

skip_random = True
mode = "naive"
if mode == "naive":
    circuit = deepcopy(NAIVE)
else:
    circuit = deepcopy(CIRCUIT)
all_nodes = get_all_nodes(circuit)

xs = []
ys = []

for _ in range(100):
    if skip_random:
        break
    indicator = torch.randint(0, 2, (len(all_nodes),))
    nodes = [node[0] for node, ind in zip(all_nodes, indicator) if ind == 1]
    c = circuit_eval(model, nodes)
    m = cobble_eval(model, nodes)
    print(f"{c}, {m} {torch.abs(c-m)}")

    xs.append(c)
    ys.append(m)

if not skip_random:
    torch.save(xs, f"pts/{mode}_random_xs.pt")
    torch.save(ys, f"pts/{mode}_random_ys.pt")

#%% [markdown] hopefully ignoarable plottig proceessin

assert os.getcwd().endswith("Easy-Transformer"), os.getcwd
fnames = os.listdir("jsons")
fnames = [fname for fname in fnames if "greedy_search_results" in fname]

xs = [[], []]
ys = [[], []]
names = []

for circuit_idx in range(0, 2):  # 0 is our circuit, 1 is naive
    for fname in fnames:
        with open(f"jsons/{fname}", "r") as f:
            data = json.load(f)
        for idx in range(100):
            key = f"run {idx}"
            if key in data:
                if (
                    f"results_{circuit_idx}" in fname and "ceval" in data[key]
                ):  # our circuit, not naive
                    xs[circuit_idx].append(data[key]["ceval"])
                    ys[circuit_idx].append(data[key]["meval"])
                    names.append(
                        str(data[key]["best set"]) + " " + str(data[key]["result"])
                    )

                else:
                    pass
#%% [markdown]
# Plot the plot for greedy or naive. Change `mode` to switch between the two.

mode = "complete"
# mode = "naive"

fig = go.Figure()
## add the grey region
# make the dotted line
minx = -2
maxx = 6
eps = 1.0
xs = np.linspace(minx - 1, maxx + 1, 100)
ys = xs

fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=f"x=y",
        line=dict(color="grey", width=2, dash="dash"),
    )
)

perf_by_sets = torch.load(
    f"pts/{mode}_perf_by_sets.pt"
)  # see the format of this file to overwrite plots

rd_set_added = False
for i, perf in enumerate(perf_by_sets):
    fig.add_trace(
        go.Scatter(
            x=[perf["mean_cur_metric_broken"]],
            y=[perf["mean_cur_metric_cobble"]],
            mode="markers",
            name=perf["removed_group"],
            marker=dict(symbol="circle", size=10, color=perf["color"]),
            showlegend=(
                (" 1" in perf["removed_group"][-2:])
                or ("Set" not in perf["removed_group"])
            ),
        )
    )
    continue

# add the greedy
greedy_xs = torch.load(f"pts/{mode}_xs.pt")
greedy_ys = torch.load(f"pts/{mode}_ys.pt")

fig.add_trace(
    go.Scatter(
        x=greedy_xs,
        y=greedy_ys,
        mode="markers",
        name="Greedy",
        marker=dict(symbol="square", size=6, color="blue"),
    )
)

# add the random
random_xs = torch.load(f"pts/{mode}_random_xs.pt")
random_ys = torch.load(f"pts/{mode}_random_ys.pt")

fig.add_trace(
    go.Scatter(
        x=random_xs,
        y=random_ys,
        mode="markers",
        name="Random",
        marker=dict(symbol="triangle-left", size=10, color="green"),
    )
)

# fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="F(C \ K)")
fig.update_yaxes(title_text="F(M \ K)")
fig.update_xaxes(showgrid=True, gridcolor="black", gridwidth=1)
fig.update_yaxes(showgrid=True, gridcolor="black", gridwidth=1)
fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

# USE THESE LINES TO SCALE SVGS PROPERLY
fig.update_xaxes(range=[-1, 6])
fig.update_yaxes(range=[-1, 6])
fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)

circuit_to_export = "natural"
fpath = f"circuit_completeness_{circuit_to_export}_CIRCUIT_at_{ctime()}.svg"
if os.path.exists(
    "/home/ubuntu/my_env/lib/python3.9/site-packages/easy_transformer/svgs"
):
    fpath = "svgs/" + fpath

fig.write_image(fpath)
fig.show()
