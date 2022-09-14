#%%
from dataclasses import dataclass
from email.mime import base
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
model = EasyTransformer(model_name, use_attn_result=True)
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
# %% [markdown]
# Each prompts is a dictionnary containing 'IO', 'S' and the "text", the sentence that will be given to the model.
# The prompt type can be "ABBA", "BABA" or "mixed" (half of the previous two) depending on the pattern you want to study
# %%
# IOI Dataset initialisation
N = 200
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
# %%
webtext = load_dataset("stas/openwebtext-10k")
owb_seqs = [
    "".join(
        show_tokens(webtext["train"]["text"][i][:2000], model, return_list=True)[
            : ioi_dataset.max_len
        ]
    )
    for i in range(ioi_dataset.N)
]
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
        return IO_logits - S_logits
    if std:
        if all:
            first_bit = (IO_logits - S_logits).detach().cpu()
        else:
            first_bit = (IO_logits - S_logits).mean().detach().cpu()
        return first_bit, torch.std(IO_logits - S_logits).detach().cpu()
    return (IO_logits - S_logits).mean().detach().cpu()


#%% [markdown]
# TODO Explain the way we're doing the minimal circuit experiment here
#%%

results = {}

for G in tqdm(list(CIRCUIT.keys())):
    print_gpu_mem(G)

    # compute METRIC(C \ W)
    heads_to_keep = get_heads_circuit(ioi_dataset, excluded_classes=[G])
    torch.cuda.empty_cache()

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    ldiff_broken_circuit, std_broken_circuit = logit_diff(model, ioi_dataset, std=True)
    torch.cuda.empty_cache()

    results[G] = {
        "ldiff_broken_circuit": ldiff_broken_circuit,
        "std_broken_circuit": std_broken_circuit,
    }
    results[G]["vs"] = {}

    # METRIC((C \ W) \cup \{ v \})

    for v in tqdm(list(CIRCUIT[G])):
        new_heads_to_keep = heads_to_keep.copy()
        v_indices = get_extracted_idx(RELEVANT_TOKENS[v], ioi_dataset)
        assert v not in new_heads_to_keep.keys()
        new_heads_to_keep[v] = v_indices

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_keep=new_heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
        )
        torch.cuda.empty_cache()
        ldiff_with_v = logit_diff(model, ioi_dataset, std=True)
        results[G]["vs"][v] = ldiff_with_v
        torch.cuda.empty_cache()
#%%
#%%
def compute_baseline(model, ioi_dataset):
    heads_to_keep = get_heads_circuit(ioi_dataset, excluded_classes=[])
    torch.cuda.empty_cache()

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    ldiff_broken_circuit, std_broken_circuit = logit_diff(model, ioi_dataset, std=True)
    torch.cuda.empty_cache()
    return ldiff_broken_circuit, std_broken_circuit


recalculate_baseline = False

if recalculate_baseline:
    circuit_baseline = compute_baseline(model, ioi_dataset)
    model.reset_hooks()
    baseline = logit_diff(model, ioi_dataset, std=True)

else:
    circuit_baseline = (3.5512, 1.4690)
    baseline = (3.5467, 1.6115)
baseline_ldiff = baseline[0]
circuit_baseline_diff = circuit_baseline[0]
#%%
fig = go.Figure()

for G in list(CIRCUIT.keys()):
    for i, v in enumerate(list(CIRCUIT[G])):
        fig.add_trace(
            go.Bar(
                x=[G],
                y=[results[G]["vs"][v][0] - results[G]["ldiff_broken_circuit"]],
                base=results[G]["ldiff_broken_circuit"],
                width=1 / (len(CIRCUIT[G]) + 1),
                offset=i / (len(CIRCUIT[G]) + 1),
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
    y0=baseline_ldiff,
    y1=baseline_ldiff,
    line_width=1,
    line=dict(
        color="blue",
        width=4,
        dash="dashdot",
    ),
)

fig.add_trace(
    go.Scatter(
        x=["name mover"],
        y=[3.3],
        text=["Logit difference of M"],
        mode="text",
        textfont=dict(
            color="blue",
            size=10,
            # family="Arail",
        ),
    )
)

fig.add_shape(
    type="line",
    xref="x",
    x0=-2,
    x1=8,
    yref="y",
    y0=circuit_baseline_diff,
    y1=circuit_baseline_diff,
    line_width=1,
    line=dict(
        color="black",
        width=4,
        dash="dashdot",
    ),
)

fig.add_trace(
    go.Scatter(
        x=["name mover"],
        y=[3.8],
        text=["Logit difference of C"],
        mode="text",
        textfont=dict(
            color="black",
            size=10,
            # family="Arail",
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
model.reset_hooks()
ldiff_baseline = logit_diff(model, ioi_dataset, std=True)
