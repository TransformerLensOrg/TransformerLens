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
# TODO Explain the way we're doing Jacob's circuit extraction experiment here
#%%
circuit_perf_scatter = []
circuit_perf = []


def basis_change(x, y):
    """
    Change the basis (1, 0) and (0, 1) to the basis
    1/sqrt(2) (1, 1) and 1/sqrt(2) (-1, 1)
    """

    return (x + y) / np.sqrt(2), (y - x) / np.sqrt(2)


for G in tqdm(list(CIRCUIT.keys()) + ["none"]):
    if G == "ablation":
        continue
    print_gpu_mem(G)
    # compute METRIC( C \ G )
    # excluded_classes = ["calibration"]
    excluded_classes = []
    if G != "none":
        excluded_classes.append(G)
    heads_to_keep = get_heads_circuit(
        ioi_dataset, excluded_classes=excluded_classes
    )  # TODO check the MLP stuff
    torch.cuda.empty_cache()

    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    ldiff_broken_circuit, std_broken_circuit = logit_diff(
        model, ioi_dataset, std=True, all=True
    )
    torch.cuda.empty_cache()
    # metric(C\G)
    # adding back the whole model

    excl_class = list(CIRCUIT.keys())
    if G != "none":
        excl_class.remove(G)
    G_heads_to_remove = get_heads_circuit(
        ioi_dataset, excluded_classes=excl_class
    )  # TODO check the MLP stuff
    torch.cuda.empty_cache()

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_remove=G_heads_to_remove,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    ldiff_cobble, std_cobble_circuit = logit_diff(
        model, ioi_dataset, std=True, all=True
    )
    torch.cuda.empty_cache()

    # metric(M\G)

    on_diagonals = []
    off_diagonals = []

    for i in range(len(ldiff_cobble)):
        circuit_perf_scatter.append(
            {
                "removed_group": G,
                "ldiff_broken": ldiff_broken_circuit[i],
                "ldiff_cobble": ldiff_cobble[i],
                "sentence": ioi_dataset.text_prompts[i],
                "template": ioi_dataset.templates_by_prompt[i],
            }
        )

        x, y = basis_change(
            circuit_perf_scatter[-1]["ldiff_broken"],
            circuit_perf_scatter[-1]["ldiff_cobble"],
        )
        circuit_perf_scatter[-1]["on_diagonal"] = x
        circuit_perf_scatter[-1]["off_diagonal"] = y
        on_diagonals.append(x)
        off_diagonals.append(y)

    circuit_perf.append(
        {
            "removed_group": G,
            "ldiff_broken": ldiff_broken_circuit.detach().mean(),
            "ldiff_cobble": ldiff_cobble.detach().mean(),
            "std_ldiff_broken": std_broken_circuit,
            "std_ldiff_cobble": std_cobble_circuit,
            "on_diagonal": np.mean(on_diagonals),
            "off_diagonal": np.mean(off_diagonals),
            "std_on_diagonal": np.std(on_diagonals),
            "std_off_diagonal": np.std(off_diagonals),
        }
    )
#%%
fig = px.scatter(
    circuit_perf,
    x="on_diagonal",
    y="off_diagonal",
    color="removed_group",
    text="removed_group",
    # hover_data=["sentence", "template"],
    opacity=0.7,
    error_x="std_on_diagonal",
    error_y="std_off_diagonal",
)

fig.update_layout(
    shapes=[
        # adds line at y=5
        dict(
            type="line",
            xref="x",
            x0=0,
            x1=7,
            yref="y",
            y0=0,
            y1=0,
        )
    ]
)

fig.show()
# %%
# TODO do the experiment with orthogonally turning things

fig = px.scatter(
    circuit_perf,
    x="on_diagonal",
    y="off_diagonal",
    color="removed_group",
    hover_data=["sentence", "template"],
    opacity=0.7,
    error_x="std_broken_circuit",
    error_y="std_cobble_circuit",
)

fig.show()
