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

    for v in tqdm(list(CIRCUIT[G].keys())):
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
            x1=12,
            yref="y",
            y0=0,
            y1=0,
        )
    ]
)

fig.show()
