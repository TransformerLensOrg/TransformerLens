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
    "".join(show_tokens(webtext["train"]["text"][i][:2000], model, return_list=True)[: ioi_dataset.max_len])
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
            first_bit = IO_logits - S_logits
        else:
            first_bit = (IO_logits - S_logits).mean().detach().cpu()
        return first_bit, torch.std(IO_logits - S_logits).detach().cpu()
    return (IO_logits - S_logits).mean().detach().cpu()


#%% [markdown]
# TODO Explain the way we're doing Jacob's circuit extraction experiment here
#%%
circuit_perf = pd.DataFrame(
    columns=["ldiff_broken", "ldiff_cobble", "std_ldiff_broken", "std_ldiff_cobble", "removed_group"]
)

ldiff_broken_circuit = []
ldiff_cobble_circuit = []
std_ldiff_broken_circuit = []
std_ldiff_cobble_circuit = []

for G in CIRCUIT.keys():
    if G == "ablation":
        continue

    # compute METRIC( C \ G )
    # excluded_classes = ["calibration"]
    excluded_classes = []
    excluded_classes.append(G)

    heads_to_keep = get_heads_circuit(ioi_dataset, excluded_classes=excluded_classes)  # TODO check the MLP stuff
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    ldiff_broken_circuit, std_broken_circuit = logit_diff(model, ioi_dataset, std=True, all=False)
    # metric(C\G)

    # adding back the whole model
    excl_class = list(CIRCUIT.keys())
    excl_class.remove(G)
    G_heads_to_remove = get_heads_circuit(ioi_dataset, excluded_classes=excl_class)  # TODO check the MLP stuff

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_remove=G_heads_to_remove,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    ldiff_cobble, std_cobble_circuit = logit_diff(model, ioi_dataset, std=True, all=False)
    # metric(M\G)

    circuit_perf = circuit_perf.append(
        {
            "removed_group": G,
            "ldiff_broken": ldiff_broken_circuit,
            "ldiff_cobble": ldiff_cobble,
            "std_ldiff_broken": std_broken_circuit,
            "std_ldiff_cobble": std_cobble_circuit,
        },
        ignore_index=True,
    )

    # ld = logit_diff(

# %%

px.scatter(circuit_perf, x="ldiff_broken", y="ldiff_cobble", text="removed_group")

# %%
