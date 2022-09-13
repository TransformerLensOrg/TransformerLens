#%%
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

# %% [markdown]
# ioi_dataset`ioi_dataset.word_idx` contains the indices of certains special words in each prompt. Example on the prompt 0

# %%
[(k, int(ioi_dataset.word_idx[k][0])) for k in ioi_dataset.word_idx.keys()]

# %%
[
    (i, t)
    for (i, t) in enumerate(
        show_tokens(ioi_dataset.ioi_prompts[0]["text"], model, return_list=True)
    )
]

# %% [markdown]
# The `ioi_dataset` ca also generate a copy of itself where some names have been flipped by a random name that is unrelated to the context with `gen_flipped_prompts`. This will be useful for patching experiments.

# %%
flipped = ioi_dataset.gen_flipped_prompts("S2")
pprint(flipped.ioi_prompts[:5])

# %% [markdown]
# IOIDataset contains many other useful features, see the definition of the class in the cell `Dataset class` for more info!

# %% [markdown]
# We also import open web text sentences to compute means that are not correlated with our IOI distribution.

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
)


def logit_diff(model, text_prompts):
    """Difference between the IO and the S logits (at the "to" token)"""
    logits = model(text_prompts).detach()
    IO_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.io_tokenIDs,
    ]
    S_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.s_tokenIDs,
    ]
    return (IO_logits - S_logits).mean().detach().cpu()


#%% [markdown]
# TODO Explain the way we're doing Jacob's circuit extraction experiment here
#%%

for G in CIRCUIT.keys():
    if G == "ablation":
        continue

    # compute METRIC( C \ G )
    excluded_classes = ["calibration"]
    if G is not None:
        excluded_classes.append(G)
    heads_to_keep, mlps_to_keep = get_heads_circuit(
        ioi_dataset, excluded_classes=excluded_classes, mlp0=True
    )  # TODO check the MLP stuff
    model = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_keep=mlps_to_keep,
        ioi_dataset=ioi_dataset,
    )

    # ld = logit_diff(

# %%
