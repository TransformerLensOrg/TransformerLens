# %% [markdown]
# ## Imports
import os
import torch

if os.environ["USER"] in ["exx", "arthur"]:  # so Arthur can safely use octobox
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
assert torch.cuda.device_count() == 1
from easy_transformer.EasyTransformer import LayerNormPre
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
from time import ctime
from functools import partial
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
    all_subsets,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
)
from ioi_circuit_extraction import (
    do_circuit_extraction,
    gen_prompt_uniform,
    get_act_hook,
    get_circuit_replacement_hook,
    get_extracted_idx,
    get_heads_circuit,
    join_lists,
    list_diff,
    process_heads_and_mlps,
    turn_keep_into_rmv,
    CIRCUIT,
)

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")


import sys

sys.path.append("../PySvelte")
import pysvelte

#%% [markdown]
# # <h1><b>Setup</b></h1>
# Import model and dataset
#%% # plot writing in the IO - S direction
model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")

print_gpu_mem("About to load model")
model = EasyTransformer.from_pretrained(
    model_name,
)  # use_attn_result adds a hook blocks.{lay}.attn.hook_result that is before adding the biais of the attention layer
model.set_use_attn_result(True)
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
# %%
# IOI Dataset initialisation
N = 150
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)

# %%

webtext = load_dataset("stas/openwebtext-10k")
MAX_CONTEXT = 1024
owb_seqs = [
    "".join(
        show_tokens(webtext["train"]["text"][i], model, return_list=True)[:MAX_CONTEXT]
    )
    for i in range(20)
]


# %%
show_attention_patterns(
    model, [(5, 5), (6, 9), (5, 8), (5, 9)], ioi_dataset[:5], mode="val"
)
# %%


def show_attention(model, seq, mode="val", heads=[], return_val_only=False):
    assert mode in [
        "attn",
        "val",
    ]  # value weighted attention or attn for attention probas
    all_attn = []
    for (layer, head) in heads:
        cache = {}
        good_names = [f"blocks.{layer}.attn.hook_attn"]
        if mode == "val":
            good_names.append(f"blocks.{layer}.attn.hook_v")
        model.cache_some(
            cache=cache, names=lambda x: x in good_names
        )  # shape: batch head_no seq_len seq_len

        logits = model([seq])

        toks = model.tokenizer(seq)["input_ids"]
        words = [model.tokenizer.decode([tok]) for tok in toks]
        attn = cache[good_names[0]].detach().cpu()[0, head, :, :]
        if mode == "val":
            vals = cache[good_names[1]].detach().cpu()[0, :, head, :].norm(dim=-1)
            attn = torch.einsum("ab,b->ab", attn, vals)
        all_attn.append(attn.unsqueeze(0))
    all_attn = torch.concat(all_attn)
    all_attn = einops.rearrange(
        all_attn, "num_heads dest_pos src_pos -> dest_pos src_pos num_heads"
    )

    if return_val_only:
        return all_attn

    toks = show_tokens(seq, model, return_list=True)

    html_object = pysvelte.AttentionMulti(
        tokens=toks, attention=all_attn, head_labels=[str(h) for h in heads]
    )
    html_object.show()


# %%

heads = [(5, 5), (5, 8)]

heads = [(7, 3), (7, 9), (8, 6), (8, 10)]


l = 3
head_for_layer = [(l, x) for x in range(12)]

show_attention(model, owb_seqs[6][:2000], mode="val", heads=heads)  # bugged here : (
# %%
prep_seq = []
for i in range(20):
    if owb_seqs[i][:2000].count("by") > 30 or owb_seqs[i][:2000].count("to") > 30:
        prep_seq.append(owb_seqs[i][:2000])


# %%
heads = [(l, h) for l in range(12) for h in range(12)]
NB_SEQ = 10
attn_score = [
    show_attention(
        model, owb_seqs[i][:2000], mode="attn", heads=heads, return_val_only=True
    )
    for i in range(NB_SEQ)
]
# %%
all_prev = []
all_dup = []
all_ind = []
for seq in range(NB_SEQ):
    toks = show_tokens(owb_seqs[seq][:2000], model, return_list=True)
    for i, t in enumerate(toks):
        if t in toks[:i]:
            prev_idx = [j for j, x in enumerate(toks) if x == t and j < i]
            prev_p_one = [j + 1 for j, x in enumerate(toks) if x == t and j < i]
            dup_attn = attn_score[seq][i, prev_idx, :]
            induct_attn = attn_score[seq][i, prev_p_one, :]
            all_dup.append(dup_attn)
            all_ind.append(induct_attn)
        all_prev.append(attn_score[seq][i, [i - 1], :])


all_prev = torch.concat(all_prev)
all_ind = torch.concat(all_ind)
all_dup = torch.concat(all_dup)


# %%
