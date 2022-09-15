#%%
from easy_transformer import EasyTransformer
import logging
import sys
from ioi_circuit_extraction import *
import optuna
from ioi_dataset import *
import IPython
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
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
from time import ctime

study = optuna.create_study(
    study_name=f"Check by heads and index @ {ctime()}", storage=storage_name
)  # ADD!

# %%
model = EasyTransformer("gpt2", use_attn_result=True).cuda()
N = 200
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)


def logit_diff(model, text_prompts, std=False):
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
    if std:
        return (IO_logits - S_logits).mean().detach().cpu(), (
            IO_logits - S_logits
        ).std().detach().cpu()
    else:
        return (IO_logits - S_logits).mean().detach().cpu()


def baseline(remove_neg=True):
    cur_stuff = []
    for circuit_class in CIRCUIT.keys():
        if circuit_class == "negative" and remove_neg:
            continue
        for head in CIRCUIT[circuit_class]:
            for relevant_token in RELEVANT_TOKENS[head]:
                cur_stuff.append((head, relevant_token))
    heads = {head: [] for head, _ in cur_stuff}
    for head, val in cur_stuff:
        heads[head].append(val)
    heads_to_keep = {}
    for head in heads.keys():
        heads_to_keep[head] = get_extracted_idx(heads[head], ioi_dataset)
    model.reset_hooks()
    new_model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    ldiff, std = logit_diff(new_model, ioi_dataset.text_prompts, std=True)
    torch.cuda.empty_cache()
    del new_model
    torch.cuda.empty_cache()
    return -ldiff, std


bl, std = baseline()
print("BASELINE:", bl, std)

#%%
relevant_stuff = []  # pairs (layer, head), TOKEN

for layer in range(12):
    for head in range(12):
        for relevant_token in ["IO", "S", "S+1", "S2", "and", "end"]:  # TODO more?
            if relevant_token != "end" and layer == 11:
                continue
            relevant_stuff.append(((layer, head), relevant_token))


def objective(trial):
    cur_stuff = []
    for i in range(20):
        cur_stuff.append(trial.suggest_categorical("idx_{}".format(i), relevant_stuff))
        # relevant_stuff.remove(cur_stuff[-1])
    print(cur_stuff[-1])
    heads = {head: [] for head, _ in cur_stuff}
    for head, val in cur_stuff:
        heads[head].append(val)
    heads_to_keep = {}
    for head in heads.keys():
        heads_to_keep[head] = get_extracted_idx(heads[head], ioi_dataset)
    model.reset_hooks()
    new_model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    ldiff = logit_diff(new_model, ioi_dataset.text_prompts)
    torch.cuda.empty_cache()
    del new_model
    torch.cuda.empty_cache()
    return -ldiff


study.optimize(objective, n_trials=1e8)
#%% # new answer extraction

from ioi_utils import *

ids = get_indices_from_sql_file("example-study.db", 1494)
print(ids)

for idx in ids:
    print(relevant_stuff[idx])

#%%
# old answer extraction
eyes = [8, 23, 66, 95, 5, 21, 49, 11, 3, 41, 26, 29, 35, 77, 99, 91, 41, 17, 47, 47]
for eye in eyes:
    print(relevant_stuff[eye])

#%% # found from searching over all our heads, at all index positions
NEW_CIRCUIT = {
    # old name mover
    (9, 6): ["S2", "end"],
    (9, 9): ["S+1", "end"],
    (10, 0): ["end"],
    # old s2 inhibition
    (7, 3): ["S2", "end"],
    (7, 9): ["S+1", "end"],
    (10, 7): [],
    (11, 10): [],
    # old induction
    (5, 5): ["end"],
    (5, 8): ["S"],
    (5, 9): [],
    (6, 9): [],
    # old duplicate
    (0, 1): ["IO"],
    (0, 10): ["end"],
    (3, 0): [],
    # old previous token
    (2, 2): [],
    (2, 9): ["S", "end"],
    (4, 11): ["S2"],
}

#%% # found from searching over all heads, at all index positions

NEWER_CIRCUIT = {
    # old name mover
    (9, 6): [],
    (9, 9): ["end"],
    (10, 0): ["S+1"],  # weird
    # old s2 inhibition
    (7, 3): [],
    (7, 9): ["IO"],
    (10, 7): [],
    (11, 10): [],
    # old induction
    (5, 5): [],
    (5, 8): [],
    (5, 9): [],
    (6, 9): [],
    # old duplicate
    (0, 1): [],
    (0, 10): [],
    (3, 0): [],
    # old previous token
    (2, 2): [],
    (2, 9): [],
    (4, 11): [],
    # new things!
    # ((4, 0), 'IO')
    # ((1, 5), 'S+1')
    # ((6, 8), 'S')
    # ((10, 6), 'IO')
    # ((10, 10), 'end')
    # ((8, 10), 'end')
    # ((9, 2), 'S+1')
    # ((5, 3), 'and')
    # ((2, 10), 'S2')
    # ((10, 4), 'S2')
    # ((0, 9), 'S')
    # ((7, 8), 'S')
    # ((1, 8), 'and')
    # ((2, 7), 'S2')
    # ((1, 5), 'end')
    # ((8, 7), 'end')
    # ((7, 0), 'S+1')
}
