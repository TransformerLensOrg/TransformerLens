# %% Random experiment : writing direction after removing name movers
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

from ioi_circuit_extraction import (
    join_lists,
    CIRCUIT,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
)

from functools import partial

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")


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


# %%
# gpt
print_gpu_mem("About to load model")
model = EasyTransformer(
    r"gpt2", use_attn_result=True
)  # use_attn_result adds a hook blocks.{lay}.attn.hook_result that is before adding the biais of the attention layer
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
# %%
# IOI Dataset initialisation
N = 200
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)

# %%


def writing_direction_heatmap(
    model,
    ioi_dataset,
    mode="attn_out",
    return_vals=False,
    dir_mode="IO - S",
    unembed_mode="normal",  # or "Neel"
    title="",
):
    """
    Plot the dot product between how much each attention head
    output with `IO-S`, the difference between the unembeds between
    the (correct) IO token and the incorrect S token
    """

    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers

    model_unembed = (
        model.unembed.W_U.detach().cpu()
    )  # note that for GPT2 embeddings and unembeddings are tides such that W_E = Transpose(W_U)

    if mode == "attn_out":  # heads, layers
        vals = torch.zeros(size=(n_heads, n_layers))
    elif mode == "mlp":
        vals = torch.zeros(size=(1, n_layers))
    else:
        raise NotImplementedError()

    N = ioi_dataset.N
    cache = {}
    model.cache_all(cache)  # TODO maybe speed up by only caching relevant things

    logits = model(ioi_dataset.text_prompts)

    for i in range(ioi_dataset.N):
        io_tok = ioi_dataset.toks[i][ioi_dataset.word_idx["IO"][i].item()]
        s_tok = ioi_dataset.toks[i][ioi_dataset.word_idx["S"][i].item()]
        io_dir = model_unembed[io_tok]
        s_dir = model_unembed[s_tok]
        if dir_mode == "IO - S":
            dire = io_dir - s_dir
        elif dir_mode == "IO":
            dire = io_dir
        elif dir_mode == "S":
            dire = s_dir
        else:
            raise NotImplementedError()

        for lay in range(n_layers):
            if mode == "attn_out":
                cur = cache[f"blocks.{lay}.attn.hook_result"][
                    i, ioi_dataset.word_idx["end"][i], :, :
                ]
            elif mode == "mlp":
                cur = cache[f"blocks.{lay}.hook_mlp_out"][:, -2, :]
            vals[:, lay] += torch.einsum("ha,a->h", cur.cpu(), dire.cpu())

    vals /= N
    show_pp(vals, xlabel="head no", ylabel="layer no", title=title)
    if return_vals:
        return vals


model.reset_hooks()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_from_nodes(
        [
            ((9, 6), "end"),
            ((9, 9), "end"),
            ((10, 0), "end"),
            ((10, 10), "end"),
            ((10, 6), "end"),
            ((10, 2), "end"),
            ((11, 3), "end"),
            ((10, 8), "end"),
        ],
        ioi_dataset,
    ),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)

# model.reset_hooks()
dir_val = writing_direction_heatmap(
    model,
    ioi_dataset,
    return_vals=True,
    mode="attn_out",
    dir_mode="IO - S",
    title="Attention head output into IO - S token unembedding (GPT2)",
)

# %% compensation mecanism exploration plot h(R + k*(IO-S)) vs R + k*(IO-S)


def compensation_plot(
    model,
    ioi_dataset,
    title="",
    layer=0,
    layer_to_get=None,
):
    """
    Plot the dot product between how much each attention head
    output with `IO-S`, the difference between the unembeds between
    the (correct) IO token and the incorrect S token
    """

    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers

    model_unembed = (
        model.unembed.W_U.detach()
    )  # note that for GPT2 embeddings and unembeddings are tides such that W_E = Transpose(W_U)

    io_dir = model_unembed[ioi_dataset.io_tokenIDs]
    s_dir = model_unembed[ioi_dataset.s_tokenIDs]
    random_dir1 = model_unembed[
        np.random.randint(0, model_unembed.shape[0], size=ioi_dataset.N)
    ]
    random_dir2 = model_unembed[
        np.random.randint(0, model_unembed.shape[0], size=ioi_dataset.N)
    ]

    IO_m_S_dirs = io_dir - s_dir  # random_dir2 - random_dir1

    vals_k = []
    K_values = np.linspace(-50, 50, 100)
    if layer_to_get is None:
        layer_to_get = layer
    else:
        layer_to_get = layer_to_get
    cache = {}
    model.cache_some(cache, lambda x: x == f"blocks.{layer_to_get}.attn.hook_result")
    for K in K_values:

        for hp in model.hook_points():
            if hp.name == f"blocks.{layer}.hook_resid_pre":
                hp.remove_hooks("both")

        def write_IO_m_S_in_resid(z, hook):
            """Add the IO - S direction to the residual. Shape of z is (batch, seq_len, embed_dim)"""
            z[:, ioi_dataset.word_idx["end"], :] = (
                z[:, ioi_dataset.word_idx["end"], :] + K * IO_m_S_dirs
            )
            return z

        # model.cache_all(cache)

        logits = model.run_with_hooks(
            ioi_dataset.text_prompts,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", write_IO_m_S_in_resid)],
            reset_hooks_start=False,
            reset_hooks_end=False,
        )
        head_out = cache[f"blocks.{layer_to_get}.attn.hook_result"][
            range(ioi_dataset.N), ioi_dataset.word_idx["end"], :, :
        ]  # keep only the end token
        vals = (
            torch.einsum("bhd,bd->bh", head_out, IO_m_S_dirs)
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        vals_k.append(vals)

    vals_k = np.array(vals_k)
    df = pd.DataFrame(
        vals_k,
        index=K_values,
        columns=[f"Head {layer_to_get}.{h}" for h in range(n_heads)],
    )
    df.index.name = "K"

    fig = px.line(df)
    fig.update_layout(
        title=f"Heads from Layer {layer_to_get} writting in the (IO-S) direction vs k*(IO-S) in resid {layer}"
        + title,
        xaxis_title=f"k",
        yaxis_title="H(R + k*(IO-S)).IO-S",
    )
    fig.show()


model.reset_hooks()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_from_nodes(
        [
            ((9, 6), "end"),
            ((9, 9), "end"),
            ((10, 0), "end"),
            ((10, 10), "end"),
            ((10, 6), "end"),
            ((10, 2), "end"),
            ((11, 3), "end"),
            ((11, 2), "end"),
            ((10, 1), "end"),
        ],
        ioi_dataset,
    ),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)
for l in [10, 11]:
    compensation_plot(model, ioi_dataset, layer=l, layer_to_get=l)

# %%
