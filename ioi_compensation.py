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
from ioi_utils import clear_gpu_mem, show_tokens, show_pp, show_attention_patterns, safe_del, compute_next_tok_dot_prod

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

from ioi_utils import circuit_from_nodes_logit_diff, get_heads_from_nodes

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")


def logit_diff(model, ioi_dataset, logits=None, all=False, std=False):
    """
    Difference between the IO and the S logits at the "to" token
    """
    if logits is None:
        text_prompts = ioi_dataset.text_prompts
        logits = model(text_prompts).detach()
    L = ioi_dataset.N
    IO_logits = logits[
        torch.arange(L),
        ioi_dataset.word_idx["end"][:L],
        ioi_dataset.io_tokenIDs[:L],
    ]
    S_logits = logits[
        torch.arange(L),
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
mean_dataset = ioi_dataset.gen_flipped_prompts("S2")
# %%

CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
        (10, 10),
        (10, 2),
        (11, 2),
        (10, 6),
        (10, 1),
        (11, 6),
        (11, 9),
        (11, 1),
        (9, 7),
        (11, 3),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (2, 9), (4, 11)],
}


RELEVANT_TOKENS = {}
for head in CIRCUIT["name mover"] + CIRCUIT["negative"] + CIRCUIT["s2 inhibition"]:
    RELEVANT_TOKENS[head] = ["end"]

for head in CIRCUIT["induction"]:
    RELEVANT_TOKENS[head] = ["S2"]

for head in CIRCUIT["duplicate token"]:
    RELEVANT_TOKENS[head] = ["S2"]

for head in CIRCUIT["previous token"]:
    RELEVANT_TOKENS[head] = ["S+1"]


ALL_NODES = []  # a node is a tuple (head, token)
for h in RELEVANT_TOKENS:
    for tok in RELEVANT_TOKENS[h]:
        ALL_NODES.append((h, tok))


def update_nm(new_nms=[], reset=False):
    global CIRCUIT, ALL_NODES, RELEVANT_TOKENS
    if reset:
        new_nms = [
            (9, 0),  ###
            (9, 6),  # ori ###
            (9, 7),  ###
            (9, 9),  # ori  ###
            (10, 0),  # ori
            (10, 1),  ###
            (10, 2),  # ~ ###
            (10, 6),  ###
            (10, 10),  ###
            (11, 1),  # ~ ###
            (11, 6),  # ~ negative h
            (11, 9),  # ~ ###
            (11, 2),  ###10
        ]
    CIRCUIT = {
        "name mover": new_nms.copy(),  # , (10, 10), (10, 6)],  # 10, 10 and 10.6 weak nm
        "negative": [
            (10, 7),
            (11, 10),
        ],
        "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
        "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
        "duplicate token": [(0, 1), (0, 10), (3, 0)],
        "previous token": [(2, 2), (2, 9), (4, 11)],
    }

    RELEVANT_TOKENS = {}
    for head in CIRCUIT["name mover"] + CIRCUIT["negative"] + CIRCUIT["s2 inhibition"]:
        RELEVANT_TOKENS[head] = ["end"]

    for head in CIRCUIT["induction"]:
        RELEVANT_TOKENS[head] = ["S2"]

    for head in CIRCUIT["duplicate token"]:
        RELEVANT_TOKENS[head] = ["S2"]

    for head in CIRCUIT["previous token"]:
        RELEVANT_TOKENS[head] = ["S+1", "and"]

    ALL_NODES = []  # a node is a tuple (head, token)
    for h in RELEVANT_TOKENS:
        for tok in RELEVANT_TOKENS[h]:
            ALL_NODES.append((h, tok))


def writing_direction_heatmap(
    model,
    ioi_dataset,
    mode="attn_out",
    return_vals=False,
    dir_mode="IO - S",
    unembed_mode="normal",  # or "Neel"
    title="",
    highlight_heads=None,
    highlight_name="",
    return_ld=False,
    return_figs=False,
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
    ld, std = logit_diff(model, ioi_dataset, logits=logits, std=True, all=False)

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
                cur = cache[f"blocks.{lay}.attn.hook_result"][i, ioi_dataset.word_idx["end"][i], :, :]
            elif mode == "mlp":
                cur = cache[f"blocks.{lay}.hook_mlp_out"][:, -2, :]
            vals[:, lay] += torch.einsum("ha,a->h", cur.cpu(), dire.cpu())

    vals /= N
    vals /= cache["ln_final.hook_scale"][range(ioi_dataset.N), ioi_dataset.word_idx["end"][i]].mean().cpu()
    all_figs = []
    all_figs.append(
        show_pp(
            vals,
            xlabel="head no",
            ylabel="layer no",
            title=title + f" Logit diff: {ld:.2f} +/- {std:.2f}",
            highlight_points=highlight_heads,
            highlight_name=highlight_name,
            return_fig=True,
        )
    )
    if return_figs and return_vals:
        return all_figs, vals
    if return_vals:
        return vals
    if return_figs:
        return all_figs


# %% check if we see distributed name movers

model.reset_hooks()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_from_nodes([((9, 6), "end"), ((9, 9), "end"), ((10, 0), "end")], ioi_dataset),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=mean_dataset,
)

mtx = writing_direction_heatmap(
    model,
    ioi_dataset,
    mode="attn_out",
    dir_mode="IO - S",
    return_vals=True,
    title="Attention head output into IO - S token unembedding (GPT2). Neg Heads and NM KO",
)
mtx_flat = mtx.flatten()
all_sorted_idx = np.abs(mtx_flat).argsort()
for i in range(20):
    x, y = np.unravel_index(all_sorted_idx[-i - 1], mtx.shape)
    print(mtx_flat[all_sorted_idx[-i - 1]], (y, x))


# %%

model.reset_hooks()
writing_direction_heatmap(
    model,
    ioi_dataset,
    mode="attn_out",
    dir_mode="IO - S",
    title="Attention head output into IO - S token unembedding (GPT2) WT",
)


J = [
    ((10, 7), "end"),
    ((11, 10), "end"),
]

J_heads = [j[0] for j in J]

to_highlight = [[j[0] for j in J_heads], [j[1] for j in J_heads]]

update_nm(reset=True)

C_minus_J = list(set(ALL_NODES.copy()) - set(J.copy()))

model.reset_hooks()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=get_heads_from_nodes(C_minus_J, ioi_dataset),  # C\J
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=mean_dataset,
)
ld_cmj, std_cmj = logit_diff(model, ioi_dataset, std=True, all=False)

# model.reset_hooks()
dir_val_C_m_J = writing_direction_heatmap(
    model,
    ioi_dataset,
    return_vals=True,
    mode="attn_out",
    dir_mode="IO - S",
    title=f"Writting dir C\J",
)

model.reset_hooks()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_from_nodes(J, ioi_dataset),  # M\J
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=mean_dataset,
)

ld_mmj, std_mmj = logit_diff(model, ioi_dataset, std=True, all=False)

dir_val_M_m_J = writing_direction_heatmap(
    model,
    ioi_dataset,
    return_vals=True,
    mode="attn_out",
    dir_mode="IO - S",
    title=f"Writting dir M\J",
)

show_pp(
    dir_val_M_m_J - dir_val_C_m_J,
    xlabel="head no",
    ylabel="layer no",
    title=f"Difference IO-S writting matrices between M and C LD: {ld_mmj - ld_cmj:.2f} +/- {std_mmj + std_cmj:.2f}",
    highlight_points=to_highlight,
    highlight_name="Name movers",
)


# %% Find a faithfull naive circuit


def get_relevant_node(circuit):
    RELEVANT_TOKENS = {}
    for head in circuit["name mover"] + circuit["negative"] + circuit["s2 inhibition"]:
        RELEVANT_TOKENS[head] = ["end"]

    for head in circuit["induction"]:
        RELEVANT_TOKENS[head] = ["S2"]

    for head in circuit["duplicate token"]:
        RELEVANT_TOKENS[head] = ["S2"]

    for head in circuit["previous token"]:
        RELEVANT_TOKENS[head] = ["S+1", "and"]

    ALL_NODES = []  # a node is a tuple (head, token)
    for h in RELEVANT_TOKENS:
        for tok in RELEVANT_TOKENS[h]:
            ALL_NODES.append((h, tok))
    return ALL_NODES


NAIVE_CIRCUIT = {
    "name mover": [
        (9, 6),
        (9, 9),
        (10, 0),
    ],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "negative": [],
    "duplicate token": [],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "previous token": [(2, 2), (2, 9), (4, 11)],
}
naive_nodes = get_relevant_node(NAIVE_CIRCUIT)
model.reset_hooks()
CIRCUIT = NAIVE_CIRCUIT.copy()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=get_heads_from_nodes(naive_nodes, ioi_dataset),  # C\J
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=mean_dataset,
)


dir_val_C_m_J = writing_direction_heatmap(
    model,
    ioi_dataset,
    return_vals=True,
    mode="attn_out",
    dir_mode="IO - S",
    title="Attention head output into IO - S token unembedding (GPT2) C",
)

# %% compensation mecanism exploration plot h(R + k*(IO-S)) vs R + k*(IO-S)

J = []
all_diff = []
for IT in range(20):

    J_heads = [j[0] for j in J]

    to_highlight = [[j[0] for j in J_heads], [j[1] for j in J_heads]]

    update_nm(J_heads)

    C_minus_J = list(set(ALL_NODES.copy()) - set(J.copy()))

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=get_heads_from_nodes(C_minus_J, ioi_dataset),  # C\J
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )

    # model.reset_hooks()
    dir_val_C_m_J, ld_C_m_J = writing_direction_heatmap(
        model,
        ioi_dataset,
        return_vals=True,
        mode="attn_out",
        dir_mode="IO - S",
        title="Attention head output into IO - S token unembedding (GPT2) C",
        return_ld=True,
    )

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_remove=get_heads_from_nodes(J, ioi_dataset),  # M\J
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
        mean_dataset=mean_dataset,
    )

    dir_val_M_m_J, ld_M_m_J = writing_direction_heatmap(
        model,
        ioi_dataset,
        return_vals=True,
        mode="attn_out",
        dir_mode="IO - S",
        return_ld=True,
    )

    diff = (dir_val_M_m_J - dir_val_C_m_J).numpy()
    show_pp(
        diff,
        xlabel="head no",
        ylabel="layer no",
        title=f"Difference IO-S writting matrices between M and C {ld_M_m_J - ld_C_m_J:.2f}",
        highlight_points=to_highlight,
        highlight_name="selected so far",
    )

    head, layer = np.where(diff == diff.max())
    head = int(head[0])
    layer = int(layer[0])

    J.append(((layer, head), "end"))
    all_diff.append(ld_M_m_J - ld_C_m_J)

# %%
fig = px.line(y=all_diff, x=range(len(all_diff)), title="LD(M\J)- LD(C\J)) after adding NM head by head")

fig.update_layout(
    xaxis=dict(tickmode="array", tickvals=list(range(len(all_diff))), ticktext=[str(h) for h, t in J]),
    xaxis_title="head added",
    yaxis_title="LD(M\J)- LD(C\J))",
)


fig.show()


fig = px.line(y=all_diff, x=range(len(all_diff)), title="LD(M\J)- LD(C\J)) after adding NM head by head")

fig.update_layout(
    xaxis=dict(tickmode="array", tickvals=list(range(len(all_diff))), ticktext=[str(h) for h, t in J]),
    xaxis_title="head added",
    yaxis_title="LD(M\J)- LD(C\J))",
)


fig.show()


fig.show()

fig = px.line(
    y=[(all_diff[i] - all_diff[i + 1]) for i in range(len(all_diff) - 1)],
    x=range(len(all_diff) - 1),
    title="diff of diff after adding NM head by head",
)

fig.update_layout(
    xaxis=dict(tickmode="array", tickvals=list(range(len(all_diff))), ticktext=[str(h) for h, t in J]),
    xaxis_title="head added",
    yaxis_title="LD(M\J)- LD(C\J))",
)

fig.add_shape(
    # Line Vertical
    dict(type="line", x0=0, y0=0.1, x1=20, y1=0.1, line=dict(color="Red", width=1))
)

fig.show()


# %%


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
    assert layer_to_get is None or type(layer_to_get) == int or layer_to_get == "final_logit_diff"

    if layer_to_get == "final_logit_diff":
        n_heads = 1
    else:
        n_heads = model.cfg["n_heads"]

    model_unembed = (
        model.unembed.W_U.detach()
    )  # note that for GPT2 embeddings and unembeddings are tides such that W_E = Transpose(W_U)

    io_dir = model_unembed[ioi_dataset.io_tokenIDs]
    s_dir = model_unembed[ioi_dataset.s_tokenIDs]
    random_dir1 = model_unembed[np.random.randint(0, model_unembed.shape[0], size=ioi_dataset.N)]
    random_dir2 = model_unembed[np.random.randint(0, model_unembed.shape[0], size=ioi_dataset.N)]

    IO_m_S_dirs = io_dir - s_dir  # random_dir2 - random_dir1

    vals_k = []
    K_values = np.linspace(-50, 50, 100)
    if layer_to_get is None:
        layer_to_get = layer
    else:
        layer_to_get = layer_to_get
    cache = {}
    model.cache_some(cache, lambda x: x in [f"blocks.{layer_to_get}.attn.hook_result", "ln_final.hook_scale"])
    for K in K_values:

        for hp in model.hook_points():
            if hp.name == f"blocks.{layer}.hook_resid_pre":
                hp.remove_hooks("both")

        def write_IO_m_S_in_resid(z, hook):
            """Add the IO - S direction to the residual. Shape of z is (batch, seq_len, embed_dim)"""
            z[:, ioi_dataset.word_idx["end"], :] = z[:, ioi_dataset.word_idx["end"], :] + K * IO_m_S_dirs
            return z

        # model.cache_all(cache)

        logits = model.run_with_hooks(
            ioi_dataset.text_prompts,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", write_IO_m_S_in_resid)],
            reset_hooks_start=False,
            reset_hooks_end=False,
        )

        if layer_to_get == "final_logit_diff":
            ld = logit_diff(model, ioi_dataset, logits)
            final_layer_norm_scaling = cache["ln_final.hook_scale"][0, -1].item()
            vals_k.append(ld * final_layer_norm_scaling)
        else:
            head_out = cache[f"blocks.{layer_to_get}.attn.hook_result"][
                range(ioi_dataset.N), ioi_dataset.word_idx["end"], :, :
            ]  # keep only the end token
            vals = torch.einsum("bhd,bd->bh", head_out, IO_m_S_dirs).mean(dim=0).detach().cpu().numpy()
            vals_k.append(vals)

    vals_k = np.array(vals_k)
    df = pd.DataFrame(vals_k, index=K_values, columns=[f"Head {layer_to_get}.{h}" for h in range(n_heads)])
    df.index.name = "K"

    fig = px.line(df)
    fig.update_layout(
        title=f"Heads from Layer {layer_to_get} writting in the (IO-S) direction vs k*(IO-S) in resid {layer}" + title,
        xaxis_title=f"k",
        yaxis_title="H(R + k*(IO-S)).IO-S" if layer_to_get != "final_logit_diff" else "logit diff (pre final LN)",
    )
    fig.show()


model.reset_hooks()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_from_nodes(
        [
            ((9, 0), "end"),
            ((9, 6), "end"),
            ((9, 9), "end"),
            ((10, 0), "end"),
            ((10, 10), "end"),
            ((10, 6), "end"),
            ((10, 2), "end"),
            ((11, 3), "end"),
            ((11, 2), "end"),
            ((10, 1), "end"),
            ((10, 7), "end"),
            ((10, 7), "end"),
            ((11, 10), "end"),
        ],
        ioi_dataset,
    ),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)

compensation_plot(model, ioi_dataset, layer=9, layer_to_get="final_logit_diff")

# %% Open web text example finding
webtext = load_dataset("stas/openwebtext-10k")

max_nb_tok = 100
nb_seq = 1000
owt_seqs = [webtext["train"]["text"][i][:2000] for i in range(nb_seq)]

# %%

from ioi_utils import get_time, find_owt_stimulus, print_toks_with_color
import os


# return max_seq, min_seq


# %%
model.reset_hooks()
for l in range(4, 12):
    for h in range(12):
        clear_gpu_mem()
        print(f"Layer {l} Head {h}")
        find_owt_stimulus(model, owt_seqs, l, h, export_to_html=True, batch_size=10, k=30)

# %%
import torch.nn.functional as F


def get_per_token_loss(model, seqs, batch_seq=False):
    toks = model.tokenizer(seqs, padding=False).input_ids
    all_losses = []
    if batch_seq:
        all_logits = model(seqs)
    for i, sentence in tqdm(enumerate(seqs)):
        if not batch_seq:
            logits = model(sentence)
        else:
            logits = all_logits[i]
        next_tok = toks[i][1:]  # nb_seq, seq_len-1
        log_probs = F.log_softmax(logits, dim=-1).cpu()
        pred_log_probs = log_probs[0, range(len(next_tok)), next_tok]
        all_losses.append(np.concatenate([-pred_log_probs.detach().cpu().numpy(), np.array([0])]))
    return all_losses


# update_nm(None, reset=True)
model.reset_hooks()
all_losses_M = get_per_token_loss(model, owt_seqs[:100])
print_toks_with_color(show_tokens(owt_seqs[0], model, return_list=True), all_losses_M[0], show_high=True, show_low=True)

model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=get_heads_from_nodes(
        ALL_NODES,
        ioi_dataset,
    ),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)
all_losses_C = get_per_token_loss(model, owt_seqs[:100], batch_seq=True)


# %%


def get_head_param(model, module, layer, head):
    if module == "OV":
        W_v = model.blocks[layer].attn.W_V[head]
        W_o = model.blocks[layer].attn.W_O[head]
        W_ov = torch.einsum("hd,bh->db", W_v, W_o)
        return W_ov
    if module == "QK":
        W_k = model.blocks[layer].attn.W_K[head]
        W_q = model.blocks[layer].attn.W_Q[head]
        W_qk = torch.einsum("hd,hb->db", W_q, W_k)
        return W_qk
    if module == "Q":
        W_q = model.blocks[layer].attn.W_Q[head]
        return W_q
    if module == "K":
        W_k = model.blocks[layer].attn.W_K[head]
        return W_k
    if module == "V":
        W_v = model.blocks[layer].attn.W_V[head]
        return W_v
    if module == "O":
        W_o = model.blocks[layer].attn.W_O[head]
        return W_o
    raise ValueError(f"module {module} not supported")


def compute_composition(model, l1, h1, l2, h2, module_1, module_2):
    W_1 = get_head_param(model, module_1, l1, h1).detach()
    W_2 = get_head_param(model, module_2, l2, h2).detach()
    W_12 = torch.einsum("db,bc->dc", W_2, W_1)
    comp_score = torch.norm(W_12) / (torch.norm(W_1) * torch.norm(W_2))

    return comp_score.cpu().numpy()


def test_composition_layers(model, l1, h1, l2, h2, module_1, module_2):

    n_heads = model.cfg["n_heads"]
    scores = []
    for h_a in range(n_heads):
        for h_b in range(n_heads):
            print(f"head {h_a} vs {h_b}")
            if h_a == h1 and h_b == h2:
                interaction_idx = idx
            scores.append(compute_composition(model, l1, h_a, l2, h_b, module_1, module_2))
            idx += 1
    print(
        f"Interaction is the {np.count_nonzero(np.array(scores) > scores[interaction_idx])}th most important interaction"
    )


def plot_composition(model, targ_l, targ_h, targ_module, test_module):
    n_heads = model.cfg["n_heads"]
    n_layers = model.cfg["n_layers"]

    shape1 = get_head_param(model, targ_module, targ_l, targ_h).detach().cpu().numpy().shape
    shape2 = get_head_param(model, test_module, 0, 0).detach().cpu().numpy().shape
    # sample 10 random matrices and compute the baseline composition score
    baseline_scores = []
    for _ in range(10):
        W_1 = np.random.randn(*shape1)
        W_2 = np.random.randn(*shape2)
        W_12 = np.einsum("db,bc->dc", W_2, W_1)
        comp_score = np.linalg.norm(W_12) / (np.linalg.norm(W_1) * np.linalg.norm(W_2))
        baseline_scores.append(comp_score)
    baseline_score = np.mean(baseline_scores)

    scores = np.zeros((targ_l, n_heads))
    for l in range(targ_l):
        for h in range(n_heads):
            scores[l, h] = compute_composition(model, l, h, targ_l, targ_h, test_module, targ_module) - baseline_score
    fig = px.imshow(
        scores,
        title=f"Composition from {test_module} to {targ_module} of {targ_l}.{targ_h}",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )
    fig.update_layout(
        xaxis_title="Head",
        yaxis_title="Layer",
    )
    fig.show()
    return scores


# %%
