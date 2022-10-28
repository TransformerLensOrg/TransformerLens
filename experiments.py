#%% [markdown]
## Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small
# <h1><b>Intro</b></h1>

# This notebook implements all experiments in our paper (which is available on arXiv).

# For background on the task, see the paper.

# Refer to the demo of the <a href="https://github.com/neelnanda-io/Easy-Transformer">Easy-Transformer</a> library here: <a href="https://github.com/neelnanda-io/Easy-Transformer/blob/main/EasyTransformer_Demo.ipynb">demo with ablation and patching</a>.
#
# Reminder of the circuit:
# <img src="https://i.imgur.com/arokEMj.png">
#%% [markdown]
# Setup (TODO cut extras)
from copy import deepcopy
import os
import torch

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
import warnings
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
from copy import deepcopy
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
    path_patching,
    max_2d,
    CLASS_COLORS,
    all_subsets,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
    scatter_attention_and_contribution,
)
from random import randint as ri
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
    ARTHUR_CIRCUIT,
)
from ioi_utils import logit_diff, probs
from ioi_utils import get_top_tokens_and_probs as g

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%% [markdown]
# Initialise model (use larger N or fewer templates for no warnings about in-template ablation)
model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)
#%% [markdown]
# Initialise dataset
N = 100
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
)

print(f"Here are two of the prompts from the dataset: {ioi_dataset.sentences[:2]}")
#%% [markdown]
# See logit difference
model_logit_diff = logit_diff(model, ioi_dataset)
model_io_probs = probs(model, ioi_dataset)
print(
    f"The model gets average logit difference {model_logit_diff.item()} over {N} examples"
)
print(f"The model gets average IO probs {model_io_probs.item()} over {N} examples")
#%% [markdown]
# The circuit
circuit = deepcopy(CIRCUIT)

# we make the ABC dataset in order to knockout other model components
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
# we then add hooks to the model to knockout all the heads except the circuit
model.reset_hooks()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=get_heads_circuit(ioi_dataset=ioi_dataset, circuit=circuit),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=abc_dataset,
)

circuit_logit_diff = logit_diff(model, ioi_dataset)
print(
    f"The circuit gets average logit difference {circuit_logit_diff.item()} over {N} examples"
)
#%% [markdown]
# Edge patching
def plot_edge_patching(
    model,
    ioi_dataset,
    receiver_hooks,  # list of tuples (hook_name, idx). If idx is not None, then at dim 2 index in with idx (used for doing things for specific attention heads)
    position,
):
    model.reset_hooks()
    default_logit_diff = logit_diff(model, ioi_dataset)
    results = torch.zeros(size=(12, 12))
    mlp_results = torch.zeros(size=(12, 1))
    for source_layer in tqdm(range(12)):
        for source_head_idx in [None] + list(range(12)):
            model.reset_hooks()

            model = path_patching(
                model=model,
                source_dataset=abc_dataset,
                target_dataset=ioi_dataset,
                ioi_dataset=ioi_dataset,
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=receiver_hooks,
                max_layer=12,
                positions=[position],
                verbose=False,
                return_hooks=False,
                freeze_mlps=False,
                have_internal_interactions=False,
            )
            cur_logit_diff = logit_diff(model, ioi_dataset)

            if source_head_idx is None:
                mlp_results[source_layer] = cur_logit_diff - default_logit_diff
            else:
                results[source_layer][source_head_idx] = (
                    cur_logit_diff - default_logit_diff
                )

            if source_layer == 1:
                assert not torch.allclose(results, 0.0 * results), results

            if source_layer == 11 and source_head_idx == 11:
                results /= default_logit_diff
                mlp_results /= default_logit_diff

                results *= 100
                mlp_results *= 100

                # show attention head results
                fig = show_pp(
                    results.T,
                    title=f"Effect of patching (Heads->Final Residual Stream State) path",
                    return_fig=True,
                    show_fig=False,
                    bartitle="% change in logit difference",
                )
                fig.show()


plot_edge_patching(
    model,
    ioi_dataset,
    receiver_hooks=[(f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None)],
    position="end",
)
#%% [markdown]
# Reproduce writing results (change the layer_no and head_no)

scatter_attention_and_contribution(
    model=model, layer_no=9, head_no=9, ioi_dataset=ioi_dataset
)
#%% [markdown]
# Look at the copy score for the Name Mover and Negative heads


def check_copy_circuit(model, layer, head, ioi_dataset, verbose=False, neg=False):
    cache = {}
    model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
    model(ioi_dataset.toks.long())
    if neg:
        sign = -1
    else:
        sign = 1
    z_0 = model.blocks[1].ln1(cache["blocks.0.hook_resid_post"])

    v = torch.einsum("eab,bc->eac", z_0, model.blocks[layer].attn.W_V[head])
    v += model.blocks[layer].attn.b_V[head].unsqueeze(0).unsqueeze(0)

    o = sign * torch.einsum("sph,hd->spd", v, model.blocks[layer].attn.W_O[head])
    logits = model.unembed(model.ln_final(o))

    k = 5
    n_right = 0

    for seq_idx, prompt in enumerate(ioi_dataset.ioi_prompts):
        for word in ["IO", "S", "S2"]:
            pred_tokens = [
                model.tokenizer.decode(token)
                for token in torch.topk(
                    logits[seq_idx, ioi_dataset.word_idx[word][seq_idx]], k
                ).indices
            ]
            if "S" in word:
                name = "S"
            else:
                name = word
            if " " + prompt[name] in pred_tokens:
                n_right += 1
            else:
                if verbose:
                    print("-------")
                    print("Seq: " + ioi_dataset.sentences[seq_idx])
                    print("Target: " + ioi_dataset.ioi_prompts[seq_idx][name])
                    print(
                        " ".join(
                            [
                                f"({i+1}):{model.tokenizer.decode(token)}"
                                for i, token in enumerate(
                                    torch.topk(
                                        logits[
                                            seq_idx, ioi_dataset.word_idx[word][seq_idx]
                                        ],
                                        k,
                                    ).indices
                                )
                            ]
                        )
                    )
    percent_right = (n_right / (ioi_dataset.N * 3)) * 100
    print(
        f"Copy circuit for head {layer}.{head} (sign={sign}) : Top {k} accuracy: {percent_right}%"
    )
    return percent_right


neg_sign = False
print(" --- Name Mover heads --- ")
check_copy_circuit(model, 9, 9, ioi_dataset, neg=neg_sign)
check_copy_circuit(model, 10, 0, ioi_dataset, neg=neg_sign)
check_copy_circuit(model, 9, 6, ioi_dataset, neg=neg_sign)

neg_sign = True
print(" --- Calibration heads --- ")
check_copy_circuit(model, 10, 7, ioi_dataset, neg=neg_sign)
check_copy_circuit(model, 11, 10, ioi_dataset, neg=neg_sign)

neg_sign = False
print(" ---  Random heads for control ---  ")
check_copy_circuit(
    model, random.randint(0, 11), random.randint(0, 11), ioi_dataset, neg=neg_sign
)
check_copy_circuit(
    model, random.randint(0, 11), random.randint(0, 11), ioi_dataset, neg=neg_sign
)
check_copy_circuit(
    model, random.randint(0, 11), random.randint(0, 11), ioi_dataset, neg=neg_sign
)
#%% [markdown]
# S-Inhibition patching

plot_edge_patching(
    model,
    ioi_dataset,
    receiver_hooks=[
        (f"blocks.{layer_idx}.attn.hook_v", head_idx)
        for layer_idx, head_idx in circuit["s2 inhibition"]
    ],
    position="S2",
)

#%% [markdown]
# Attention probs of NMs

ys = []
average_attention = {}

for idx, dataset in enumerate([ioi_dataset, abc_dataset]):
    fig = go.Figure()
    for heads_raw in circuit["name mover"][
        :3
    ]: 
        heads = [heads_raw]
        average_attention[heads_raw] = {}
        cur_ys = []
        cur_stds = []
        att = torch.zeros(size=(dataset.N, dataset.max_len, dataset.max_len))
        for head in tqdm(heads):
            att += show_attention_patterns(
                model, [head], dataset, return_mtx=True, mode="attn"
            )
        att /= len(heads)

        vals = att[torch.arange(dataset.N), ioi_dataset.word_idx["end"][: dataset.N], :]
        evals = torch.exp(vals)
        val_sum = torch.sum(evals, dim=1)
        assert val_sum.shape == (dataset.N,), val_sum.shape

        for key in ioi_dataset.word_idx.keys():
            end_to_s2 = att[
                torch.arange(dataset.N),
                ioi_dataset.word_idx["end"][: dataset.N],
                ioi_dataset.word_idx[key][: dataset.N],
            ]
            cur_ys.append(end_to_s2.mean().item())
            cur_stds.append(end_to_s2.std().item())
            average_attention[heads_raw][key] = end_to_s2.mean().item()
        fig.add_trace(
            go.Bar(
                x=list(ioi_dataset.word_idx.keys()),
                y=cur_ys,
                error_y=dict(type="data", array=cur_stds),
                name=str(heads_raw),
            )
        )
        fig.update_layout(title_text=f"Attention of NMs from END to various positions on {["ioi_dataset", "abc_dataset"][idx]}")
    fig.show()
#%% [markdown] And attention on a single sentence

model.reset_hooks()
show_attention_patterns(model, [(9, 9), (9, 6), (10, 0)], ioi_dataset[:1])

#%% [markdown] Position and token signals???



#%% [markdown] Backup NMs

#%% [markdown] Random sequence stuff

#%% [markdown] Direct and indirect effects of MLPs
