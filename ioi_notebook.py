#%% [markdown]
# # GPT-2 small Indirect Object Identification
# <h1><b>Intro</b></h1>
# This notebook is an implementation of the IOI experiments (some with adjustments from the <a href="https://docs.google.com/presentation/d/19H__CYCBL5F3M-UaBB-685J-AuJZNsXqIXZR-O4j9J8/edit#slide=id.g14659e4d87a_0_290">presentation</a>.
# It should be able to be run as by just git cloning this repo (+ some easy installs).
#
# ### Task
# We're interested in GPT-2 ability to complete sentences like "Alice and Bob went to the store, Alice gave a bottle of milk to"...
# GPT-2 knows that it have to output a name that is not the subject (Alice) and that was present in the context: Bob.
# The first apparition of Alice is called "S" (or sometimes "S1") for "Subject", and Bob is the indirect object ("IO"). Even if the sentences we generate contains the last word of the sentence "Bob", we'll never look at the transformer output here. What's matter is the next-token prediction on the token "to", sometime called the "end" token.
#
# ### Tools
# In this notebook, we define a class `IOIDataset` to handle the generation and utils for this particular dataset.
#
# Refer to the demo of the [`easy_transformer` library](https://github.com/neelnanda-io/Easy-Transformer) here: <a href="https://colab.research.google.com/drive/1MLwJ7P94cizVs2LD8Qwi-vLGSoH-cHxq?usp=sharing">demo with ablation & patching</a>.
#
# Reminder of the circuit:
# <img src="https://i.imgur.com/PPtTQRh.png">

# %% [markdown]
# ## Imports
from easy_transformer.EasyTransformer import MODEL_NAMES_DICT, LayerNormPre
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
    ARTHUR_CIRCUIT,
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
N = 100
ioi_dataset_baba = IOIDataset(prompt_type="BABA", N=N, tokenizer=model.tokenizer)
ioi_dataset_abba = IOIDataset(prompt_type="ABBA", N=N, tokenizer=model.tokenizer)
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts("S2")  # we flip the second b for a random c
pprint(abca_dataset.text_prompts[:5])

abca_dataset_abba = ioi_dataset_abba.gen_flipped_prompts("S2")
abca_dataset_baba = ioi_dataset_baba.gen_flipped_prompts("S2")


def logit_diff(model, ioi_dataset, all=False):
    """Difference between the IO and the S logits (at the "to" token)"""
    logits = model(ioi_dataset.text_prompts).detach()
    IO_logits = logits[
        torch.arange(ioi_dataset.N),
        ioi_dataset.word_idx["end"],
        ioi_dataset.io_tokenIDs,
    ]
    S_logits = logits[
        torch.arange(ioi_dataset.N),
        ioi_dataset.word_idx["end"],
        ioi_dataset.s_tokenIDs,
    ]
    if all:
        return IO_logits - S_logits
    return (IO_logits - S_logits).mean().detach().cpu()


# %% [markdown]
# ioi_dataset `ioi_dataset.word_idx` contains the indices of certains special words in each prompt. Example on the prompt 0
# %%
[(k, int(ioi_dataset.word_idx[k][0])) for k in ioi_dataset.word_idx.keys()]
# %%
[(i, t) for (i, t) in enumerate(show_tokens(ioi_dataset.ioi_prompts[0]["text"], model, return_list=True))]
# %% [markdown]
# The `ioi_dataset` can also generate a copy of itself where some names have been flipped by a random name that is unrelated to the context with `gen_flipped_prompts`. This will be useful for patching experiments.
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
    "".join(show_tokens(webtext["train"]["text"][i][:2000], model, return_list=True)[: ioi_dataset.max_len])
    for i in range(ioi_dataset.N)
]
# %% [markdown]
# # <h1><b>Initial evidence</b></h1>
# %% [markdown]
# ### Layer Ablations
# %% [markdown]
# The first series of experiment: we define our metric, here, how much the logit for IO is bigger than S, we ablate part of the network and see what matters. Globally, it shows that the behavior is distributed accross many parts of the network, we cannot draw much conclusion from this alone.
# %%
def mean_at_end(z, mean, hook):  # to ablate at particular indices, we have to define a custom ablation function
    z[torch.arange(len(ioi_dataset.ioi_prompts)), ioi_dataset.word_idx["end"], :] = mean[
        torch.arange(len(ioi_dataset.ioi_prompts)), ioi_dataset.word_idx["end"], :
    ]
    return z


# %%
metric = ExperimentMetric(metric=logit_diff, dataset=ioi_dataset, relative_metric=True)
config_mlp = AblationConfig(
    abl_type="custom",
    abl_fn=mean_at_end,
    mean_dataset=owb_seqs,
    target_module="mlp",
    head_circuit="result",
    cache_means=True,
    verbose=False,
)
abl_mlp = EasyAblation(model, config_mlp, metric)
mlp_result = abl_mlp.run_ablation()

config_attn_layer = AblationConfig(
    abl_type="custom",
    abl_fn=mean_at_end,
    mean_dataset=owb_seqs,
    target_module="attn_layer",
    head_circuit="result",
    cache_means=True,
    verbose=False,
)
abl_attn_layer = EasyAblation(model, config_attn_layer, metric)
attn_result = abl_attn_layer.run_ablation()

layer_ablation = torch.cat([mlp_result, attn_result], dim=0)

fig = px.imshow(
    layer_ablation,
    labels={"x": "Layer"},
    title="Logit Difference Variation after Mean Ablation (on Open Web text) at all tokens",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
)

fig.update_layout(yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["mlp", "attention layer"]))
fig.show()
#%% Mean everywhere

config_mlp = AblationConfig(
    abl_type="mean",
    mean_dataset=owb_seqs,
    target_module="mlp",
    head_circuit="result",
    cache_means=True,
    verbose=False,
)
abl_mlp = EasyAblation(model, config_mlp, metric)
mlp_result = abl_mlp.run_ablation()

config_attn_layer = AblationConfig(
    abl_type="mean",
    mean_dataset=owb_seqs,
    target_module="attn_layer",
    head_circuit="result",
    cache_means=True,
    verbose=False,
)
abl_attn_layer = EasyAblation(model, config_attn_layer, metric)
attn_result = abl_attn_layer.run_ablation()

layer_ablation = torch.cat([mlp_result, attn_result], dim=0)

fig = px.imshow(
    layer_ablation,
    labels={"x": "Layer"},
    title="Logit Difference Variation after Mean Ablation (on Open Web text) at all tokens",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
)

fig.update_layout(yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["mlp", "attention layer"]))
fig.show()


# %% random ablation

metric = ExperimentMetric(metric=logit_diff, dataset=ioi_dataset, relative_metric=True)
config_mlp = AblationConfig(
    abl_type="random",
    mean_dataset=owb_seqs,
    target_module="mlp",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    nb_metric_iteration=10,
    max_seq_len=ioi_dataset.max_len,
)
abl_mlp = EasyAblation(model, config_mlp, metric)
mlp_result = abl_mlp.run_ablation()

config_attn_layer = AblationConfig(
    abl_type="random",
    mean_dataset=owb_seqs,
    target_module="attn_layer",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    nb_metric_iteration=10,
    max_seq_len=ioi_dataset.max_len,
)
abl_attn_layer = EasyAblation(model, config_attn_layer, metric)
attn_result = abl_attn_layer.run_ablation()

layer_ablation = torch.cat([mlp_result, attn_result], dim=0)

fig = px.imshow(
    layer_ablation,
    labels={"x": "Layer"},
    title="Logit Difference Variation after Random Ablation at all tokens",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
)

fig.update_layout(yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["mlp", "attention layer"]))
fig.show()


# %% ABC-mean ablation of mlps
metric = ExperimentMetric(metric=logit_diff, dataset=ioi_dataset, relative_metric=True)
config_mlp = AblationConfig(
    abl_type="mean",
    mean_dataset=abca_dataset.text_prompts,
    target_module="mlp",
    head_circuit="result",
    cache_means=True,
    verbose=False,
)
abl_mlp = EasyAblation(
    model,
    config_mlp,
    metric,
    mean_by_groups=True,
    groups=ioi_dataset.groups,  # abc ablation
)
mlp_result = abl_mlp.run_ablation()
fig = show_pp(
    mlp_result.T,
    title="Logit Difference Variation after ABC-mean KO of MLPs at all tokens",
    xlabel="Layer",
    ylabel="MLP",
    return_fig=True,
)
fig.write_image("svgs/abc_ablate_mlps.svg")
# %%
len(ioi_dataset.text_prompts)
# %% [markdown]
# ### Attention Heads Ablations

# %% [markdown]
# #### Mean ablation

# %%
metric = ExperimentMetric(metric=logit_diff, dataset=ioi_dataset, relative_metric=True)
config = AblationConfig(
    abl_type="mean",
    mean_dataset=owb_seqs,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=False,
)
abl = EasyAblation(model, config, metric)
result = abl.run_ablation()
plotly.offline.init_notebook_mode(connected=True)
px.imshow(
    result,
    labels={"y": "Layer", "x": "Head"},
    title="Logit Difference Variation after Mean Ablation (on Open Web text) at all tokens",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
).show()

# %%

metric = ExperimentMetric(metric=logit_diff, dataset=ioi_dataset, relative_metric=True)
config = AblationConfig(
    abl_type="random",
    mean_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    nb_metric_iteration=1,
    max_seq_len=ioi_dataset.max_len,
)
abl = EasyAblation(model, config, metric)
result = abl.run_ablation()
plotly.offline.init_notebook_mode(connected=True)
px.imshow(
    result,
    labels={"y": "Layer", "x": "Head"},
    title="Logit Difference Variation after Random Ablation (on Open Web text) at all tokens",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
).show()

# %% [markdown]
# Blue squares corresponds to head that when ablated makes the logit diff *bigger*. (the color are
# inverted relative to the picture of the slides because here we plot the relative variation vs the relative *drop* in the presentation plots)

# %% [markdown]
# #### Zero ablations

# %%
def zero_at_end(z, mean, hook):
    z[torch.arange(len(ioi_dataset.ioi_prompts)), ioi_dataset.word_idx["end"], :] = 0.0
    return z


metric = ExperimentMetric(metric=logit_diff, dataset=ioi_dataset, relative_metric=True)

config = AblationConfig(
    abl_type="custom",
    abl_fn=zero_at_end,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    mean_dataset=ioi_dataset.text_prompts,
    verbose=False,  # TODO new version of experiments has a bug with verbose
)
abl = EasyAblation(model, config, metric)
result = abl.run_ablation()

px.imshow(
    result,
    labels={"y": "Layer", "x": "Head"},
    title='Logit Difference Variation after Zero Ablation on "to" token',
    color_continuous_midpoint=0,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu",
).show()

# %% [markdown]
# Zero ablation gives roughtly the same results expect for 1.10 that appears strongly. Suggesting that its mean value is really important for the logit diff, even if its specific value on each prompt does not matter to much.

# %% [markdown]
# #### Symetric ablation
# Here we write in the symetric direction relative to the mean. The same head appears.

# %%
def sym_mean(z, mean, hook):
    return mean - (z - mean)


config = AblationConfig(
    abl_type="custom",
    abl_fn=sym_mean,
    target_module="attn_head",
    head_circuit="z",
    cache_means=True,
    mean_dataset=ioi_dataset.text_prompts,
    verbose=False,
)
abl = EasyAblation(model, config, metric)
result = abl.run_ablation()

px.imshow(
    result,
    labels={"y": "Layer", "x": "Head"},
    title='Logit Difference Variation after Zero Ablation on "to" token',
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
).show()
# %% [markdown]
# ### Which head write in the direction Embed(IO) - Embed(S) ?
#%%

MODEL_CFG = model.cfg
MODEL_EPS = model.cfg.eps


def get_layer_norm_div(x, eps=MODEL_EPS):
    mean = x.mean(dim=-1, keepdim=True)
    new_x = (x - mean).detach().clone()
    return (new_x.var(dim=-1, keepdim=True).mean() + eps).sqrt()


def layer_norm(x, cfg=MODEL_CFG):
    return LayerNormPre(cfg)(x)


def pytorch_layer_norm(x, eps=MODEL_EPS):
    return torch.nn.LayerNorm(normalized_shape=x.shape[-1], eps=eps)(x)


m = torch.randn(2, 3, 4)
assert torch.allclose(layer_norm(m), pytorch_layer_norm(m))


def writing_direction_heatmap(
    model,
    ioi_dataset,
    show=["attn"],  # can add "mlp" to this
    return_vals=False,
    dir_mode="IO - S",
    unembed_mode="normal",
    title="",
    verbose=False,
    return_figs=False,
):
    """
    Plot the dot product between how much each attention head
    output with `IO-S`, the difference between the unembeds between
    the (correct) IO token and the incorrect S token
    """

    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    model_unembed = (
        model.unembed.W_U.detach().cpu()
    )  # note that for GPT2 embeddings and unembeddings are tides such that W_E = Transpose(W_U)

    unembed_bias = model.unembed.b_U.detach().cpu()

    attn_vals = torch.zeros(size=(n_heads, n_layers))
    mlp_vals = torch.zeros(size=(n_layers,))
    logit_diffs = logit_diff(model, ioi_dataset, all=True).cpu()

    for i in tqdm(range(ioi_dataset.N)):
        io_tok = ioi_dataset.toks[i][ioi_dataset.word_idx["IO"][i].item()]
        s_tok = ioi_dataset.toks[i][ioi_dataset.word_idx["S"][i].item()]
        io_dir = model_unembed[io_tok]
        s_dir = model_unembed[s_tok]
        unembed_bias_io = unembed_bias[io_tok]
        unembed_bias_s = unembed_bias[s_tok]
        if dir_mode == "IO - S":
            dire = io_dir - s_dir
        elif dir_mode == "IO":
            dire = io_dir
        elif dir_mode == "S":
            dire = s_dir
        else:
            raise NotImplementedError()
        dire.to("cuda")
        cache = {}
        model.cache_all(cache, device="cuda")  # TODO maybe speed up by only caching relevant things
        logits = model(ioi_dataset.text_prompts[i])

        res_stream_sum = torch.zeros(size=(d_model,), device="cuda") # cuda implem to speed up things
        res_stream_sum += cache["blocks.0.hook_resid_pre"][0, -2, :]  # .detach().cpu()
        # the pos and token embeddings

        layer_norm_div = get_layer_norm_div(cache["blocks.11.hook_resid_post"][0, -2, :])

        for lay in range(n_layers):
            cur_attn = (
                cache[f"blocks.{lay}.attn.hook_result"][0, -2, :, :]
                # + model.blocks[lay].attn.b_O.detach()  # / n_heads
            )
            cur_mlp = cache[f"blocks.{lay}.hook_mlp_out"][:, -2, :][0]

            # check that we're really extracting the right thing
            res_stream_sum += torch.sum(cur_attn, dim=0)
            res_stream_sum += model.blocks[lay].attn.b_O  # .detach().cpu()
            res_stream_sum += cur_mlp
            assert torch.allclose(
                res_stream_sum,
                cache[f"blocks.{lay}.hook_resid_post"][0, -2, :].detach(),
                rtol=1e-3,
                atol=1e-3,
            ), lay

            cur_mlp -= cur_mlp.mean()
            for i in range(n_heads):
                cur_attn[i] -= cur_attn[i].mean()
                # we layer norm the end result of the residual stream,
                # (which firstly centres the residual stream)
                # so to estimate "amount written in the IO-S direction"
                # we centre each head's output
            cur_attn /= layer_norm_div  # ... and then apply the layer norm division
            cur_mlp /= layer_norm_div

            attn_vals[:n_heads, lay] += torch.einsum("ha,a->h", cur_attn.cpu(), dire.cpu())
            mlp_vals[lay] = torch.einsum("a,a->", cur_mlp.cpu(), dire.cpu())

        res_stream_sum -= res_stream_sum.mean()
        res_stream_sum = layer_norm(res_stream_sum.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        cur_writing = torch.einsum("a,a->", res_stream_sum, dire.to("cuda")) + unembed_bias_io - unembed_bias_s

        assert i == 11 or torch.allclose(  # ??? and it's way off, too
            cur_writing,
            logit_diffs[i],
            rtol=1e-2,
            atol=1e-2,
        ), f"{i=} {cur_writing=} {logit_diffs[i]}"

    attn_vals /= ioi_dataset.N
    mlp_vals /= ioi_dataset.N
    all_figs = []
    if "attn" in show:
        all_figs.append(show_pp(attn_vals, xlabel="head no", ylabel="layer no", title=title, return_fig=True))
    if "mlp" in show:
        all_figs.append(show_pp(mlp_vals.unsqueeze(0).T, xlabel="", ylabel="layer no", title=title, return_fig=True))
    if return_figs and return_vals:
        return all_figs, attn_vals, mlp_vals
    if return_vals:
        return attn_vals, mlp_vals
    if return_figs:
        return all_figs


torch.cuda.empty_cache()

all_figs, attn_vals, mlp_vals = writing_direction_heatmap(
    model,
    ioi_dataset,
    return_vals=True,
    show=["attn", "mlp"],
    dir_mode="IO - S",
    title="Output into IO - S token unembedding direction",
    verbose=True,
    return_figs=True,
)
modules = ["attn", "mlp"]

for i, fig in enumerate(all_figs):
    fig.write_image(f"svgs/writing_direction_heatmap_module_{modules[i]}.svg")
# %% [markdown]
# We observe heads that are writting in to push IO more than S (the blue suare), but also other hat writes in the opposite direction (red squares). The brightest blue square (9.9, 9.6, 10.0) are name mover heads. The two red (11.10 and 10.7) are the callibration heads.
# %%
show_attention_patterns(model, [(9, 9), (9, 6), (10, 0)], ioi_dataset[:1])
# %%
show_attention_patterns(model, [(11, 10), (10, 7)], ioi_dataset[:1])
#%%
att = show_attention_patterns(model, [(8, 10)], abca_dataset[:2], return_mtx=True, mode="attn")


# %% [markdown]
# ### Plot attention vs direction
# %% [markdown]
# We want to investigate what are the head we observed doing. By plotting attention patterns we see that they are paying preferential attention to IO.

# %%
from ioi_utils import max_2d

attn_vals = writing_direction_heatmap(
    model,
    ioi_dataset,
    return_vals=True,
    dir_mode="IO - S",
    title="Output into IO - S token unembedding direction",
)
k = 20
print(f"Top {k} heads (by magnitude):")
all_grps = []
top_heads, vals = max_2d(torch.abs(attn_vals.T), k=k)  # remove abs to just get positive contributors
for i in range(len(top_heads)):
    grp = "None"
    for gr in CIRCUIT.keys():
        if top_heads[i] in CIRCUIT[gr]:
            grp = gr
    all_grps.append(grp)

    print(f"{top_heads[i]}: {vals[i]} {grp}")
head_names = [str(top_heads[i]) for i in range(len(top_heads))]


grps_color = {
    "None": "rgb(150, 150, 150)",
    "name mover": "rgb(27, 158, 119)",
    "negative": "rgb(217, 95, 2)",
    "s2 inhibition": "rgb(117, 112, 179)",
    "induction": "rgb(231, 41, 138)",
    "duplicate token": "rgb(102, 166, 30)",
    "previous token": "rgb(230, 171, 2)",
}

all_colors = [
    "rgb(27, 158, 119)",
    "rgb(217, 95, 2)",
    "rgb(117, 112, 179)",
    "rgb(150, 150, 150)",
]  # , "rgb(102, 166, 30)", "rgb(230, 171, 2)"]
all_groups = list(grps_color.keys())

colors = [all_groups.index(all_grps[i]) for i in range(len(top_heads))]

fig = px.bar(vals, color_discrete_sequence=all_colors, color=all_grps)
fig.update_layout(
    title=f"Head projection on IO-S",
    xaxis_title="Head",
    yaxis_title="Projection on IO-S",
    xaxis={
        "ticktext": head_names,
        "tickvals": list(range(len(head_names))),
        "tickfont": dict(size=13),
        "side": "bottom",
    },
    height=400,
)
fig.show()
fig.write_image("svgs/head_projection_on_IO-S_bar_chart.svg")


# %% After KO
from ioi_circuit_extraction import (
    join_lists,
    CIRCUIT,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
    do_circuit_extraction,
    list_diff,
)
from ioi_utils import get_heads_from_nodes

model.reset_hooks()
model, _ = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_from_nodes([((9, 9), "end"), ((10, 0), "end"), ((9, 6), "end")], ioi_dataset),  # C\J
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=abca_dataset,
)

#%% [markdown]
# <h2>Copying</h2>
# CLAIM: heads 9.6, 9.9 and 10.0 copy the IO into the residual stream, <b>by attending to the IO token</b>
#%% # the more attention, the more writing
from ioi_utils import scatter_attention_and_contribution

scatter_attention_and_contribution(model, 11, 2, ioi_dataset)
scatter_attention_and_contribution(model, 9, 9, ioi_dataset)
scatter_attention_and_contribution(model, 9, 6, ioi_dataset)
scatter_attention_and_contribution(model, 10, 0, ioi_dataset)

# %%
df = scatter_attention_and_contribution(model, 10, 7, ioi_dataset, return_vals=True)
df.columns = [c.replace(" ", "_") for c in df.columns]
a1, a2 = df["Attn_Prob_on_Name"].to_numpy(), df["Dot_w_Name_Embed"].to_numpy()
a2 = [float(x) for x in a2]
a1 = [float(x) for x in a1]
print(np.corrcoef(a1, a2))

#%% # for control purposes, check that there is unlikely to be a correlation between attention and writing for unimportant heads
scatter_attention_and_contribution(
    model,
    random.randint(0, 11),
    random.randint(0, 11),
    ioi_dataset.ioi_prompts[:500],
    gpt_model="gpt2",
)
# %% [markdown]
# They all demonstrate a straightforward relationship: the more they pay attention to a token, the more they write in its direction. A.k.a. they are just copying the token embedding of this position.

# %% [markdown]
# ### Check copy circuit of Name Movers Heads

# %% [markdown]
# To ensure that the name movers heads are indeed only copying information, we conduct a "check copying circuit" experiment. This means that we only keep the first layer of the transformer and apply the OV circuit of the head and decode the logits from that. Every other component of the transformer is deleted (i.e. zero ablated).
#
#%%
def check_copy_circuit(model, layer, head, ioi_dataset, verbose=False, neg=False):
    cache = {}
    model.cache_some(cache, lambda x: x == "blocks.0.hook_resid_post")
    model(ioi_dataset.text_prompts)
    if neg:
        sign = -1
    else:
        sign = 1
    z_0 = model.blocks[1].ln1(cache["blocks.0.hook_resid_post"])
    v = z_0 @ model.blocks[layer].attn.W_V[head].T + model.blocks[layer].attn.b_V[head]
    o = sign * torch.einsum("sph,dh->spd", v, model.blocks[layer].attn.W_O[head])
    logits = model.unembed(model.ln_final(o))
    k = 5
    n_right = 0

    for seq_idx, prompt in enumerate(ioi_dataset.ioi_prompts):
        # print(prompt)
        for word in ["IO", "S", "S2"]:
            pred_tokens = [
                model.tokenizer.decode(token)
                for token in torch.topk(logits[seq_idx, ioi_dataset.word_idx[word][seq_idx]], k).indices
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
                    print("Seq: " + ioi_dataset.text_prompts[seq_idx])
                    print("Target: " + ioi_dataset.ioi_prompts[seq_idx][name])
                    print(
                        " ".join(
                            [
                                f"({i+1}):{model.tokenizer.decode(token)}"
                                for i, token in enumerate(
                                    torch.topk(
                                        logits[seq_idx, ioi_dataset.word_idx[word][seq_idx]],
                                        k,
                                    ).indices
                                )
                            ]
                        )
                    )
    percent_right = (n_right / (ioi_dataset.N * 3)) * 100
    print(f"Copy circuit for head {layer}.{head} (sign={sign}) : Top {k} accuracy: {percent_right}%")
    return percent_right


neg_sign = True
print(" --- Name Mover heads --- ")

check_copy_circuit(model, 9, 9, ioi_dataset, neg=neg_sign)
check_copy_circuit(model, 10, 0, ioi_dataset, neg=neg_sign)
check_copy_circuit(model, 9, 6, ioi_dataset, neg=neg_sign)

print(" --- Calibration heads --- ")
check_copy_circuit(model, 10, 7, ioi_dataset, neg=neg_sign)
check_copy_circuit(model, 11, 10, ioi_dataset, neg=neg_sign)

print(" ---  Random heads for control ---  ")
check_copy_circuit(model, random.randint(0, 11), random.randint(0, 11), ioi_dataset, neg=neg_sign)
check_copy_circuit(model, random.randint(0, 11), random.randint(0, 11), ioi_dataset, neg=neg_sign)
check_copy_circuit(model, random.randint(0, 11), random.randint(0, 11), ioi_dataset, neg=neg_sign)
#%% [markdown]
# For calibration heads, we observe a reverse trend to name movers, the more is pays attention to a name, the more it write in its *oposite* direction. Why is that?
# You need to remember the training objective of the transformer: it has to predict accurate probability distribution over all the next tokens.
# If previously it was able to recover the IO, in the final layer it has to callibrate the probability of this particular token, it cannot go all in "THE NEXT TOKEN IS BOB" with 100% proba.
# This is why we observe calibration mechanisms that do back and forth and seems to inhibate information put by earlier modules.

# You can see this similarly as open loop / closed loop optimization. It's easier to make a good guess by making previous rough estimate more precise than making a good guess in one shot.
#%%
scatter_attention_and_contribution(model, 10, 7, ioi_dataset.ioi_prompts[:500], gpt_model="gpt2")
scatter_attention_and_contribution(model, 11, 10, ioi_dataset.ioi_prompts[:500], gpt_model="gpt2")
# %% [markdown]
# ### Patching experiments
# %% [markdown]
# What causes name mover heads to pay attention to IO? To figure this out, we'll patch activation from sentences like "A..B..C..to"  to "A..B..B..to" where A, B and C are differents names.
# %%
print_gpu_mem()
# %%
pprint(ioi_dataset.text_prompts[:5])

# %% # here...

heads_to_measure = [(9, 6), (9, 9), (10, 0)]  # name movers
heads_by_layer = {9: [6, 9], 10: [0]}
layers = [9, 10]
hook_names = [f"blocks.{l}.attn.hook_attn" for l in layers]

text_prompts = [prompt["text"] for prompt in ioi_dataset.ioi_prompts]


def attention_probs(
    model, text_prompts, variation=True
):  # we have to redefine logit differences to use the new abba dataset
    """Difference between the IO and the S logits at the "to" token"""
    cache_patched = {}
    model.cache_some(cache_patched, lambda x: x in hook_names)  # we only cache the activation we're interested
    logits = model(text_prompts).detach()
    # we want to measure Mean(Patched/baseline) and not Mean(Patched)/Mean(baseline)
    model.reset_hooks()
    cache_baseline = {}
    model.cache_some(cache_baseline, lambda x: x in hook_names)  # we only cache the activation we're interested
    logits = model(text_prompts).detach()
    # attn score of head HEAD at token "to" (end) to token IO

    attn_probs_variation_by_keys = []
    for key in ["IO", "S", "S2"]:
        attn_probs_variation = []
        for i, hook_name in enumerate(hook_names):
            layer = layers[i]
            for head in heads_by_layer[layer]:
                attn_probs_patched = cache_patched[hook_name][
                    torch.arange(len(text_prompts)),
                    head,
                    ioi_dataset.word_idx["end"],
                    ioi_dataset.word_idx[key],
                ]
                attn_probs_base = cache_baseline[hook_name][
                    torch.arange(len(text_prompts)),
                    head,
                    ioi_dataset.word_idx["end"],
                    ioi_dataset.word_idx[key],
                ]
                if variation:
                    attn_probs_variation.append(
                        ((attn_probs_patched - attn_probs_base) / attn_probs_base).mean().unsqueeze(dim=0)
                    )
                else:
                    attn_probs_variation.append(attn_probs_patched.mean().unsqueeze(dim=0))
        attn_probs_variation_by_keys.append(torch.cat(attn_probs_variation).mean(dim=0, keepdim=True))

    attn_probs_variation_by_keys = torch.cat(attn_probs_variation_by_keys, dim=0)
    return attn_probs_variation_by_keys.detach().cpu()


# %%
circuit = CIRCUIT.copy()
average_changes = torch.zeros(size=(12, 12, 3))
no_times_used = torch.zeros(size=(12,))

for idx, (layer, head_idx) in enumerate(tqdm(circuit["name mover"])):
    print(idx, "of", len(circuit["name mover"]))
    hook_name = f"blocks.{layer}.attn.hook_attn"
    text_prompts = [prompt["text"] for prompt in ioi_dataset.ioi_prompts]

    def patch_particular_token(
        z, source_act, hook, token_type
    ):  # we patch at the "to" token. We have to use custom patching when we specify particular tokens to ablate.
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[token_type]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[token_type]
        ]
        return z

    config = PatchingConfig(
        source_dataset=abca_dataset.text_prompts,
        target_dataset=ioi_dataset.text_prompts,
        target_module="attn_head",
        head_circuit="result",  # we patch "result", the result of the attention head
        cache_act=True,
        verbose=False,
        patch_fn=partial(patch_particular_token, token_type="IO"),  # AND CHANGE THIS SHIT!
        layers=(0, layer),
    )  # we stop at layer "LAYER" because it's useless to patch after layer 9 if what we measure is attention of a head at layer 9.

# %%
def patch_positions(z, source_act, hook, positions=["S2"]):  # we patch at the "to" token
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z


patch_last_tokens = partial(patch_positions, positions=["end"])


config = PatchingConfig(
    source_dataset=abca_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",  # we patch "result", the result of the attention head
    cache_act=True,
    verbose=False,
    patch_fn=patch_last_tokens,
    layers=(0, max(layers) - 1),
)  # we stop at layer "LAYER" because it's useless to patch after layer 9 if what we measure is attention of a head at layer 9.

metric = ExperimentMetric(attention_probs, config.target_dataset, relative_metric=False, scalar_metric=False)

patching = EasyPatching(model, config, metric)
result = patching.run_patching()

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Variation in attention probs of Head {str(heads_to_measure)} from token "to" to {key} after Patching ABC->ABB on "to"',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )
    fig.write_image(f"svgs/variation_average_nm_attn_prob_key_{key}_patching_ABC_END.svg")
    fig.show()
# %% [markdown]
# We can clearly identify the S2-inhibition heads: 8.6, 8.10, 7.3 and 7.9. Patching them with activation from ABC causes 9.9 to pay less attention to IO and more to S and S2. To have a a better sense of what is going on behind these plots, we can see how patching impact the attention patterns of 9.9 on sample sentences.

# To have more confidence in these result, we can run a similar experiment by ABC-knock out the same heads
# %% Cross check with
def ablate_end(z, mean, hook):
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"]] = mean[
        torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"]
    ]
    return z


abl_config = AblationConfig(
    abl_type="custom",
    mean_dataset=abca_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    abl_fn=ablate_end,
    layers=(0, max(layers) - 1),
)

metric = ExperimentMetric(attention_probs, ioi_dataset.text_prompts, relative_metric=False, scalar_metric=False)

ablation = EasyAblation(
    model,
    abl_config,
    metric,
    semantic_indices=None,
    mean_by_groups=True,
    groups=ioi_dataset.groups,  # abc ablation
)

result = ablation.run_ablation()

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Average attention proba of Heads {str(heads_to_measure)}  from token "to" to {key} after ABC-ablation END',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )

    fig.write_image(f"svgs/ABC-ablation at END average nm probs to key-{key} at {ctime()}.svg")
    fig.show()
# %%
print_gpu_mem()
# %% [markdown]
# #### Plotting attention patterns
# %%
LAYER = 9
HEAD = 9

IDX = 8
model.reset_hooks()  ##before patching
show_attention_patterns(
    model,
    [(9, 9)],
    ioi_dataset[IDX : IDX + 1],
    mode="val",
    title_suffix=" Pre-patching",
)
# %%
def one_sentence_patching(z, source_act, hook):  # we patch at the "to" token
    # print(source_act.shape, z.shape)
    z[0, ioi_dataset.word_idx["end"][IDX]] = source_act[0, ioi_dataset.word_idx["end"][IDX]]
    return z


config2 = PatchingConfig(
    source_dataset=abca_dataset.text_prompts[IDX : IDX + 1],
    target_dataset=ioi_dataset.text_prompts[IDX : IDX + 1],
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=one_sentence_patching,
    layers=(0, LAYER),
)

metric2 = ExperimentMetric(
    lambda x, y: 0,
    dataset=ioi_dataset.text_prompts[IDX : IDX + 1],
    relative_metric=False,
    scalar_metric=False,
)

patching2 = EasyPatching(model, config2, metric2)

l, h = (8, 6)  # (8,10), (7,3), (7,9)]:
hk_name, hk = patching2.get_hook(l, h)
model.add_hook(hk_name, hk)  # we patch head 8.6
show_attention_patterns(
    model,
    [(9, 9)],
    ioi_dataset[IDX : IDX + 1],
    mode="val",
    title_suffix=" Post-patching",
)
# %% [markdown]
# Here we plotted attention probas weighed by the values. We observe that patching one single head at one tokens reduce by 40% the attentions on the IO token.
#
# Takeaway from the experiments: there are some information written in the residual stream at the END token by S2-inhibition heads that are used by name mover heads to generate their attention patterns. This is an example of Q-composition.
#
# Attention pattern of S2-inihibition heads. They seems to generally track the subject on key words such as "and" and "to".

# %%
show_attention_patterns(model, [(7, 3), (7, 9), (8, 6), (8, 10)], ioi_dataset[IDX : IDX + 1])

# %% [markdown]
# #### Patching at S2
# %% [markdown]
# What happend if we patch at S2 instead of END?
# %%
LAYER = 9
positions = ["S+1"]
warning.warn("This seems important for the future cells...")
patcher = partial(patch_positions, positions=positions)

config = PatchingConfig(
    source_dataset=abca_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=patcher,
    layers=(0, max(layers) - 1),
)

metric = ExperimentMetric(attention_probs, config.target_dataset, relative_metric=False, scalar_metric=False)
patching = EasyPatching(model, config, metric)
result = patching.run_patching()

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Average attention proba of Heads {str(heads_to_measure)} from token "to" to {key} after Patching ABC->ABB on {positions}',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )

    fig.write_image(f"svgs/patching at S2 average nm {key} at {ctime()}.svg")
    fig.show()
#%% [markdown] ...  but does anything change when we choose the metric of probability of S Inhibition heads?
# ARTHUR GOT CARRIED AWAY WITH EXPERIMENTS: SOME OF THESE CELLS ARE NOT IMPORTANT
from ioi_utils import attention_on_token

LAYER = 5
HEAD = 5


def metric_function(model, text_prompts):
    # return attention_on_token(model, ioi_dataset[:len(text_prompts)], layer=LAYER, head_idx=HEAD, token="S2")
    return attention_on_token(model, ioi_dataset[: len(text_prompts)], layer=LAYER, head_idx=HEAD, token="S+1")


# metric_function =

config = PatchingConfig(
    source_dataset=abca_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=patcher,
    layers=(0, LAYER - 1),
)
metric = ExperimentMetric(metric_function, config.target_dataset, relative_metric=False, scalar_metric=False)
patching = EasyPatching(model, config, metric)
result = patching.run_patching()
#%%
model.reset_hooks()
baseline = metric_function(model, ioi_dataset.text_prompts)
print("The baseline is", baseline)

fig = px.imshow(
    result[:, :] - baseline,
    labels={"y": "Layer", "x": "Head"},
    title=f'Average {LAYER}.{HEAD} from token "to" to {key} after Patching ABC->ABB on {positions}',
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu",
)

fig.write_image(f"svgs/patching at S2 average S2 at {ctime()}.svg")
fig.show()
# %% cross check with the same experiment with ablation
def ablate_s2(z, mean, hook):
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S2"]] = mean[
        torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S2"]
    ]
    return z


abl_config = AblationConfig(
    abl_type="custom",
    mean_dataset=abca_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    abl_fn=ablate_s2,
    layers=(0, max(layers) - 1),
)

metric = ExperimentMetric(attention_probs, ioi_dataset.text_prompts, relative_metric=False, scalar_metric=False)

ablation = EasyAblation(
    model,
    abl_config,
    metric,
    semantic_indices=None,
    mean_by_groups=True,
    groups=ioi_dataset.groups,
)

result = ablation.run_ablation()

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Average attention proba of Heads {str(heads_to_measure)}  from token "to" to {key} after ABC-ablation S2',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )

    fig.write_image(f"svgs/ABC-ablation at S2 average nm probs to key-{key} at {ctime()}.svg")
    fig.show()

# %% [markdown]
# A bunch of other head appear, in layer earlier than the S2-inhibition heads: 0.1, 0.10, 3.0 and 5.5, 5.8, 5.9, 6.9. We claim that they influence the values that S2 will read. Let's visualize their attention patterns.

# %% [markdown]

# #### Duplicate tokens heads

# %%
show_attention_patterns(model, [(0, 1), (0, 10), (3, 0)], ioi_dataset[:2])

# %% [markdown]
# #### Induction-ish heads

# %%

show_attention_patterns(model, [(5, 5), (5, 8), (5, 9), (6, 9)], ioi_dataset[:2])

# %% [markdown]
# ### More patching: patching at S+1

# %% [markdown]
# Some of the important heads for 9.9 attention at S2 have attention pattern that looks like induction: they pay attention to the token that follows a previous occurence of the query, in this case, because the query is S2, the previous occurence is S.
#
# What append if we patch at the token after S? (We'll call it S+1)
#
# To do this we first have to create a new dataset where S1 is flipped compared to the original dataset.

# %%
acba_dataset = ioi_dataset.gen_flipped_prompts("S")  # we flip the first occurence of S
acba_dataset.text_prompts[0], ioi_dataset.text_prompts[0]

# %%


def patch_s_plus_1(z, source_act, hook):  # we patch at the "to" token
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S"] + 1] = source_act[
        torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S"] + 1
    ]
    return z


config = PatchingConfig(
    source_dataset=acba_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=patch_s_plus_1,
    layers=(0, max(layers) - 1),
)

metric = ExperimentMetric(attention_probs, config.target_dataset, relative_metric=False, scalar_metric=False)

patching = EasyPatching(model, config, metric)
result = patching.run_patching()

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Average attention proba of Heads {str(heads_to_measure)}  from token "to" to {key} after Patching ABC->ABB on S+1',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )

    fig.write_image(f"svgs/patching average S+1 (ACB, ABB) nm {key} at {ctime()}.svg")
    fig.show()


# %% Check with a similar ablation


def ablate_s_plus_1(z, mean, hook):
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S"] + 1] = mean[
        torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S"] + 1
    ]
    return z


abl_config = AblationConfig(
    abl_type="custom",
    mean_dataset=abca_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    abl_fn=ablate_s_plus_1,
    layers=(0, max(layers) - 1),
)

metric = ExperimentMetric(attention_probs, ioi_dataset.text_prompts, relative_metric=False, scalar_metric=False)

ablation = EasyAblation(
    model,
    abl_config,
    metric,
    semantic_indices=None,
    mean_by_groups=True,
    groups=ioi_dataset.groups,
)

result = ablation.run_ablation()

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Average attention proba of Heads {str(heads_to_measure)}  from token "to" to {key} after ABC-ablation S+1',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )

    fig.write_image(f"svgs/ABC-ablation at S+1 average nm probs to key-{key} at {ctime()}.svg")
    fig.show()


# %% [markdown]
# It seems that the heads 4.11, 4.7, 4.3, 2.2 and 5.6 are important. Let's look at their patterns. The majority of them look like they're attending to the previous tokens.

# %%
show_attention_patterns(model, [(4, 7), (5, 6), (4, 11), (2, 2), (4, 3)], ioi_dataset[34:35])

# %% [markdown]
# Here is the (approximative) story of what's going on here:
# * Early layers heads implement duplicate test: "if the key and queries are the same token, copy the token in the residual stream". This has for effect to write "S is duplicate" on the S2 residual stream.
# * In parallel, previous tokens heads copy information about the previous token.
# * At S2, induction heads find a match: S2 match the information written at S+1 about S by a previous token head, so they have a trong attention proba on S+1. However, contrary to plain induction, they don't copy information about S+1, they'll also write information about S in their residual stream. This is another source of information "S is a duplicate token" at S2.
# * On the END token, S2-Inhibition heads use value composition to copy information about duplicate tokens from the two previus sources. Their attention seems to be "look for the subject of the sentence". (But their attention pattern is not well understood for now)
# * The name mover heads use the value of S in their queries, with a negative sign. This has for effect to gives IO greatest scores, then their OV circuit is a simple copy for names.
# * Same story for the calibrations heads but they flip the sign of the direction (and surely something more complex is going on to calibrate the norm).

# <img src="https://i.imgur.com/PPtTQRh.png">


#%% [markdown]
# <h1><b>Faithfulness</b></h1>
# These experiments isolate the circuit, e.g by ablating all other heads.
# <h2>Ablations</h2>
# For each template, e.g `Then, [A] and [B] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]` we ablate only the indices we claim are important and we retain a positive logit difference between `IO` and `S`, as well the "score" (whether the IO logit remains in the top 10 logit AND IO > S), though have some performace degradation, particularly when we don't ablate the calibration heads.

#%% # run normal ablation experiments
num_templates = 10  # len(ABBA_TEMPLATES)
template_type = "ABBA"
if template_type == "ABBA":
    templates = ABBA_TEMPLATES[:num_templates]
elif template_type == "BABA":
    templates = BABA_TEMPLATES[:num_templates]
else:
    raise NotImplementedError()

## new logit diff definition
def logit_diff_target(model, ioi_data, target_dataset=None, all=False, std=False):
    """
    Difference between the IO and the S logits at the "to" token

    If `target_dataset` is specified, assume the IO and S tokens are the same as those in `target_dataset`
    """

    global ioi_dataset  # probably when we call this function, we want `ioi_data` to be the main dataset. So make it that way.
    if "IOIDataset" in str(type(ioi_data)):
        text_prompts = ioi_data.text_prompts
        ioi_dataset = ioi_data
    else:
        text_prompts = ioi_data

    if target_dataset is None:
        target_dataset = ioi_dataset

    logits = model(text_prompts).detach()
    L = len(text_prompts)
    IO_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"][:L],
        target_dataset.io_tokenIDs[:L],
    ]
    S_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"][:L],
        target_dataset.s_tokenIDs[:L],
    ]

    if all and not std:
        return IO_logits - S_logits
    if std:
        if all:
            first_bit = IO_logits - S_logits
        else:
            first_bit = (IO_logits - S_logits).mean().detach().cpu()
        return first_bit, torch.std(IO_logits - S_logits)
    return (IO_logits - S_logits).mean().detach().cpu()


def score(model, ioi_dataset, all=False, verbose=False):
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach()
    L = len(text_prompts)
    end_logits = logits[
        torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], :
    ]  # batch * sequence length * vocab_size
    io_logits = end_logits[torch.arange(len(text_prompts)), ioi_dataset.io_tokenIDs[:L]]
    assert len(list(end_logits.shape)) == 2, end_logits.shape
    top_10s_standard = torch.topk(end_logits, dim=1, k=10).values[:, -1]
    good_enough = end_logits > top_10s_standard.unsqueeze(-1)
    selected_logits = good_enough[torch.arange(len(text_prompts)), ioi_dataset.io_tokenIDs[:L]]

    # is IO > S ???
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
    IO_greater_than_S = (IO_logits - S_logits) > 0

    # calculate percentage passing both tests
    answer = torch.sum((selected_logits & IO_greater_than_S).float()).detach().cpu() / len(text_prompts)

    selected = torch.sum(selected_logits) / len(text_prompts)
    greater = torch.sum(IO_greater_than_S) / len(text_prompts)

    if verbose:
        print(f"Score calc: {answer}; {selected} and {greater}")
    return answer


def io_probs(model, ioi_dataset, mode="IO"):  # also S mode
    assert mode in ["IO", "S"]
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach()
    assert len(list(logits.shape)) == 3, logits.shape
    L = len(text_prompts)
    assert logits.shape[0] == L
    end_logits = logits[
        torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], :
    ]  # batch * sequence length * vocab_size
    end_probs = torch.softmax(end_logits, dim=-1)
    ids = ioi_dataset.io_tokenIDs if mode == "IO" else ioi_dataset.s_tokenIDs
    probs = end_probs[torch.arange(len(text_prompts)), ids[:L]]
    return probs.mean().detach().cpu()


N = 10  # number per template
template_prompts = [
    gen_prompt_uniform(
        templates[i : i + 1],
        NAMES,
        NOUNS_DICT,
        N=N,
        symmetric=False,
    )
    for i in range(len(templates))
]

circuit = CIRCUIT.copy()

for ablate_negative in [
    False,
    True,
]:
    ld_data = []
    for template_idx in tqdm(range(num_templates)):
        prompts = template_prompts[template_idx]
        ioi_dataset = IOIDataset(prompt_type=template_type, N=N, symmetric=False, prompts=prompts)
        abca_dataset = ioi_dataset.gen_flipped_prompts("S2")
        assert torch.all(ioi_dataset.toks != 50256)  # no padding anywhere
        assert len(ioi_dataset.sem_tok_idx.keys()) != 0, "no semantic tokens found"
        for key in ioi_dataset.sem_tok_idx.keys():
            idx = ioi_dataset.sem_tok_idx[key][0]
            assert torch.all(ioi_dataset.sem_tok_idx[key] == idx), f"{key} {ioi_dataset.sem_tok_idx[key]}"
            # check that semantic ablation = normal ablation

        model.reset_hooks()
        ld_initial = logit_diff(model, ioi_dataset)

        model.reset_hooks()

        heads_to_keep = get_heads_circuit(
            ioi_dataset,
            circuit=circuit,
            excluded_classes=["negative"] if ablate_negative else [],
        )

        model, abl = do_circuit_extraction(
            model=model,
            ioi_dataset=ioi_dataset,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            mean_dataset=abca_dataset,
        )

        ld_final = logit_diff(model, ioi_dataset)
        ld_data.append((ld_initial, ld_final))

    xs = [ld_data[i][0].item() for i in range(num_templates)]
    ys = [ld_data[i][1].item() for i in range(num_templates)]

    x_label = "Baseline Logit Diff"  # IO Probability"
    y_label = "Ablated Logit Diff"  # IO Probability"

    d = {
        x_label: xs,
        y_label: ys,
    }
    d["beg"] = [template[:10] for template in templates]
    d["sentence"] = [template for template in templates]
    d["commas"] = [template.count(",") for template in templates]

    df = pd.DataFrame(d)
    px.scatter(
        df,
        x=x_label,
        y=y_label,
        hover_data=["sentence"],
        text="beg",
        title=f"Change in logit diff when {ablate_negative=}",
    ).show()
#%% # let's check that the circuit isn't changing relative to which template we are using


N = 100  # number per template
template_prompts = [
    gen_prompt_uniform(
        templates[i : i + 1],
        NAMES,
        NOUNS_DICT,
        N=N,
        symmetric=False,
    )
    for i in range(len(templates))
]

three_d = torch.zeros(size=(num_templates, 12, 12))

for template_idx in tqdm(range(num_templates)):
    prompts = template_prompts[template_idx]
    ioi_dataset = IOIDataset(prompt_type=template_type, N=N, symmetric=False, prompts=prompts)
    assert torch.all(ioi_dataset.toks != 50256)  # no padding anywhere
    assert len(ioi_dataset.sem_tok_idx.keys()) != 0, "no semantic tokens found"
    for key in ioi_dataset.sem_tok_idx.keys():
        idx = ioi_dataset.sem_tok_idx[key][0]
        assert torch.all(ioi_dataset.sem_tok_idx[key] == idx), f"{key} {ioi_dataset.sem_tok_idx[key]}"
        # check that semantic ablation = normal ablation

    attn_vals, mlp_vals = writing_direction_heatmap(
        model,
        ioi_dataset,
        title=f"Writing Direction Heatmap for {template_idx}",
        return_vals=True,
    )
    three_d[template_idx] = attn_vals
    continue
show_pp(three_d, animate_axis=0, title="Writing Direction Heatmap for all templates")


#%% [markdown]
# <h2>Circuit extraction</h2>
#%%
ld = logit_diff_target(model, ioi_dataset[:N], all=True)


from ioi_circuit_extraction import turn_keep_into_rmv, list_diff

# %% # sanity check
if True:  # I think this is some test
    type(ioi_dataset)
    old_ld = logit_diff_target(model, ioi_dataset[:N])
    model, abl_cricuit_extr = do_circuit_extraction(
        heads_to_remove={
            (0, 4): [list(range(ioi_dataset.max_len)) for _ in range(N)]
        },  # annoyingly sometimes needs to be edited...
        mlps_to_remove={},
        heads_to_keep=None,
        mlps_to_keep=None,
        model=model,
        ioi_dataset=ioi_dataset[:N],
        metric=logit_diff_target,
    )
    ld = logit_diff_target(model, ioi_dataset[:N])
    metric = ExperimentMetric(
        metric=logit_diff_target,
        dataset=ioi_dataset.text_prompts[:N],
        relative_metric=False,
    )
    config = AblationConfig(
        abl_type="mean",
        mean_dataset=ioi_dataset.text_prompts[:N],
        target_module="attn_head",
        head_circuit="result",
        cache_means=True,
    )
    abl = EasyAblation(
        model,
        config,
        metric,
        semantic_indices=ioi_dataset[:N].sem_tok_idx,
        mean_by_groups=True,  # TO CHECK CIRCUIT BY GROUPS
        groups=ioi_dataset.groups,
    )
    res = abl.run_experiment()
    print(ld, res[:5, :5])


#%%
def score_target(model, ioi_dataset, k=1, target_dataset=None, all=False):
    if target_dataset is None:
        target_dataset = ioi_dataset
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach()
    # print(get_corner(logits))
    # print(text_prompts[:2])
    L = len(text_prompts)
    end_logits = logits[
        torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], :
    ]  # batch * sequence length * vocab_size
    io_logits = end_logits[torch.arange(len(text_prompts)), target_dataset.io_tokenIDs[:L]]
    assert len(list(end_logits.shape)) == 2, end_logits.shape
    top_10s_standard = torch.topk(end_logits, dim=1, k=k).values[:, -1]
    good_enough = end_logits >= top_10s_standard.unsqueeze(-1)
    selected_logits = good_enough[torch.arange(len(text_prompts)), target_dataset.io_tokenIDs[:L]]
    # print(torch.argmax(end_logits, dim=-1))
    # is IO > S ???
    IO_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"][:L],
        target_dataset.io_tokenIDs[:L],
    ]
    S_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"][:L],
        target_dataset.s_tokenIDs[:L],
    ]
    IO_greater_than_S = (IO_logits - S_logits) > 0

    # calculate percentage passing both tests
    answer = torch.sum((selected_logits & IO_greater_than_S).float()).detach().cpu() / len(text_prompts)

    selected = torch.sum(selected_logits) / len(text_prompts)
    greater = torch.sum(IO_greater_than_S) / len(text_prompts)

    # print(f"Kevin gives: {answer}; {selected} and {greater}")
    return answer


def print_top_k(model, ioi_dataset, K=1, n=10):
    logits = model(ioi_dataset.text_prompts).detach()
    end_logits = logits[
        torch.arange(len(ioi_dataset.text_prompts)), ioi_dataset.word_idx["end"], :
    ]  # batch * sequence length * vocab_size
    probs = np.around(torch.nn.functional.log_softmax(end_logits, dim=-1).cpu().numpy(), 2)
    topk = torch.topk(end_logits, dim=1, k=K).indices
    for x in range(n):
        print("-------------------")
        print(ioi_dataset.text_prompts[x])
        print(
            " ".join(
                [f"({i+1}):{model.tokenizer.decode(token)} : {probs[x][token]}" for i, token in enumerate(topk[x])]
            )
        )


# %%  Running circuit extraction

from ioi_circuit_extraction import (
    join_lists,
    CIRCUIT,
    RELEVANT_TOKENS,
    get_extracted_idx,
    get_heads_circuit,
)

heads_to_keep = get_heads_circuit(ioi_dataset)

mlps_to_keep = {}

model.reset_hooks()
old_ld, old_std = logit_diff_target(model, ioi_dataset, all=True, std=True)
old_score = score_target(model, ioi_dataset)
model.reset_hooks()
model, _ = do_circuit_extraction(
    mlps_to_remove={},  # we have to keep every mlps
    heads_to_keep=heads_to_keep,
    model=model,
    ioi_dataset=ioi_dataset,
    metric=logit_diff_target,
)

ldiff, std = logit_diff_target(model, ioi_dataset, std=True, all=True)
score = score_target(model, ioi_dataset)  # k=K ??

# %%
print(f"Logit difference = {ldiff.mean().item()} +/- {std}. score={score.item()}")
print(f"Original logit_diff_target = {old_ld.mean()} +/- {old_std}. score={old_score}")

df = pd.DataFrame(
    {
        "Logit difference": ldiff.cpu(),
        "Random (for separation)": np.random.random(len(ldiff)),
        "beg": [prompt[:10] for prompt in ioi_dataset.text_prompts],
        "sentence": [prompt for prompt in ioi_dataset.text_prompts],
        "#tokens before first name": [prompt.count("Then") for prompt in ioi_dataset.text_prompts],
        "template": ioi_dataset.templates_by_prompt,
        "misc": [
            (str(prompt.count("Then")) + str(ioi_dataset.templates_by_prompt[i]))
            for (i, prompt) in enumerate(ioi_dataset.text_prompts)
        ],
    }
)

px.scatter(
    df,
    x="Logit difference",
    y="Random (for separation)",
    hover_data=["sentence", "template"],
    text="beg",
    color="misc",
    title=f"Prompt type = {ioi_dataset.prompt_type}",
)


# %%
for key in ioi_dataset.word_idx:
    print(key, ioi_dataset.word_idx[key][8])

# %% [markdown]
# <h2>Global patching</h2>

# %%
def do_global_patching(
    source_mlps_to_patch=None,
    source_mlps_to_keep=None,
    target_mlps_to_patch=None,
    target_mlps_to_keep=None,
    source_heads_to_keep=None,
    source_heads_to_patch=None,
    target_heads_to_keep=None,
    target_heads_to_patch=None,
    source_ioi_dataset=None,
    target_ioi_dataset=None,
    model=None,
):
    """
    if `ablate` then ablate all `heads` and `mlps`
        and keep everything else same
    otherwise, ablate everything else
        and keep `heads` and `mlps` the same
    """
    # check if we are either in keep XOR remove move from the args

    patching, heads, mlps = get_circuit_replacement_hook(
        target_heads_to_patch,  # head
        target_mlps_to_patch,
        target_heads_to_keep,
        target_mlps_to_keep,
        source_heads_to_patch,  # head2
        source_mlps_to_patch,
        source_heads_to_keep,
        source_mlps_to_keep,
        target_ioi_dataset,
        model,
    )
    metric = ExperimentMetric(lambda x: 0, [])

    config = PatchingConfig(
        patch_fn=patching,
        source_dataset=source_ioi_dataset.text_prompts,  # TODO nb of prompts useless ?
        target_dataset=target_ioi_dataset.text_prompts,
        target_module="attn_head",
        head_circuit="result",
        verbose=True,
        cache_act=True,
    )
    ptch = EasyPatching(
        model,
        config,
        metric,
    )
    model.reset_hooks()

    for layer, head in heads.keys():
        model.add_hook(*ptch.get_hook(layer, head))
    for layer in mlps.keys():
        model.add_hook(*ptch.get_hook(layer, head=None, target_module="mlp"))

    return model, ptch


# %%
N = 100
target_ioi_dataset = IOIDataset(
    prompt_type="mixed", N=N, symmetric=True, prefixes=None
)  # annoyingly you could swap "target" and "source" as names, and I think the original dataset would be a "source" in some ways (a bit confusing!)
source_ioi_dataset = target_ioi_dataset.gen_flipped_prompts("IO")

# %%
print(
    "The target dataset (has both ABBA and BABA sentences):",
    target_ioi_dataset.text_prompts[:3],
)
print(
    "The source dataset (with IO randomly changed):",
    source_ioi_dataset.text_prompts[:3],
)

source_heads_to_keep, source_mlps_to_keep = get_heads_circuit(
    source_ioi_dataset, excluded_classes=["negative"], mlp0=True
)
target_heads_to_keep, target_mlps_to_keep = get_heads_circuit(
    target_ioi_dataset, excluded_classes=["negative"], mlp0=True
)

K = 1
model.reset_hooks()
old_ld, old_std = logit_diff_target(model, target_ioi_dataset, target_dataset=target_ioi_dataset, all=True, std=True)
model.reset_hooks()
old_score = score_target(model, target_ioi_dataset, target_dataset=target_ioi_dataset, k=K)
model.reset_hooks()
old_ld_source, old_std_source = logit_diff_target(
    model, target_ioi_dataset, target_dataset=source_ioi_dataset, all=True, std=True
)
model.reset_hooks()
old_score_source = score_target(model, target_ioi_dataset, target_dataset=source_ioi_dataset, k=K)
model.reset_hooks()
model, _ = do_global_patching(
    source_mlps_to_patch=source_mlps_to_keep,
    source_mlps_to_keep=None,
    target_mlps_to_patch=target_mlps_to_keep,
    target_mlps_to_keep=None,
    source_heads_to_keep=None,
    source_heads_to_patch=source_heads_to_keep,
    target_heads_to_keep=None,
    target_heads_to_patch=target_heads_to_keep,
    model=model,
    source_ioi_dataset=source_ioi_dataset,
    target_ioi_dataset=target_ioi_dataset,
)

ldiff_target, std_ldiff_target = logit_diff_target(
    model, target_ioi_dataset, target_dataset=target_ioi_dataset, std=True, all=True
)
score_target_result = score_target(model, target_ioi_dataset, target_dataset=target_ioi_dataset, k=K)
ldiff_source, std_ldiff_source = logit_diff_target(
    model, target_ioi_dataset, target_dataset=source_ioi_dataset, std=True, all=True
)
score_source = score_target(model, target_ioi_dataset, target_dataset=source_ioi_dataset, k=K)
# %%
print(f"Original logit_diff on TARGET dataset (no patching yet!) = {old_ld.mean()} +/- {old_std}. Score {old_score}")
print(
    f"Original logit_diff on SOURCE dataset (no patching yet!) = {old_ld_source.mean()} +/- {old_std_source}. Score {old_score_source}"
)
print(
    f"logit_diff on TARGET dataset (*AFTER* patching) = {ldiff_target.mean()} +/- {std_ldiff_target}. Score {score_target_result}"
)
print(
    f"Logit_diff on SOURCE dataset (*AFTER* patching) = {ldiff_source.mean()} +/- {std_ldiff_source}. Score {score_source}"
)
df = pd.DataFrame(
    {
        "Logit difference in source": ldiff_source.cpu(),
        "Random (for interactivity)": np.random.random(len(ldiff_source)),
        "beg": [prompt["text"][:10] for prompt in ioi_dataset.ioi_prompts],
        "sentence": [prompt["text"] for prompt in ioi_dataset.ioi_prompts],
        "#tokens before first name": [prompt["text"].count("Then") for prompt in ioi_dataset.ioi_prompts],
        "template": ioi_dataset.templates_by_prompt,
        "misc": [
            (str(prompt["text"].count("Then")) + str(ioi_dataset.templates_by_prompt[i]))
            for (i, prompt) in enumerate(ioi_dataset.ioi_prompts)
        ],
    }
)

px.scatter(
    df,
    x="Logit difference in source",
    y="Random (for interactivity)",
    hover_data=["sentence", "template"],
    text="beg",
    color="misc",
    title=ioi_dataset.prompt_type,
)
#%% evidence for the S2 story
# ablating V for everywhere except S2 barely affects LD. But ablating all V has LD go to almost 0
from ioi_circuit_extraction import ARTHUR_CIRCUIT, get_extracted_idx, RELEVANT_TOKENS

heads_to_keep = {}

for circuit_class in ARTHUR_CIRCUIT.keys():
    for head in ARTHUR_CIRCUIT[circuit_class]:
        heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)

model, abl = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)

for layer, head_idx in [(7, 9), (8, 6), (7, 3), (8, 10)]:
    # use abl.mean_cache
    cur_tensor_name = f"blocks.{layer}.attn.hook_v"
    s2_token_idxs = get_extracted_idx(["S2"], ioi_dataset)
    mean_cached_values = abl.mean_cache[cur_tensor_name][:, :, head_idx, :].cpu().detach()

    def s2_v_ablation_hook(z, act, hook):  # batch, seq, head dim, because get_act_hook hides scary things from us
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_v", hook.name
        assert len(list(z.shape)) == 3, z.shape
        assert list(z.shape) == list(act.shape), (z.shape, act.shape)

        true_s2_values = z[:, s2_token_idxs, :].clone()
        z = mean_cached_values.cuda()  # hope that we don't see chaning values of mean_cached_values...
        # z[:, s2_token_idxs, :] = true_s2_values

        return z

    cur_hook = get_act_hook(
        s2_v_ablation_hook,
        alt_act=abl.mean_cache[cur_tensor_name],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(cur_tensor_name, cur_hook)

new_ld = logit_diff(model, ioi_dataset)
#%% # how frequently does the model predict the correct name?
for i in tqdm(range(1000)):
    ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
    prbs = probs(model, ioi_dataset)
    print(prbs.mean())


# %% Investigation of identified heads on random tokens
seq_len = 100
rand_tokens = torch.randint(1000, 10000, (4, seq_len))
rand_tokens_repeat = einops.repeat(rand_tokens, "batch pos -> batch (2 pos)")


induction_scores_array = np.zeros((model.cfg.n_layers, model.cfg.n_heads))


def calc_induction_score(attn_pattern, hook):
    # Pattern has shape [batch, index, query_pos, key_pos]
    induction_stripe = attn_pattern.diagonal(1 - seq_len, dim1=-2, dim2=-1)
    induction_scores = einops.reduce(induction_stripe, "batch index pos -> index", "mean")
    # Store the scores in a common array
    induction_scores_array[hook.layer()] = induction_scores.detach().cpu().numpy()


def filter_attn_hooks(hook_name):
    split_name = hook_name.split(".")
    return split_name[-1] == "hook_attn"


induction_logits = model.run_with_hooks(rand_tokens_repeat, fwd_hooks=[(filter_attn_hooks, calc_induction_score)])
px.imshow(induction_scores_array, labels={"y": "Layer", "x": "Head"}, color_continuous_scale="Blues")


# %%
from ioi_utils import compute_next_tok_dot_prod
import torch.nn.functional as F

IDX = 0


def zero_ablate(hook, z):
    return torch.zeros_like(z)


head_mask = torch.empty((12, 12), dtype=torch.bool)
head_mask = torch.fill(head_mask, False)
head_mask[5, 5] = True
head_mask[6, 9] = False

attn_head_mask = head_mask


def prune_attn_heads(value, hook):
    # Value has shape [batch, pos, index, d_head]
    mask = head_mask[hook.layer()]
    value[:, :, mask] = 0.0
    return value


def filter_value_hooks(name):
    return name.split(".")[-1] == "hook_v"


def compute_logit_probs(rand_tokens_repeat, model):
    induction_logits = model(rand_tokens_repeat)
    induction_log_probs = F.log_softmax(induction_logits, dim=-1)
    induction_pred_log_probs = torch.gather(
        induction_log_probs[:, :-1].cuda(), -1, rand_tokens_repeat[:, 1:, None].cuda()
    )[..., 0]
    return induction_pred_log_probs[:, seq_len:].mean().cpu().detach().numpy()


compute_logit_probs(rand_tokens_repeat, model)

# %%
induct_head = [(5, 1), (7, 2), (7, 10), (6, 9)]
all_means = []
for k in range(len(induct_head) + 1):
    results = []
    for _ in range(10):
        head_mask = torch.empty((12, 12), dtype=torch.bool)
        head_mask = torch.fill(head_mask, False)
        rd_set = rd.sample(induct_head, k=k)
        for (l, h) in rd_set:
            head_mask[l, h] = True

        def prune_attn_heads(value, hook):
            # Value has shape [batch, pos, index, d_head]
            mask = head_mask[hook.layer()]
            value[:, :, mask] = 0.0
            return value

        model.reset_hooks()
        model.add_hook(filter_value_hooks, prune_attn_heads)
        results.append(compute_logit_probs(rand_tokens_repeat, model))

    results = np.array(results)
    all_means.append(results.mean())

fig = px.bar(
    all_means, title="Loss on repeated random tokens sequences (average on 10 random set of KO heads) 5.5 excluded"
)


fig.update_layout(
    xaxis_title="Number of induction head zero-KO",
    yaxis_title="Induction loss",
)
fig.show()
# %%
model.add_hook(filter_value_hooks, prune_attn_heads)


# model.reset_hooks()
prod = np.zeros((model.cfg.n_layers, model.cfg.n_heads, len(rand_tokens_repeat[IDX])))
for layer in range(12):
    for head in range(12):
        print(f"Layer {layer}, head {head}")
        prod[layer, head, :] = np.concatenate(
            [
                np.array([0]),
                np.array(
                    compute_next_tok_dot_prod(
                        model, rand_tokens_repeat[IDX : IDX + 1], layer, head, seq_tokenized=True
                    )[IDX]
                ),
            ]
        )


# %%


POS = 130
K = 1
for x in np.random.randint(0, seq_len, K):
    print(f"Position {x}")
    fig = px.imshow(prod[:, :, x], labels={"y": "Layer", "x": "Head"}, color_continuous_scale="Blues")
    fig.show()

for x in np.random.randint(seq_len, 2 * seq_len, K):
    print(f"Position {x}")
    fig = px.imshow(prod[:, :, x], labels={"y": "Layer", "x": "Head"}, color_continuous_scale="Blues")
    fig.show()

# %%


def sort_mtx(mtx, return_idx=False, sort_by_abs=False):
    mtx_flat = mtx.flatten()
    if sort_by_abs:
        all_sorted_idx = np.abs(mtx_flat).argsort()
    else:
        all_sorted_idx = mtx_flat.argsort()

    sorted_idx = []
    sorted_values = []
    for i in range(len(all_sorted_idx)):
        # print(np.unravel_index(all_sorted_idx[-i - 1], mtx.shape))
        x, y = np.unravel_index(all_sorted_idx[-i - 1], mtx.shape)
        sorted_idx.append((x, y))
        sorted_values.append(mtx[x, y])
    if return_idx:
        return sorted_values, sorted_idx
    else:
        return sorted_values


# %%

important_heads = []
# get heads with average of absolute dot product above 10
for layer in range(12):
    for head in range(12):
        if np.mean(np.abs(prod[layer, head, seq_len:])) > 5:
            important_heads.append((layer, head))

#%%
circuit_heads = []
for n in CIRCUIT.keys():
    circuit_heads += CIRCUIT[n]

# %%

vals, idx = sort_mtx(np.abs(prod[:, :, seq_len:]).mean(axis=-1), return_idx=True, sort_by_abs=True)


colors = [((l, h) in circuit_heads) for (l, h) in idx]

fig = px.bar(
    x=[f"{x[0]}.{x[1]}" for x in idx],
    y=vals,
    color=colors,
    title="Average absolute dot prod with next token on repeated sequence (after zero-KO of 6.9 and 5.5)",
)

fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": [f"{x[0]}.{x[1]}" for x in idx]})
fig.show()

# %%
heads = [(10, 10), (9, 6), (10, 6), (10, 7), (11, 10), (10, 1), (9, 9), (10, 2), (7, 2), (9, 1), (9, 0), (11, 7)]

all_prods = {f"{layer}.{head}": prod[layer, head, :] for (layer, head) in important_heads}

px.line(all_prods, labels={"index": "Position in sequence of random tokens", "value": "Dot product"})

# %%
