#%%
from curses import A_ALTCHARSET
import warnings

from ioi_utils import logit_diff, probs
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
import warnings
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
N = 200
ioi_dataset_baba = IOIDataset(prompt_type="BABA", N=N, tokenizer=model.tokenizer)
ioi_dataset_abba = IOIDataset(prompt_type="ABBA", N=N, tokenizer=model.tokenizer)
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts("S2")  # we flip the second b for a random c
pprint(abca_dataset.text_prompts[:5])


acc_dataset = ioi_dataset.gen_flipped_prompts("S")
dcc_dataset = acc_dataset.gen_flipped_prompts("IO")


acba_dataset = ioi_dataset.gen_flipped_prompts("S1")  # we flip the first occurence of S
acba_dataset.text_prompts[0], ioi_dataset.text_prompts[0]


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


def patch_positions(z, source_act, hook, positions=["S2"]):  # we patch at the "to" token
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z


#################### END SETUP ####################
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
    layers=(0, 9 - 1),
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


# %% Redo the patching experiment by freezing the induction heads


cache = {}


def filter_induct_heads(name):
    if "attn.hook_result" in name:
        layer = int(name.split(".")[1])
        if layer in [5, 6]:
            return True
    return False


model.reset_hooks()
model.cache_some(cache, filter_induct_heads)
logit = model(ioi_dataset.text_prompts)
model.reset_hooks()


induct_heads = {5: [5, 8, 9], 6: [9]}
missing = "5.9"


def freeze_attention_head(z, hook):
    layer = int(hook.name.split(".")[1])
    z[:, :, induct_heads[layer], :] = cache[hook.name][:, :, induct_heads[layer], :]
    return z


patching.other_hooks = [(filter_induct_heads, freeze_attention_head)]

result = patching.run_patching()


for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Average attention proba {str(heads_to_measure)} from token "to" to {key} after Patching ABC->ABB on S+1. Working head: {missing}',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )
    fig.show()
# %% ########################## at S2 position -- S-IN EXPERIMENTS ##########################

s_p_1_flipped = ioi_dataset.gen_flipped_prompts("S+1")
positions = ["S2"]

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
# %%
patching.other_hooks = []
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

# %%
heads = {7: [3, 9], 8: [6, 10], 9: [6, 9], 10: [0]}  # 7: [3, 9], 8: [6, 10], 9: [6, 9], 10: [0]


for l in range(12):
    if l not in heads:
        heads[l] = []
# heads = {7: list(range(12)), 8: [6, 10]}  # , 9: [6, 9], 10: [0]

nm = [(9, 6), (9, 9), (10, 0)]

layers_to_freeze = list(heads.keys())
all_execept_heads = {l: [x for x in range(12) if x not in heads[l]] for l in layers_to_freeze}

use_complement = True
if use_complement:
    heads_to_freeze = all_execept_heads
else:
    heads_to_freeze = heads

mlp_to_freeze = [7, 8, 9, 10, 11]

# %%
def filter_act(name):
    if "attn.hook_result" in name:
        layer = int(name.split(".")[1])
        if layer in layers_to_freeze:
            return True
    if "hook_mlp_out" in name:
        layer = int(name.split(".")[1])
        if layer in mlp_to_freeze:
            return True
    return False


cache = {}

model.reset_hooks()
model.cache_some(cache, filter_act)
logit = model(ioi_dataset.text_prompts)
model.reset_hooks()


def freeze_attention_head_end(z, hook):
    layer = int(hook.name.split(".")[1])
    # print("yo")

    # print shape
    # print(z[range(ioi_dataset.N), ioi_dataset.word_idx["end"]][:, heads[layer], :].shape)
    if "attn.hook_result" in hook.name:
        rge = torch.tensor(list(range(ioi_dataset.N))).unsqueeze(1)
        end = torch.tensor(ioi_dataset.word_idx["end"]).unsqueeze(1)
        # print(z[rge, end, heads[layer], :].shape)

        z[rge, end, heads_to_freeze[layer], :] = cache[hook.name][rge, end, heads_to_freeze[layer], :]
    if "hook_mlp_out" in hook.name:
        z[range(ioi_dataset.N), ioi_dataset.word_idx["end"], :] = cache[hook.name][
            range(ioi_dataset.N), ioi_dataset.word_idx["end"], :
        ]
    return z
    # print(layer, heads[layer])
    # print(z[range(ioi_dataset.N), ioi_dataset.word_idx["S2"]][:, heads[layer], :].shape)


# %%


patching.other_hooks = [(filter_act, freeze_attention_head_end)]
# patching.other_hooks.append((filter_induct_heads, freeze_attention_head))

result = patching.run_patching()
# %%
for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=(
            f"Average attention proba of Heads {str(heads_to_measure)} from token 'to' to {key} after Patching ABC->ABB on {positions} <br>"
            + (f"Freezing all heads at END except {heads}" if use_complement else f"Freezing at END {heads}")
            + f"<br> MLP frozen at END {mlp_to_freeze}"
        ),
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )

    # fig.write_image(f"svgs/patching at S2 average nm {key} at {ctime()}.svg")
    fig.show()

# %%


from ioi_utils import logit_diff

model.reset_hooks()
print(logit_diff(model, ioi_dataset))
model.reset_hooks()
model.add_hook(filter_act, freeze_attention_head_end)

logit_diff(model, ioi_dataset)

# %%
IDX = 50


def one_sentence_patching(z, source_act, hook):  # we patch at the "to" token
    # print(source_act.shape, z.shape)
    z[0, ioi_dataset.word_idx["S2"][IDX]] = source_act[0, ioi_dataset.word_idx["S2"][IDX]]
    return z


def freeze_attention_head_end_one_sentence(z, hook):
    layer = int(hook.name.split(".")[1])
    # print("yo")

    # print shape
    # print(z[range(ioi_dataset.N), ioi_dataset.word_idx["end"]][:, heads[layer], :].shape)

    # rge = torch.tensor(range(ioi_dataset.N)).unsqueeze(1)
    # end = torch.tensor(ioi_dataset.word_idx["end"]).unsqueeze(1)
    # print(z[rge, end, heads[layer], :].shape)

    z[0, ioi_dataset.word_idx["end"][IDX], heads[layer], :] = cache[hook.name][
        IDX, ioi_dataset.word_idx["end"][IDX], heads[layer], :
    ]
    return z
    # print(layer, heads[layer])
    # print(z[range(ioi_dataset.N), ioi_dataset.word_idx["S2"]][:, heads[layer], :].shape)


config2 = PatchingConfig(
    source_dataset=abca_dataset.text_prompts[IDX : IDX + 1],
    target_dataset=ioi_dataset.text_prompts[IDX : IDX + 1],
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=one_sentence_patching,
    layers=(0, max(layers) - 1),
)

metric2 = ExperimentMetric(
    lambda x, y: 0,
    dataset=ioi_dataset.text_prompts[IDX : IDX + 1],
    relative_metric=False,
    scalar_metric=False,
)

patching2 = EasyPatching(model, config2, metric2)

l, h = (5, 5)  # (8,10), (7,3), (7,9)]:
hk_name, hk = patching2.get_hook(l, h)
model.add_hook(hk_name, hk)  # we patch head 8.6
model.add_hook(filter_act, freeze_attention_head_end_one_sentence)
show_attention_patterns(
    model,
    [(9, 9)],
    ioi_dataset[IDX : IDX + 1],
    mode="attn",
    title_suffix=" Post-patching",
)


# %%

model.reset_hooks()
show_attention_patterns(
    model,
    [(9, 9)],
    ioi_dataset[IDX : IDX + 1],
    mode="attn",
    title_suffix=" Pre-patching",
)

##################### Investigating the ACC patching experments #####################

# %% Investigating the ACC patching experments


# %%

positions = ["end"]
patcher = partial(patch_positions, positions=positions)

config = PatchingConfig(
    source_dataset=dcc_dataset.text_prompts,  # abca_dataset.text_prompts,
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


# %%
patching.other_hooks = []
result = patching.run_patching()

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Average attention proba of Heads {str(heads_to_measure)} from token "to" to {key} after Patching ABC->ABB on {positions}',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )
    fig.write_image(f"svgs/patching at {positions} average nm {key} at {ctime()}.svg")
    fig.show()

# %%
from ioi_utils import probs


ld = []
pr = []


for L in range(-1, 12):
    all_heads = [(l, h) for h in range(12) for l in range(L + 1)]

    model.reset_hooks()
    c = {}

    # for l in range(L + 1):
    #     hk_name, hk = patching.get_hook(l, target_module="mlp")
    #     model.add_hook(hk_name, hk)

    for l, h in all_heads:  # [(7, 3), (7, 9), (8, 6), (8, 10)]:
        hk_name, hk = patching.get_hook(l, h)
        model.add_hook(hk_name, hk)

    model.cache_all(c)
    ld.append(logit_diff(model, ioi_dataset))
    pr.append(probs(model, ioi_dataset))
    # print(c["blocks.2.attn.hook_result"][0, ioi_dataset.word_idx["end"][0], 0, :5])

    # print(patching.act_cache["blocks.2.attn.hook_result"][0, ioi_dataset.word_idx["end"][0], 0, :5])

    print(c["blocks.2.hook_mlp_out"][0, ioi_dataset.word_idx["end"][0], :5])

    print(patching.act_cache["blocks.2.hook_mlp_out"][0, ioi_dataset.word_idx["end"][0], :5])


px.line(
    x=range(-1, 12),
    y=ld,
    title="Logit diff after ABC patching at end until layer L (included)",
    labels={"x": "L", "y": "Logit diff"},
).show()


px.line(
    x=range(-1, 12),
    y=pr,
    title="IO Proba after ABC patching at end until layer L (included)",
    labels={"x": "L", "y": "IO Proba "},
).show()


# %% one sentence

IDX = 50


def one_sentence_patching(z, source_act, hook):  # we patch at the "to" token
    # print(source_act.shape, z.shape)
    z[0, ioi_dataset.word_idx["end"][IDX]] = source_act[0, ioi_dataset.word_idx["end"][IDX]]
    return z


config2 = PatchingConfig(
    source_dataset=acc_dataset.text_prompts[IDX : IDX + 1],
    target_dataset=ioi_dataset.text_prompts[IDX : IDX + 1],
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=one_sentence_patching,
    layers=(0, max(layers) - 1),
)

metric2 = ExperimentMetric(
    lambda x, y: 0,
    dataset=ioi_dataset.text_prompts[IDX : IDX + 1],
    relative_metric=False,
    scalar_metric=False,
)

patching2 = EasyPatching(model, config2, metric2)

l, h = (5, 5)  # (8,10), (7,3), (7,9)]:

all_heads = [(l, h) for h in range(12) for l in range(9)]


for l, h in all_heads:
    hk_name, hk = patching2.get_hook(l, h)
    model.add_hook(hk_name, hk)  # we patch head 8.6

# hk_name, hk = patching2.get_hook(l, h)
# model.add_hook(hk_name, hk)  # we patch head 8.6
# model.add_hook(filter_act, freeze_attention_head_end_one_sentence)
show_attention_patterns(
    model,
    [(9, 9)],
    ioi_dataset[IDX : IDX + 1],
    mode="attn",
    title_suffix=" Post-patching",
)

# %%
IDX = 35

ioi_dataset.ioi_prompts[IDX]["text"] = "When Kevin, Sean and Ben went to the school, Sean gave a computer to"

model.reset_hooks()
show_attention_patterns(
    model,
    [(9, 9), (9, 6), (10, 0), (10, 7), (11, 10)],
    ioi_dataset[IDX : IDX + 1],
    mode="attn",
    title_suffix=" Pre-patching",
)


# %%
######################## flip s+1 #############################

s_p_1_flipped = ioi_dataset.gen_flipped_prompts("S+1")


def patch_s2(z, source_act, hook):  # we patch at the "to" token
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S2"]] = source_act[
        torch.arange(ioi_dataset.N), s_p_1_flipped.word_idx["S2"]
    ]
    return z


config = PatchingConfig(
    source_dataset=s_p_1_flipped.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=patch_s_plus_1,
    layers=(0, 9 - 1),
)

metric = ExperimentMetric(attention_probs, config.target_dataset, relative_metric=False, scalar_metric=False)

patching = EasyPatching(model, config, metric)
result = patching.run_patching()


for i, key in enumerate(["IO", "S", "S2"]):

    title = f'Average attention proba of Heads {str(heads_to_measure)} from token "to" to {key} after Patching Flipped S+1->ABB on S2'
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=title,
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )

    fig.write_image(f"svgs/{title}.svg")
    fig.show()

# %%
probs(model, s_p_1_flipped)


# %%
######################## Check dependance of S-IN output to position #############################

positions = ["end"]


flip_pref_dataset = dcc_dataset.gen_flipped_prompts("prefix")

flip_template_dataset = ioi_dataset.gen_flipped_prompts("template")


def patch_end(z, source_act, hook):  # we patch at the "to" token
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"]] = source_act[
        torch.arange(ioi_dataset.N), flip_template_dataset.word_idx["end"]
    ]
    return z


config = PatchingConfig(
    source_dataset=flip_template_dataset.text_prompts,  # abca_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=patch_end,
    layers=(0, max(layers) - 1),
)

metric = ExperimentMetric(attention_probs, config.target_dataset, relative_metric=False, scalar_metric=False)
patching = EasyPatching(model, config, metric)


# %%


patching.other_hooks = []
result = patching.run_patching()

for i, key in enumerate(["IO", "S", "S2"]):
    title = f'Average attention proba of Heads {str(heads_to_measure)} from token "to" to {key} after Patching Flip_Template-> IOI on {positions}'
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=title,
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )
    fig.write_image(f"svgs/{title} at {ctime()}.svg")
    fig.show()

# %%


all_heads = [(8, 6), (8, 10), (7, 3), (7, 9)]
model.reset_hooks()

for l, h in all_heads:
    hk_name, hk = patching.get_hook(l, h)
    model.add_hook(hk_name, hk)  # we patch head 8.6
print(f"Logit diff on IOI : {probs(model, ioi_dataset)}")


# %%


IDX = 50


def one_sentence_patching(z, source_act, hook):  # we patch at the "to" token
    # print(source_act.shape, z.shape)
    z[0, ioi_dataset.word_idx["end"][IDX]] = source_act[0, flip_template_dataset.word_idx["end"][IDX]]
    return z


config2 = PatchingConfig(
    source_dataset=flip_template_dataset.text_prompts[IDX : IDX + 1],
    target_dataset=ioi_dataset.text_prompts[IDX : IDX + 1],
    target_module="attn_head",
    head_circuit="result",
    cache_act=True,
    verbose=False,
    patch_fn=one_sentence_patching,
    layers=(0, max(layers) - 1),
)

metric2 = ExperimentMetric(
    lambda x, y: 0,
    dataset=ioi_dataset.text_prompts[IDX : IDX + 1],
    relative_metric=False,
    scalar_metric=False,
)

patching2 = EasyPatching(model, config2, metric2)

l, h = (5, 5)  # (8,10), (7,3), (7,9)]:

all_heads = [(8, 6), (8, 10), (7, 3), (7, 9)]


for l, h in all_heads:
    hk_name, hk = patching2.get_hook(l, h)
    model.add_hook(hk_name, hk)  # we patch head 8.6

# hk_name, hk = patching2.get_hook(l, h)
# model.add_hook(hk_name, hk)  # we patch head 8.6
# model.add_hook(filter_act, freeze_attention_head_end_one_sentence)
show_attention_patterns(
    model,
    [(9, 9)],
    ioi_dataset[IDX : IDX + 1],
    mode="attn",
    title_suffix=" Post-patching",
)

# %%
model.reset_hooks()
ioi_dataset.text_prompts[IDX] = "Then, Alicia and Cody had a long argument, and afterwards Cody said to Alicia"
show_attention_patterns(
    model,
    [(9, 9)],
    ioi_dataset[IDX : IDX + 1],
    mode="attn",
    title_suffix=" Pre-patching",
)

# %%
