#%% [markdown]
# Arthur investigation into dropout
from copy import deepcopy
import torch

from easy_transformer.experiments import get_act_hook
from utils_induction import *

assert torch.cuda.device_count() == 1
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from easy_transformer.EasyTransformer import (
    EasyTransformer,
)
from time import ctime
from functools import partial

import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import random
import einops
from IPython import get_ipython
from copy import deepcopy
from ioi_dataset import (
    IOIDataset,
)
from ioi_utils import (
    path_patching,
    max_2d,
    CLASS_COLORS,
    e,
    show_pp,
    show_attention_patterns,
    scatter_attention_and_contribution,
)
from random import randint as ri
from easy_transformer.experiments import get_act_hook
from ioi_circuit_extraction import (
    do_circuit_extraction,
    get_heads_circuit,
    CIRCUIT,
)
import random as rd
from ioi_utils import logit_diff, probs
from ioi_utils import get_top_tokens_and_probs as g

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%% [markdown]
# Make models

gpt2 = EasyTransformer.from_pretrained("gpt2").cuda()
gpt2.set_use_attn_result(True)

opt = EasyTransformer.from_pretrained("facebook/opt-125m").cuda()
opt.set_use_attn_result(True)

neo = EasyTransformer.from_pretrained("EleutherAI/gpt-neo-125M").cuda()
neo.set_use_attn_result(True)

solu = EasyTransformer.from_pretrained("solu-10l-old").cuda()
solu.set_use_attn_result(True)

model_names = ["gpt2", "opt", "neo", "solu"]
model_name = "gpt2"
model = eval(gpt2)

saved_tensors = []
#%% [markdown]
# Make induction dataset

seq_len = 10
batch_size = 5
interweave = 10  # have this many things before a repeat

rand_tokens = torch.randint(1000, 10000, (batch_size, seq_len))
rand_tokens_repeat = torch.zeros(
    size=(batch_size, seq_len * 2)
).long()  # einops.repeat(rand_tokens, "batch pos -> batch (2 pos)")

for i in range(seq_len // interweave):
    rand_tokens_repeat[
        :, i * (2 * interweave) : i * (2 * interweave) + interweave
    ] = rand_tokens[:, i * interweave : i * interweave + interweave]
    rand_tokens_repeat[
        :, i * (2 * interweave) + interweave : i * (2 * interweave) + 2 * interweave
    ] = rand_tokens[:, i * interweave : i * interweave + interweave]
rand_tokens_control = torch.randint(1000, 10000, (batch_size, seq_len * 2))

rand_tokens = prepend_padding(rand_tokens, model.tokenizer)
rand_tokens_repeat = prepend_padding(rand_tokens_repeat, model.tokenizer)
rand_tokens_control = prepend_padding(rand_tokens_control, model.tokenizer)


def calc_score(attn_pattern, hook, offset, arr):
    # Pattern has shape [batch, index, query_pos, key_pos]
    stripe = attn_pattern.diagonal(offset, dim1=-2, dim2=-1)
    scores = einops.reduce(stripe, "batch index pos -> index", "mean")
    # Store the scores in a common array
    arr[hook.layer()] = scores.detach().cpu().numpy()
    # return arr
    return attn_pattern

def filter_attn_hooks(hook_name):
    split_name = hook_name.split(".")
    return split_name[-1] == "hook_attn"

arrs = []
#%% [markdown]
# sweeeeeet plot

show_losses(
    models=[eval(model_name) for model_name in model_names],
    model_names=model_names,
    rand_tokens_repeat=rand_tokens_repeat,
    seq_len=seq_len,
    mode="logits",
)
#%% [markdown]
# Induction scores
# Use this to get a "shortlist" of the heads that matter most for ind

def calc_induction_score(attn_pattern, hook):
    # Pattern has shape [batch, index, query_pos, key_pos]
    induction_stripe = attn_pattern.diagonal(1 - seq_len, dim1=-2, dim2=-1)
    induction_scores = einops.reduce(
        induction_stripe, "batch index pos -> index", "mean"
    )
    # Store the scores in a common array
    induction_scores_array[hook.layer()] = induction_scores.detach().cpu().numpy()


def filter_attn_hooks(hook_name):
    split_name = hook_name.split(".")
    return split_name[-1] == "hook_attn"

model.reset_hooks()
more_hooks = []

# for head in [(11, head_idx) for head_idx in range(5)]: # nduct_heads[:5]:
    # more_hooks.append(hooks[head])

for model_name in model_names:
    model = eval(model_name)
    induction_scores_array = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    induction_logits = model.run_with_hooks(
        rand_tokens_repeat, fwd_hooks= more_hooks + [(filter_attn_hooks, calc_induction_score)], # , reset_hooks_start=False,
    )
    induction_scores_array = torch.tensor(induction_scores_array)
    fig = px.imshow(
        induction_scores_array,
        labels={"y": "Layer", "x": "Head"},
        color_continuous_scale="Blues",       
    )
    # add title
    fig.update_layout(
        title_text=f"Induction scores for {model_name}",
        title_x=0.5,
        title_font_size=20,
    )
    fig.show()
    saved_tensors.append(induction_scores_array)
#%% [markdown]
# Various experiments with hooks on things and a heatmap

def random_patching(z, act, hook):
    b = z.shape[0]
    z[torch.arange(b)] = act[torch.randperm(b)]
    return z

cache = {}
model.reset_hooks()
model.cache_some(
    cache,
    lambda x: "attn.hook_result" in x or "mlp_out" in x,
    suppress_warning=True,
)
logits, loss = model(
    rand_tokens_control, return_type="both", loss_return_per_token=True
).values()

hooks = {}
all_heads_and_mlps = [(layer, head_idx) for layer in range(model.cfg.n_layers) for head_idx in [None] + list(range(model.cfg.n_heads))]

for layer, head_idx in all_heads_and_mlps:
    hook_name = f"blocks.{layer}.attn.hook_result"
    if head_idx is None:
        hook_name = f"blocks.{layer}.hook_mlp_out"

    hooks[(layer, head_idx)] = (
        hook_name,
        get_act_hook(
            random_patching,
            alt_act=cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        ),
    )
model.reset_hooks()

#%% [markdown]
# Use this cell to get a rough grip on which heads matter the most

model.reset_hooks()
both_results = []
the_extra_hooks = None

# initial_logits, initial_loss = model(
#     rand_tokens_repeat, return_type="both", loss_return_per_token=True
# ).values()

for idx, extra_hooks in enumerate([[]]): # , [hooks[((6, 1))]], [hooks[(11, 4)]], the_extra_hooks]):
    if extra_hooks is None:
        break
    results = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads))
    mlp_results = torch.zeros(size=(model.cfg.n_layers, 1))
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    initial_loss = model(
        rand_tokens_repeat, return_type="both", loss_return_per_token=True
    )["loss"][:, -seq_len // 2 :].mean()
    print(f"Initial loss: {initial_loss.item()}")

    for source_layer in tqdm(range(model.cfg.n_layers)):
        for source_head_idx in [None] + list(range(model.cfg.n_heads)):
            model.reset_hooks()
            receiver_hooks = []
            receiver_hooks.append((f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None))

            if False:
                model = path_patching_attribution(
                    model=model,
                    tokens=rand_tokens_repeat,
                    patch_tokens=rand_tokens_control,
                    sender_heads=[(source_layer, source_head_idx)],
                    receiver_hooks=receiver_hooks,
                    start_token=seq_len + 1,
                    end_token=2 * seq_len,
                    device="cuda",
                    freeze_mlps=True,
                    return_hooks=False,
                    extra_hooks=extra_hooks,
                )
                title="Direct"

            else:
                # model.add_hook(*hooks[(6, 1)])
                model.add_hook(*hooks[(source_layer, source_head_idx)])
                title="Indirect"

            # model.reset_hooks()
            # for hook in hooks:
            #     model.add_hook(*hook)

            loss = model(
                rand_tokens_repeat, return_type="both", loss_return_per_token=True
            )["loss"][:, -seq_len // 2 :].mean()

            if (source_layer, source_head_idx) != (6, 1):
                a = hooks.pop((source_layer, source_head_idx))
                e("a")

            if source_head_idx is None:
                mlp_results[source_layer] = loss - initial_loss
            else:
                results[source_layer][source_head_idx] = loss - initial_loss

            if source_layer == model.cfg.n_layers-1 and source_head_idx == model.cfg.n_heads-1:
                fname = f"svgs/patch_and_freeze_{ctime()}_{ri(2134, 123759)}"
                fig = show_pp(
                    results.T.detach(),
                    title=f"{title} effect of removing heads on logit diff_{fname}",
                    # + ("" if idx == 0 else " (with top 3 name movers knocked out)"),
                    return_fig=True,
                    show_fig=False,
                )
                both_results.append(results.clone())
                fig.show()
                show_pp(mlp_results.detach().cpu())
                saved_tensors.append(results.clone().cpu())
                saved_tensors.append(mlp_results.clone().cpu())
#%% [markdown]
# Get top 5 induction heads

no_heads = 5
induct_heads = max_2d(induction_scores_array, no_heads)[0]
print(induct_heads)

# sort induction_heads by size in results
induct_heads = sorted(induct_heads, key=lambda x: results[x[0]][x[1]], reverse=True)

# have a look at these numbers
for layer, head in induct_heads:
    print(f"Layer: {layer}, Head: {head}, Induction score: {induction_scores_array[layer][head]}, Loss diff: {results[layer][head]}")

print(induct_heads)

#%% [markdown]
# Look at attention patterns of things

my_heads = max_2d(torch.abs(results), k=20)[0]
print(my_heads)

for LAYER, HEAD in my_heads:
    model.reset_hooks()
    hook_name = f"blocks.{LAYER}.attn.hook_attn" # 4 12 50 50
    new_cache = {}
    model.cache_some(new_cache, lambda x: hook_name in x)
    # model.add_hook(*hooks[((6, 1))])
    # model.add_hooks(hooks)
    model(rand_tokens_repeat)

    att = new_cache[hook_name]
    mean_att = att[:, HEAD].mean(dim=0)
    show_pp(mean_att, title=f"Mean attention for head {LAYER}.{HEAD}")

#%% [markdown]
# Look into compensation in both cases despite it seeming very different

cache = {}
model.reset_hooks()
model.cache_some(
    cache,
    lambda x: "attn.hook_result" in x or "mlp_out" in x,
    suppress_warning=True,
    # device=device,
)
logits, loss = model(
    rand_tokens_control, return_type="both", loss_return_per_token=True
).values()

# top_heads = [
#     (9, 9),
#     (9, 6),
#     (10, 1),
#     (7, 10),
#     (10, 0),
#     (11, 9),
#     (7, 2),
#     (6, 9),
#     # (10, 6),
#     # (10, 3),
# ]

top_heads = [
    (9, 6),
    (10, 0),
    (7, 2),
    (9, 9),
    (7, 10),
    (9, 1),
    (11, 5),
    (6, 9),
    (10, 1),
    (11, 9),
    (8, 1),
    (10, 6),
    (5, 1),
    (10, 10),
    (10, 3),
]

top_heads = [
    (6, 1),
    (8, 1),
    (6, 6),
    (8, 0),
    (8, 8),
]

# top_heads = induct_heads
top_heads = [(5, 1), (7, 2), (7, 10), (6, 9), (5, 5)]

hooks = {}

# top_heads = [
#     (layer, head_idx)
#     for layer in range(model.cfg.n_layers)
#     for head_idx in [None] + list(range(model.cfg.n_heads))
# ]

skipper = 0
# top_heads = max_2d(results, 20)[0][skipper:]


# def zero_all(z, act, hook):
#     z[:] = 0
#     return z


def random_patching(z, act, hook):
    b = z.shape[0]
    z[torch.arange(b)] = act[torch.randperm(b)]
    return z


for layer, head_idx in top_heads:
    hook_name = f"blocks.{layer}.attn.hook_result"
    if head_idx is None:
        hook_name = f"blocks.{layer}.hook_mlp_out"

    hooks[(layer, head_idx)] = (
        hook_name,
        get_act_hook(
            random_patching,
            alt_act=cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        ),
    )
model.reset_hooks()

#%% [markdown]
# Line graph

tot = len(induct_heads)

initial_loss = model(
        rand_tokens_repeat, return_type="both", loss_return_per_token=True
    )["loss"][:, -seq_len // 2 :].mean()

# induct_heads = max_2d(torch.tensor(induction_scores_array), tot)[0]
# induct_heads = [(6, 1), (8, 0), (6, 11), (8, 1), (8, 8)]

hooks = {head:hooks[head] for head in induct_heads}

def get_random_subset(l, size):
    return [l[i] for i in sorted(random.sample(range(len(l)), size))]

ys = []
ys2 = []
max_len = tot  # 20 - skipper
no_iters = 30

def loss_metric(
    model,
    rand_tokens_repeat,
    seq_len,
):
    cur_loss = model(
        rand_tokens_repeat, return_type="both", loss_return_per_token=True
    )["loss"][:, -seq_len // 2 :].mean()
    return cur_loss.item()

def logits_metric(
    model,
    rand_tokens_repeat,
    seq_len,
):
    """Double implemented from utils_induction..."""
    logits = model(rand_tokens_repeat, return_type="logits")
    # print(logits.shape) # 5 21 50257

    assert len(logits.shape) == 3, logits.shape
    batch_size, _, vocab_size = logits.shape
    seq_indices = einops.repeat(torch.arange(seq_len) + seq_len, "a -> b a", b=batch_size)
    batch_indices = einops.repeat(torch.arange(batch_size), "b -> b a", a=seq_len)
    logits_on_correct = logits[batch_indices, seq_indices, rand_tokens_repeat[:, seq_len + 1:]]

    return logits_on_correct[:, -seq_len // 2 :].mean().item()

metric = logits_metric
mode = "random subset"

for subset_size in tqdm(range(max_len+1)):
    model.reset_hooks()

    curv = 0
    curw = initial_loss.item()  # "EXPECTED" increase
    for _ in range(30):
        model.reset_hooks()

        ordered_hook_list = []
        if mode == "random subset":
            ordered_hook_list = get_random_subset(list(hooks.items()), subset_size)
        elif mode == "decreasing":
            ordered_hook_list = list(hooks.items())[skipper:skipper+subset_size]

        for hook in ordered_hook_list:
            model.add_hook(*hook[1])
            # curw += results[hook[0]].item()

        cur_metric = metric(
            model, rand_tokens_repeat, seq_len,
        )
        # print(f"Layer {layer}, head {head_idx}: {loss.mean().item()}")

        curv += cur_metric
    curv /= no_iters
    curw /= no_iters
    ys.append(curv)
    # curw = (
    #     initial_loss.item()
    #     + torch.sum(max_2d(results, 15)[1][skipper : skipper + subset_size]).item()
    # )
    curw = curv
    ys2.append(curw)

# plot the results
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(0, max_len+1)),
        y=ys,
        mode="lines+markers",
        name="Top N heads removed",
        line=dict(color="Black", width=1),
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, max_len+1)),
        y=ys2,
        mode="lines+markers",
        name="Sum of direct effects",
        line=dict(color="Red", width=1),
    )
)

# add the line from (0, ys[0]) to (tot-1, ys[tot-1])
fig.add_trace(
    go.Scatter(
        x=[0, max_len],
        y=[ys[0], ys[-1]],
        mode="lines",
        name="Expected",
        line=dict(color="Blue", width=1),
    )
)

# add x axis labels
fig.update_layout(
    xaxis_title="Number of heads removed",
    yaxis_title="Logits on correct",
    title="Effect of removing heads on correct logits (decreasing importance)",
)


#%% [markdown]

for tens in [froze_results, froze_mlp, flow_results, flow_mlp]:
    print(torch.sum(tens))

#%% [markdown]
# Induction compensation

from ioi_utils import compute_next_tok_dot_prod
import torch.nn.functional as F

IDX = 0


def zero_ablate(hook, z):
    return torch.zeros_like(z)


head_mask = torch.empty((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.bool)
head_mask[:] = False
head_mask[5, 5] = True
head_mask[6, 9] = False

attn_head_mask = head_mask


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