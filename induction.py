#%% [markdown]
# Arthur investigation into dropout
from copy import deepcopy
import torch

from easy_transformer.experiments import get_act_hook
from induction_utils import path_patching_attribution, prepend_padding, patch_all

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
# Initialise model (use larger N or fewer templates for no warnings about in-template ablation)
gpt2 = EasyTransformer.from_pretrained("gpt2").cuda()
gpt2.set_use_attn_result(True)

neo = EasyTransformer.from_pretrained("EleutherAI/gpt-neo-125M").cuda()
neo.set_use_attn_result(True)

model = neo
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

abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)

#%% [markdown]
# Induction

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

if True:  # might hog memory
    ys = [[], []]

    for idx, model_name in enumerate(["neo", "gpt2"]):
        model = eval(model_name)
        logits, loss = model(
            rand_tokens_repeat, return_type="both", loss_return_per_token=True
        ).values()
        print(
            model_name,
            loss[:, -seq_len // 2 :].mean().item(),
            loss[:, -seq_len // 2 :].std().item(),
        )
        mean_loss = loss.mean(dim=0)
        ys[idx] = mean_loss.detach().cpu()  # .numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ys[0], name="neo"))
    fig.add_trace(go.Scatter(y=ys[1], name="gpt2"))
    fig.update_layout(title="Loss over time")

    # add a line at x = 50 saying that this should be the first guessable
    fig.add_shape(
        type="line",
        x0=seq_len,
        y0=0,
        x1=seq_len,
        y1=ys[0].max(),
        line=dict(color="Black", width=1, dash="dash"),
    )
    # add a label to this line
    fig.add_annotation(
        x=seq_len,
        y=ys[0].max(),
        text="First case of induction",
        showarrow=False,
        font=dict(size=16),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=-seq_len - 5,
    )

    fig.show()

#%% [markdown]
# Which heads are the most important for induction?

model.reset_hooks()
both_results = []
the_extra_hooks = None

initial_logits, initial_loss = model(
    rand_tokens_repeat, return_type="both", loss_return_per_token=True
).values()

for idx, extra_hooks in enumerate([[], the_extra_hooks]):
    if extra_hooks is None:
        break
    results = torch.zeros(size=(12, 12))
    mlp_results = torch.zeros(size=(12, 1))
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    initial_loss = model(
        rand_tokens_repeat, return_type="both", loss_return_per_token=True
    )["loss"][:, -seq_len // 2 :].mean()
    print(f"Initial loss: {initial_loss.item()}")

    for source_layer in tqdm(range(12)):
        for source_head_idx in [None] + list(range(12)):
            model.reset_hooks()
            receiver_hooks = []
            receiver_hooks.append(("blocks.11.hook_resid_post", None))
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
            )

            # model.reset_hooks()
            # for hook in hooks:
            #     model.add_hook(*hook)

            loss = model(
                rand_tokens_repeat, return_type="both", loss_return_per_token=True
            )["loss"][:, -seq_len // 2 :].mean()

            if source_head_idx is None:
                mlp_results[source_layer] = loss - initial_loss
            else:
                results[source_layer][source_head_idx] = loss - initial_loss

            if source_layer == 11 and source_head_idx == 11:
                fname = f"svgs/patch_and_freeze_{ctime()}_{ri(2134, 123759)}"
                fig = show_pp(
                    results.T.detach(),
                    title=f"Direct effect of removing heads on logit diff"
                    + ("" if idx == 0 else " (with top 3 name movers knocked out)"),
                    return_fig=True,
                    show_fig=False,
                )
                both_results.append(results.clone())
                fig.show()
                show_pp(mlp_results.detach().cpu())
#%% [markdown]
# look into compensation in both cases despite it seeming very different

cache = {}

model.cache_some(
    cache,
    lambda x: "attn.hook_result" in x,
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

top_heads = [(5, 1), (7, 2), (7, 10), (6, 9), (5, 5)]

hooks = {}

top_heads = 

def zero_all(z, act, hook):
    z[:] = 0
    return z


for layer, head_idx in top_heads:
    hook_name = f"blocks.{layer}.attn.hook_result"

    hooks[(layer, head_idx)] = (
        hook_name,
        get_act_hook(
            zero_all,
            alt_act=cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        ),
    )

#%%


def get_random_subset(l, size):
    return [l[i] for i in sorted(random.sample(range(len(l)), size))]


ys = []
ys2 = []
max_len = 5
no_iters = 30

for subset_size in range(max_len):
    model.reset_hooks()

    curv = 0
    curw = 0  # "EXPECTED" increase
    for _ in range(30):
        model.reset_hooks()
        # for hook in list(hooks.items())[:subset_size]:
        for hook in get_random_subset(list(hooks.items()), subset_size):
            model.add_hook(*hook[1])
            # curw += results[hook[0]].item()
        loss = model(
            rand_tokens_repeat, return_type="both", loss_return_per_token=True
        )["loss"][:, -seq_len // 2 :].mean()
        # print(f"Layer {layer}, head {head_idx}: {loss.mean().item()}")
        curv += loss.mean().item()
    curv /= no_iters
    curw /= no_iters
    ys.append(curv)
    curw += ys[0]
    ys2.append(curw)

# plot the results
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(0, max_len)),
        y=ys,
        mode="lines+markers",
        name="Top N heads removed",
        line=dict(color="Black", width=1),
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(0, max_len)),
        y=ys2,
        mode="lines+markers",
        name="Sum of direct effects",
        line=dict(color="Red", width=1),
    )
)

# add x axis labels
fig.update_layout(
    xaxis_title="Number of heads removed",
    yaxis_title="Loss",
    title="Effect of removing heads on logit difference",
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


head_mask = torch.empty((12, 12), dtype=torch.bool)
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
# %%
induct_head = [(5, 1), (7, 2), (7, 10), (6, 9), (5, 5)]
# induct_head = [
#     (6, 1),
#     (8, 1),
#     (6, 6),
#     (8, 0),
#     (8, 8),
# ]  # max_2d(torch.tensor(arrs[0]), k=5) equiv

all_means = []
for k in range(len(induct_head) + 1):
    results = []
    for _ in range(10):
        head_mask = torch.empty((12, 12), dtype=torch.bool)
        head_mask[:] = False
        rd_set = rd.sample(induct_head, k=k)
        for (l, h) in rd_set:
            head_mask[l, h] = True

        def prune_attn_heads(value, hook):
            # Value has shape [batch, pos, index, d_head]
            mask = head_mask[hook.layer()]  # just the heads at this particular value
            value[:, :, mask] = 0.0
            return value

        def zero_ablate(z, hook):
            z[:] = 0.0
            return z

        model.reset_hooks()
        for l, h in rd_set:
            # if l == 7:
            # heads = [(7, 2), (7, 10)]
            # else:
            heads = [(l, h)]
            for layer, head_idx in heads:
                hook_name = f"blocks.{layer}.attn.hook_v"
                hook = get_act_hook(
                    zero_ablate,
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                    name=hook_name,
                )
                model.add_hook(hook_name, hook)

        # model.reset_hooks()
        # model.add_hook(filter_value_hooks, prune_attn_heads)
        results.append(compute_logit_probs(rand_tokens_repeat, model))

    results = np.array(results)
    all_means.append(results.mean())

fig = px.bar(
    all_means,
    title="Loss on repeated random tokens sequences (average on 10 random set of KO heads) 5.5 excluded",
)

fig.update_layout(
    xaxis_title="Number of induction head zero-KO",
    yaxis_title="Induction loss",
)
fig.show()

# %%
