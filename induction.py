#%% [markdown]
# Arthur investigation into dropout
from copy import deepcopy
import torch

from easy_transformer.experiments import get_act_hook

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

model = gpt2
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

seq_len = 100
rand_tokens = torch.randint(1000, 10000, (4, seq_len))
rand_tokens_repeat = einops.repeat(rand_tokens, "batch pos -> batch (2 pos)")


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

for mode, offset in [
    ("induction", 1 - seq_len),
    ("duplicate", -seq_len),
    ("previous", -1),
]:
    arr = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    old_arr = deepcopy(arr)
    logits, loss = model.run_with_hooks(
        rand_tokens_repeat,
        fwd_hooks=[(filter_attn_hooks, partial(calc_score, offset=offset, arr=arr))],
        return_type="both",
        loss_return_per_token=True,
    ).values()
    fig = px.imshow(
        arr,
        labels={"y": "Layer", "x": "Head"},
        color_continuous_scale="Blues",
    )
    fig.update_layout(title=f"Attention pattern for {mode} mode")
    fig.show()
    arrs.append(arr)

#%% [markdown]
# Compare loss

for model_name in ["gpt2", "neo"]:
    model = eval(model_name)
    offset = 1 - seq_len
    logits, loss = model.run_with_hooks(
        rand_tokens_repeat,
        fwd_hooks=[(filter_attn_hooks, partial(calc_score, offset=offset, arr=arr))],
        return_type="both",
        loss_return_per_token=True,
    ).values()

    # verify that loss = - log p
    ps = torch.softmax(logits, dim=-1)
    neg_log_ps = -torch.log(ps)  # 4 200 50257
    losses = [[] for _ in range(4)]

    for i in range(suff.shape[0]):
        for j in range(suff.shape[1] - 101, suff.shape[1] - 1):
            losses[i].append(neg_log_ps[i][j][rand_tokens_repeat[i][j + 1]])

    losses = torch.tensor(losses)

    assert torch.allclose(
        losses[:, -100:].cpu(),
        loss[:, -100:].cpu(),
        rtol=1e-5,
        atol=1e-5,  # OK, just about works!
    )

    print(model_name, loss[:, -50:].mean().item(), loss[:, -50:].std().item())

# see lab notes, seems OK to compare

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
# induct_head = [(5, 1), (7, 2), (7, 10), (6, 9), (5, 5)]
induct_head = [
    (6, 1),
    (8, 1),
    (6, 6),
    (8, 0),
    (8, 8),
]  # max_2d(torch.tensor(arrs[0]), k=5) equiv

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
