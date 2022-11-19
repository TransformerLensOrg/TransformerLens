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
## Setup
from copy import deepcopy
import torch

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

from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
)
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import random
import einops
from IPython import get_ipython
from copy import deepcopy
from easy_transformer.ioi_dataset import (
    IOIDataset,
)
from easy_transformer.ioi_utils import (
    path_patching,
    max_2d,
    CLASS_COLORS,
    show_pp,
    show_attention_patterns,
    scatter_attention_and_contribution,
)
from random import randint as ri
from easy_transformer.ioi_circuit_extraction import (
    do_circuit_extraction,
    get_heads_circuit,
    CIRCUIT,
)
from easy_transformer.ioi_utils import logit_diff, probs
from easy_transformer.ioi_utils import get_top_tokens_and_probs as g

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
)  # TODO make this a seeded dataset

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
abc_dataset = (  # TODO seeded
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
## Path patching


def plot_path_patching(
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
                D_new=abc_dataset,
                D_orig=ioi_dataset,
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=receiver_hooks,
                positions=[position],
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


plot_path_patching(
    model,
    ioi_dataset,
    receiver_hooks=[(f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None)],
    position="end",
)
#%% [markdown]
## Writing direction results

# (change the layer_no and head_no)

scatter_attention_and_contribution(
    model=model, layer_no=9, head_no=9, ioi_dataset=ioi_dataset
)
#%% [markdown]
## Copy score

# For NMs and Negative NMs


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
print(" --- Negative heads --- ")
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
## S-Inhibition patching

plot_path_patching(
    model,
    ioi_dataset,
    receiver_hooks=[
        (f"blocks.{layer_idx}.attn.hook_v", head_idx)
        for layer_idx, head_idx in circuit["s2 inhibition"]
    ],
    position="S2",
)

#%% [markdown]
## Attention probs of NMs

ys = []
average_attention = {}

for idx, dataset in enumerate([ioi_dataset, abc_dataset]):
    fig = go.Figure()
    for heads_raw in circuit["name mover"][:3]:
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
        fig.update_layout(
            title_text=f'Attention of NMs from END to various positions on {["ioi_dataset", "abc_dataset"][idx]}'
        )
    fig.show()
#%% [markdown]
## Visualize attention patterns

model.reset_hooks()
show_attention_patterns(model, [(9, 9), (9, 6), (10, 0)], ioi_dataset[:1])

#%% [markdown]
## Token and position signal results

signal_specific_datasets = (
    {}
)  # keys are (token signal, positionnal signal) -1: inverted, 0: uncorrelated, 1: same as in ioi_dataset

# if ABB is the original pattern

signal_specific_datasets[(0, 1)] = ioi_dataset.gen_flipped_prompts(
    ("IO", "RAND")
).gen_flipped_prompts(
    ("S", "RAND")
)  # random name flip S1 and S2 are flipped to the same random name #DCC
signal_specific_datasets[(0, -1)] = signal_specific_datasets[
    (0, 1)
].gen_flipped_prompts(
    ("IO", "S1")
)  # CDC


signal_specific_datasets[(-1, -1)] = ioi_dataset.gen_flipped_prompts(
    ("S2", "IO")
)  # ABA
signal_specific_datasets[(-1, 1)] = signal_specific_datasets[
    (-1, -1)
].gen_flipped_prompts(
    ("IO", "S1")
)  # BAA


signal_specific_datasets[(1, -1)] = ioi_dataset.gen_flipped_prompts(("IO", "S1"))  # BAB
signal_specific_datasets[(1, 1)] = ioi_dataset  # ABB original dataset


def patch_end(z, source_act, hook):  # we patch at the "to" token
    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"]] = source_act[
        torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"]
    ]
    return z


s_inhibition_heads = [(8, 6), (8, 10), (7, 3), (7, 9)]

logit_diff_per_signal = np.zeros((3, 2))

for k, source_dataset in signal_specific_datasets.items():

    config = PatchingConfig(
        source_dataset=source_dataset.toks.long(),
        target_dataset=ioi_dataset.toks.long(),
        target_module="attn_head",
        head_circuit="result",
        cache_act=True,
        verbose=False,
        patch_fn=patch_end,
        layers=(0, 9 - 1),
    )
    metric = ExperimentMetric(lambda x: x, ioi_dataset)  # dummy metric
    patching = EasyPatching(model, config, metric)

    model.reset_hooks()

    for l, h in s_inhibition_heads:
        hk_name, hk = patching.get_hook(
            l, h
        )  # we use the EasyPatching as a hook generator without running the experiment
        model.add_hook(hk_name, hk)

    tok_s, pos_s = k
    logit_diff_per_signal[tok_s + 1, (pos_s + 1) // 2] = logit_diff(model, ioi_dataset)


fig = px.imshow(logit_diff_per_signal)


fig.update_layout(
    yaxis=dict(
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=[
            "Token signal inverted",
            "Token signal uncorrelated",
            "Token signal original",
        ],
    ),
    xaxis=dict(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Position signal inverted", "Position signal original"],
    ),
)
fig.show()

#%% [markdown]
## Backup NM results
# After ablating several attention heads, we actually an increase in logit difference

model.reset_hooks()
default_logit_diff = logit_diff(model, ioi_dataset)
print(f"Recall that the initial logit diff is {default_logit_diff}")

top_name_movers = [(9, 9), (9, 6), (10, 0)]
exclude_heads = [(layer, head_idx) for layer in range(12) for head_idx in range(12)]
for head in top_name_movers:
    exclude_heads.remove(head)

the_extra_hooks = do_circuit_extraction(
    model=model,
    heads_to_remove=get_heads_circuit(
        ioi_dataset=ioi_dataset,
        circuit={"name mover": top_name_movers},
    ),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=abc_dataset,
    return_hooks=True,
    excluded=exclude_heads,
)
model.reset_hooks()
for hook in the_extra_hooks:
    model.add_hook(*hook)
hooked_logit_diff = logit_diff(model, ioi_dataset)
print(
    f"After knocking out the three most important MLPs, logit diff is {hooked_logit_diff}"
)
model.reset_hooks()

both_results = []
pos = "end"

for idx, extra_hooks in enumerate([[], the_extra_hooks]):
    results = torch.zeros(size=(12, 12))
    mlp_results = torch.zeros(size=(12, 1))

    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    hooked_logit_diff = logit_diff(model, ioi_dataset)
    model.reset_hooks()

    for source_layer in tqdm(range(12)):
        for source_head_idx in list(range(12)):
            model.reset_hooks()
            receiver_hooks = []
            receiver_hooks.append(("blocks.11.hook_resid_post", None))
            model = path_patching(
                model=model,
                D_new=abc_dataset,
                D_orig=ioi_dataset,
<<<<<<< HEAD
=======
                ioi_dataset=ioi_dataset,
>>>>>>> arthur/haoxing
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=receiver_hooks,
                positions=[pos],
                return_hooks=False,
                extra_hooks=extra_hooks,
            )
            cur_logit_diff = logit_diff(model, ioi_dataset)

            if source_head_idx is None:
                mlp_results[source_layer] = cur_logit_diff - hooked_logit_diff
            else:
                results[source_layer][source_head_idx] = (
                    cur_logit_diff - hooked_logit_diff
                )

            if source_layer == 11 and source_head_idx == 11:
                fname = f"svgs/patch_and_freeze_{pos}_{ctime()}_{ri(2134, 123759)}"
                fig = show_pp(
                    results.T,
                    title=f"Direct effect of removing heads on logit diff"
                    + ("" if idx == 0 else " (with top 3 name movers knocked out)"),
                    return_fig=True,
                    show_fig=False,
                )

                both_results.append(results.clone())
                fig.show()

#%% [markdown]
# Plot the two sets of results

from easy_transformer.ioi_utils import CLASS_COLORS

cc = deepcopy(CLASS_COLORS)
circuit = deepcopy(CIRCUIT)


def what_class(layer, head, circuit):
    for circuit_class in circuit:
        if (layer, head) in circuit[circuit_class]:
            return circuit_class
    return "duplicate token"


# plot the most important heads

for idx, results in enumerate(both_results):
    k = 15
    top_heads = max_2d(torch.abs(results), k=k)[  # backup results or initial results
        0
    ]  # initial results is the patch with no KOs; direct effect on logits

    exclude_heads = []
    exclude_heads = [
        (layer_idx, head)
        for layer_idx in range(12)
        for head in range(12)
        if what_class(layer_idx, head, circuit=circuit)
        not in ["name mover", "negative", "s2 inhibition"]
    ]

    fig = go.Figure()
    heights = [
        results[layer][head]
        for layer, head in top_heads
        if (layer, head) not in exclude_heads
    ]
    colors = [
        cc[what_class(layer, head_idx, circuit=circuit)]
        for layer, head_idx in top_heads
        if (layer, head_idx) not in exclude_heads
    ]

    # plot a bar chart
    fig.add_trace(
        go.Bar(
            x=[str(x) for x in top_heads if x not in exclude_heads],
            y=heights,
            orientation="v",
            marker_color=colors,
        )
    )

    # set y axis range to [-1, 1]
    fig.update_yaxes(range=[-3, 3])

    # update y axis
    fig.update_yaxes(title_text="Change in logit diffenrence after direct patching")

    # update title
    fig.update_layout(
        title="Most important heads by direct effect on logits"
        + ("" if idx == 0 else " (with top 3 name movers knocked out)")
    )
    fig.show()

#%% [markdown]
## Validation outside of IOI
# Are the tasks of looking at previous tokens, inducting, and duplicating tokens performed on the general OWT distribution, rather than just p_IOI? Investigation of identified heads on random tokens

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


for mode, offset in [
    ("induction", 1 - seq_len),
    ("duplicate", -seq_len),
    ("previous", -1),
]:
    arr = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    old_arr = deepcopy(arr)
    logits = model.run_with_hooks(
        rand_tokens_repeat,
        fwd_hooks=[(filter_attn_hooks, partial(calc_score, offset=offset, arr=arr))],
    )
    # print(torch.allclose(arr, old_arr))
    fig = px.imshow(
        arr,
        labels={"y": "Layer", "x": "Head"},
        color_continuous_scale="Blues",
    )
    fig.update_layout(title=f"Attention pattern for {mode} mode")
    fig.show()
