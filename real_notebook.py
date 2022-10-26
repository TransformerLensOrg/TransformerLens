#%% [markdown]
## Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small
# <h1><b>Intro</b></h1>

# This notebook implements all experiments in our paper (which is available on arXiv).

# For background on the task, see the paper.

# Refer to the demo of the <a href="https://github.com/neelnanda-io/Easy-Transformer">Easy-Transformer</a> library here: <a href="https://github.com/neelnanda-io/Easy-Transformer/blob/main/EasyTransformer_Demo.ipynb">demo with ablation and patching</a>.
#
# Reminder of the circuit:
# <img src="https://i.imgur.com/arokEMj.png">

#%% [markdown] Setup # TODO cut extras
from copy import deepcopy
import os
import torch

if os.environ["USER"] in ["exx", "arthur"]:  # so Arthur can safely use octobox
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    path_patching_without_internal_interactions,
    max_2d,
    CLASS_COLORS,
    all_subsets,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
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
#%% [markdown] Model and Dataset (use larger N or fewer templates for no warnings about in-template ablation)
model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)
N = 100
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
)
model_logit_diff = logit_diff(model, ioi_dataset)

print(
    f"The model gets average logit difference {model_logit_diff.item()} over {N=} examples"
)
#%% [markdown] the circuit

circuit = deepcopy(CIRCUIT)

# we make the ABC dataset in order to knockout other model components
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))  # (), suppress_warnings=True)
    .gen_flipped_prompts(("S", "RAND"))  # , suppress_warnings=True)
    .gen_flipped_prompts(("S1", "RAND"), manual_word_idx=ioi_dataset.word_idx)
)

# we then add hooks to the model to knockout all the heads except the circuit
model.reset_hooks()
relevant_heads = get_heads_circuit(ioi_dataset=ioi_dataset, circuit=circuit)
# relevant_heads.pop((5, 9))
model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=get_heads_circuit(ioi_dataset=ioi_dataset, circuit=circuit),
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    mean_dataset=abc_dataset,
)

circuit_logit_diff = logit_diff(model, ioi_dataset)
print(
    f"The circuit gets average logit difference {circuit_logit_diff.item()} over {N=} examples"
)
#%% [markdown] edge patching
model.reset_hooks()
default_logit_diff = logit_diff(model, ioi_dataset)

for pos in ["S2"]:
    results = torch.zeros(size=(12, 12))
    mlp_results = torch.zeros(size=(12, 1))
    for source_layer in tqdm(range(12)):
        for source_head_idx in [None] + list(range(12)):
            model.reset_hooks()
            receiver_hooks = []
            for layer, head_idx in circuit["s2 inhibition"]:
                # receiver_hooks.append((f"blocks.{layer}.attn.hook_q", head_idx))
                # receiver_hooks.append((f"blocks.{layer}.attn.hook_v", head_idx))
                receiver_hooks.append((f"blocks.{layer}.attn.hook_k", head_idx))
            # receiver_hooks.append(
            #     (f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None)
            # )

            model = path_patching_without_internal_interactions(
                model=model,
                source_dataset=abc_dataset,
                target_dataset=ioi_dataset,
                ioi_dataset=ioi_dataset,
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=receiver_hooks,
                max_layer=12,
                positions=[pos],
                verbose=False,
                return_hooks=False,
                freeze_mlps=False,
            )
            cur_logit_diff = logit_diff(model, ioi_dataset)

            if source_head_idx is None:
                mlp_results[source_layer] = cur_logit_diff - default_logit_diff
            else:
                results[source_layer][source_head_idx] = (
                    cur_logit_diff - default_logit_diff
                )

            if source_layer == 11 and source_head_idx == 11:
                results /= default_logit_diff
                mlp_results /= default_logit_diff

                show_pp((results - results[11][11]).T)
                show_pp((mlp_results - results[11][11]).T)

                # show attention head results
                # fname = f"svgs/patch_and_freeze_{pos}_{ctime()}_{ri(2134, 123759)}"
                # fig = show_pp(
                #     results.T,
                #     title=f"{fname=} {pos=} patching NMs",
                #     return_fig=True,
                #     show_fig=False,
                # )

                # fig.write_image(f"svgs/to_duplicate_token_K_{pos}.png")

                # fig.write_image(fname + ".png")
                # fig.write_image(fname + ".svg")
                # fig.show()

                # # # show mlp results # mlps are (hopefully not anymore???) fucked
                # fname = f"svgs/patch_and_freeze_mlp_{ctime()}_{ri(2134, 123759)}"
                # fig = show_pp(
                #     mlp_results.T,
                #     title=f"{fname}",
                #     return_fig=True,
                #     show_fig=False,
                # )
                # fig.write_image(fname + ".png")
                # fig.write_image(fname + ".svg")
                # fig.show()
#%% Delete this: hacky stuff
#%% [markdown] IOI dataset with prepend_bos...

ioi_dataset_2 = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
)

#%%
for new_N in range(1, 3):
    # d = IOIDataset(prompt_type="mixed", N=new_N, tokenizer=model.tokenizer, prepend_bos=True, has_start_padding_and_start_is_end=True)
    d = [ioi_dataset, abc_dataset, ioi_dataset][new_N - 1]
    print(f"new_N={new_N}")
    for i in range(new_N):
        for key in d.word_idx.keys():
            print(
                f"key={key} {int(d.word_idx[key][i])} {d.tokenizer.decode(d.toks[i][d.word_idx[key][i]])}"
            )
print("Seem fine?")
# %%
my_ioi_dataset = IOIDataset()
#%%
def patch_positions(z, source_act, hook, positions=["end"]):
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z


def patch_all(z, source_act, hook):
    return source_act


#%% [markdown] define the patch and freeze function


def direct_patch_and_freeze(
    model,
    source_dataset,
    target_dataset,
    ioi_dataset,
    sender_heads,
    receiver_hooks,
    max_layer,
    positions=["end"],
    verbose=False,
):
    """
    Patch in the effect of `sender_heads` on `receiver_hooks` only
    (though MLPs are "ignored", so are slight confounders)
    """

    sender_hooks = []

    for layer, head_idx in sender_heads:
        if head_idx is None:
            raise NotImplementedError()

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]

    sender_cache = {}
    model.reset_hooks()
    model.cache_some(
        sender_cache, lambda x: x in sender_hook_names, suppress_warning=True
    )
    # print(f"{sender_hook_names=}")
    source_logits = model(source_dataset.toks.long())

    target_cache = {}
    model.reset_hooks()
    model.cache_all(target_cache, suppress_warning=True)
    target_logits = model(target_dataset.toks.long())

    # for all the Q, K, V things
    model.reset_hooks()
    for layer in range(max_layer):
        for head_idx in range(12):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)

                if hook_name in receiver_hook_names:
                    continue

                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                    name=hook_name,
                )
                model.add_hook(hook_name, hook)

    # we can override the hooks above for the sender heads, though
    for hook_name, head_idx in sender_hooks:
        assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name])
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=sender_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        model.add_hook(hook_name, hook)

    # measure the receiver heads' values
    receiver_cache = {}
    model.cache_some(
        receiver_cache, lambda x: x in receiver_hook_names, suppress_warning=True
    )
    receiver_logits = model(target_dataset.toks.long())

    # patch these values in
    model.reset_hooks()
    for hook_name, head_idx in receiver_hooks:
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=receiver_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        model.add_hook(hook_name, hook)
    return model


#%% [markdown] first patch-and-freeze experiments
# TODO why are there effects that come AFTER the patching?? it's before 36 mins in voko I think

dataset_names = [
    # "ioi_dataset",
    # "abca_dataset",
    # "dcc_dataset",
    # "acca_dataset",
    # "acba_dataset",
    "abc_dataset",
    # "totally_diff_dataset",
]

results = [torch.zeros(size=(12, 12)) for _ in range(len(dataset_names))]
mlp_results = [torch.zeros(size=(12, 1)) for _ in range(len(dataset_names))]

# patch all heads into the name mover input (hopefully find S2 Inhibition)

model.reset_hooks()
default_logit_diff = logit_diff(model, ioi_dataset)

for pos in ["S+1", "IO", "S2", "end", "S"]:
    print(pos)
    results = [torch.zeros(size=(12, 12)) for _ in range(len(dataset_names))]
    mlp_results = [torch.zeros(size=(12, 1)) for _ in range(len(dataset_names))]
    for source_layer in tqdm(range(12)):
        for source_head_idx in list(range(12)):
            for dataset_idx, dataset_name in enumerate(dataset_names):
                dataset = eval(dataset_name)
                model.reset_hooks()

                receiver_hooks = []

                for layer, head_idx in circuit["induction"]:
                    # receiver_hooks.append((f"blocks.{layer}.attn.hook_q", head_idx))
                    # receiver_hooks.append((f"blocks.{layer}.attn.hook_v", head_idx))
                    receiver_hooks.append((f"blocks.{layer}.attn.hook_k", head_idx))

                model = direct_patch_and_freeze(
                    model=model,
                    source_dataset=abc_dataset,
                    target_dataset=ioi_dataset,
                    ioi_dataset=ioi_dataset,
                    sender_heads=[(source_layer, source_head_idx)],
                    receiver_hooks=receiver_hooks,
                    max_layer=12,
                    positions=[pos],
                    verbose=False,
                )

                cur_logit_diff = logit_diff(model, ioi_dataset)

                if source_head_idx is None:
                    mlp_results[dataset_idx][source_layer] = (
                        cur_logit_diff - default_logit_diff
                    )
                else:
                    results[dataset_idx][source_layer][source_head_idx] = (
                        cur_logit_diff - default_logit_diff
                    )

                if source_layer == 11 and source_head_idx == 11:
                    show_pp((results[0] - results[0][11][11]).T)
