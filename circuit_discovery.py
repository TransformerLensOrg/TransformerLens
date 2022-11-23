# %%
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from time import ctime
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from easy_transformer.ioi_dataset import IOIDataset
import pickle

from easy_transformer import EasyTransformer

from easy_transformer.ioi_dataset import (
    IOIDataset,
)
from easy_transformer.utils_circuit_discovery import (
    direct_path_patching,
    logit_diff_io_s,
    HypothesisTree,
    logit_diff_from_logits,
    get_datasets,
)

from easy_transformer.ioi_utils import (
    show_pp,
)

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
model_name = "gpt2"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']

model = EasyTransformer.from_pretrained(model_name)
model.set_use_attn_result(True)
model.set_use_headwise_qkv_input(True)

#%% [markdown]
# # Load data

dataset_new, dataset_orig = get_datasets()

#%% [markdown]
# Get the initial logit difference

model.reset_hooks()
logit_diff_initial = logit_diff_io_s(model, dataset_orig)

#%% [markdown]
# Do a direct path patching run

receivers_to_senders = {
    ("blocks.11.hook_resid_post", None): [
        (9, 4, "end"),
        (9, None, "end"),
    ],  # path the edge (head 9.4 -> logits) at the END position
    ("blocks.9.hook_resid_mid", None): [
        (9, 4, "end")
    ],  # path the edge (head 9.4 -> MLP 9) at END (hook_resid_mid is the input to this MLP)
    ("blocks.9.attn.hook_v_input", 4): [
        (5, 9, "S2")
    ],  # path the edge (head 0.0 -> 9.4 value) at the END position (hook_v_input is the input to the attention layer)
}

# let's choose the (0.0 -> 9.4) edge as the edge to path the new distribution from
last_guy = list(receivers_to_senders.items())[-1]
initial_receivers_to_senders = [(last_guy[0], last_guy[1][0])]

#%% [markdown]
# Now do the direct path patching

model = direct_path_patching(
    model=model,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    initial_receivers_to_senders=initial_receivers_to_senders,
    receivers_to_senders=receivers_to_senders,
    orig_positions=dataset_orig.word_idx,
    new_positions=dataset_new.word_idx,
    orig_cache=None,
    new_cache=None,
)
ans = logit_diff_io_s(model, dataset_orig)
model.reset_hooks()
print(f"{ans=}")
print(f"{logit_diff_initial=}, {ans=}")
assert np.abs(logit_diff_initial - ans) > 1e-5, "!!!"
# should be a fairly small effect

#%%

orig_positions = OrderedDict()
new_positions = OrderedDict()

keys = ["IO", "S+1", "S", "S2", "end"]
for key in keys:
    orig_positions[key] = dataset_orig.word_idx[key]
    new_positions[key] = dataset_new.word_idx[key]

h = HypothesisTree(
    model,
    metric=logit_diff_io_s,
    dataset=dataset_orig,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    threshold=0.1,
    orig_positions=orig_positions,
    new_positions=new_positions,
    # untested...
    use_caching=True,
    direct_paths_only=False,
)

#%%
while True:
    h.eval(show_graphics=True)
    a = h.show()
    # save digraph object
    with open("hypothesis_tree.dot", "w") as f:
        f.write(a.source)
    # convert to png
    from subprocess import call

    call(
        [
            "dot",
            "-Tpng",
            "hypothesis_tree.dot",
            "-o",
            f"pngs/hypothesis_tree_{ctime()}.png",
            "-Gdpi=600",
        ]
    )
