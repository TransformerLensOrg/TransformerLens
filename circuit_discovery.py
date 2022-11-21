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
    path_patching,
    path_patching_old,
    logit_diff_io_s,
    HypothesisTree,
    logit_diff_from_logits,
    get_datasets,
    path_patching_up_to,
    path_patching_up_to_old,
)

from easy_transformer.ioi_utils import (
    show_pp,
)

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
# %%
model_name = "EleutherAI/gpt-neo-125M"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']

model = EasyTransformer.from_pretrained(model_name)
model.set_use_attn_result(True)
model.set_use_headwise_qkv_input(True)

#%%

dataset_new, dataset_orig = get_datasets()
#%%

model = path_patching_old(
    model=model,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    senders=[(9, 9)],
    # receiver_hooks=[(f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None)],
    receiver_hooks=[(f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None)],
    position=dataset_orig.word_idx["end"],
)

logits_old = model(dataset_orig.toks.long()).cpu()
#%%

logit_difference = logit_diff_from_logits(logits_old, dataset_orig)
print(f"{logit_difference=}")

#%%

model = path_patching(
    model=model,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    initial_senders=[(9, 9)],
    receiver_to_senders={
        ("blocks.11.hook_resid_post", None): [(11, None)],
        ("blocks.11.hook_mlp_out", None): [(9, 9)],
    },
    position=dataset_orig.word_idx["end"].item(),
)

logits_new = model(dataset_orig.toks.long()).cpu()
new_logit_difference = logit_diff_from_logits(logits_new, dataset_orig)
print(f"{new_logit_difference=}")

#%%

assert torch.allclose(logit_difference.cpu(), new_logit_difference.cpu())


#%%

attn_results, mlp_results = path_patching_up_to(
    model=model,
    receiver_hook=("blocks.11.hook_resid_post", None),
    important_nodes=[],
    metric=logit_diff_from_logits,
    dataset=dataset_orig,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    position=dataset_orig.word_idx["end"].item(),
    orig_cache=None,
    new_cache=None,
)

model.reset_hooks()
logits = model(dataset_orig.toks.long())
initial_logit_difference = logit_diff_from_logits(logits, dataset_orig).cpu().detach()

#%%
attn_results -= initial_logit_difference
attn_results /= initial_logit_difference
mlp_results -= initial_logit_difference
mlp_results /= initial_logit_difference

show_pp(attn_results, title="attn_results")
#%%

positions = OrderedDict()
positions["end"] = dataset_orig.word_idx["end"].item()
h = HypothesisTree(
    model,
    metric=logit_diff_io_s,
    dataset=dataset_orig,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    threshold=0.05,
    possible_positions=positions,
    use_caching=True,
    direct_paths_only=True,
)

#%%
while True:
    h.eval()
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
    #%%
# %%
# the circuit discovery algorithm
#
# we want to write down the process of discovery the IOI circuit as an algorithm
# 1. start at the logits at the token position 'end', run path patching on each head and mlp.
# 2. pick threshold (probably in terms of percentage change in metric we care about?), identify components that have effect sizes above threshold
# 3. for comp in identified components:
## a. run path patching on all components upstream to it, with the q, k, or v part of comp as receiver
#%% [markdown]
# Main part of the automatic circuit discovery algorithm

positions = OrderedDict()
positions["IO"] = dataset_orig.word_idx["IO"]
positions["S"] = dataset_orig.word_idx["S"]
positions["S+1"] = dataset_orig.word_idx["S+1"]
positions["S2"] = dataset_orig.word_idx["S2"]
positions["end"] = dataset_orig.word_idx["end"]

h = HypothesisTree(
    model,
    metric=logit_diff_io_s,
    dataset=dataset_orig,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    threshold=0.2,
    possible_positions=positions,
    use_caching=True,
    direct_paths_only=True,
)

#%%

h.eval()

#%%

assert torch.allclose(
    torch.tensor(h.attn_results).float(),
    torch.tensor(attn_results).float(),
    atol=1e-4,
    rtol=1e-4,
)

# %%
h.eval(auto_threshold=3, verbose=True, show_graphics=True)
while h.current_node is not None:
    h.eval(auto_threshold=3, verbose=True, show_graphics=True)
    with open("ioi_small.pkl", "wb") as f:
        pickle.dump(h, f, pickle.HIGHEST_PROTOCOL)

# %%
with open("ioi_small.pkl", "rb") as f:
    h = pickle.load(f)
