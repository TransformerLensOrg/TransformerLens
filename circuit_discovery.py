# %%
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
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
)

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
# %%
model_name = "gpt2"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']

model = EasyTransformer.from_pretrained(model_name)
# if torch.cuda.is_available():
# model.to("cuda")
model.set_use_attn_result(True)

#%%


def get_datasets():
    """from unity"""
    batch_size = 1
    orig = "When John and Mary went to the store, John gave a bottle of milk to Mary"
    new = "When John and Mary went to the store, Charlie gave a bottle of milk to Mary"
    prompts_orig = [
        {"S": "John", "IO": "Mary", "TEMPLATE_IDX": -42, "text": orig}
    ]  # TODO make ET dataset construction not need TEMPLATE_IDX
    prompts_new = [dict(**prompts_orig[0])]
    prompts_new[0]["text"] = new
    dataset_orig = IOIDataset(
        N=batch_size, prompts=prompts_orig, prompt_type="mixed"
    )  # TODO make ET dataset construction not need prompt_type
    dataset_new = IOIDataset(N=batch_size, prompts=prompts_new, prompt_type="mixed")
    return dataset_new, dataset_orig


def logit_diff_from_logits(
    logits,
    ioi_dataset,
):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    assert len(logits.shape) == 3
    assert logits.shape[0] == len(ioi_dataset)

    IO_logits = logits[
        torch.arange(len(ioi_dataset)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.io_tokenIDs,
    ]
    S_logits = logits[
        torch.arange(len(ioi_dataset)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.s_tokenIDs,
    ]

    return IO_logits - S_logits


dataset_new, dataset_orig = get_datasets()

#%%

model = path_patching_old(
    model=model,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    senders=[(9, 8)],
    receiver_hooks=[(f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None)],
    position=dataset_orig.word_idx["end"],
)

logits_old = model(dataset_orig.toks.long()).cpu()

#%%

logit_difference = logit_diff_from_logits(logits_old, dataset_orig)
print(f"{logit_difference=}")

#%%

new_logits = path_patching(
    model=model,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    initial_senders=[(9, 8)],  # List[Tuple[int, Optional[int]]],
    receiver_to_senders={
        ("blocks.11.hook_resid_post", None): [(9, 8)],
    },
    position=dataset_orig.word_idx["end"].item(),
)

#%%

new_logit_difference = logit_diff_from_logits(new_logits, dataset_orig)
print(f"{new_logit_difference=}")

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
positions["IO"] = ioi_dataset.word_idx["IO"]
positions["S"] = ioi_dataset.word_idx["S"]
positions["S+1"] = ioi_dataset.word_idx["S+1"]
positions["S2"] = ioi_dataset.word_idx["S2"]
positions["end"] = ioi_dataset.word_idx["end"]

h = HypothesisTree(
    model,
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    threshold=0.2,
    possible_positions=positions,
    use_caching=True,
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
