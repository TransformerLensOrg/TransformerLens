# %%
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from ioi_dataset import IOIDataset
import pickle

from easy_transformer import EasyTransformer

from ioi_dataset import (
    IOIDataset,
)
from utils_circuit_discovery import path_patching, logit_diff_io_s, HypothesisTree

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
# %%
model_name = "gpt2"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']

model = EasyTransformer.from_pretrained(model_name)
if torch.cuda.is_available():
    model.to("cuda")
model.set_use_attn_result(True)

# %%
N = 50
ioi_dataset = IOIDataset(
    prompt_type="ABBA",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
)
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
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
