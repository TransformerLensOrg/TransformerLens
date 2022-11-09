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
from utils_circuit_discovery import (
    path_patching, 
    logit_diff_io_s,
    HypothesisTree
)

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
orig = "When John and Mary went to the store, John gave a bottle of milk to Mary."
new = "When John and Mary went to the store, Charlie gave a bottle of milk to Mary."
#new = "A completely different gibberish sentence blalablabladfghjkoiuytrdfg"

model.reset_hooks()
logit = model(orig)[0,16,5335]

model = path_patching(
    model, 
    orig, 
    new, 
    [(5, 5)],
    [('blocks.8.attn.hook_v', 6)],
    12,
    position=torch.tensor([16]),
)

new_logit = model(orig)[0,16,5335]
model.reset_hooks()
print(logit, new_logit)

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
positions['IO'] = ioi_dataset.word_idx['IO']
positions['S'] = ioi_dataset.word_idx['S']
positions['S+1'] = ioi_dataset.word_idx['S+1']
positions['S2'] = ioi_dataset.word_idx['S2']
positions['end'] = ioi_dataset.word_idx['end']

h = HypothesisTree(
    model, 
    metric=logit_diff_io_s, 
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(), 
    new_data=abc_dataset.toks.long(), 
    threshold=0.2,
    possible_positions=positions,
    use_caching=True
)

# %%
h.eval(auto_threshold=3, verbose=True, show_graphics=True)
while h.current_node is not None:
    h.eval(auto_threshold=3, verbose=True, show_graphics=True)
    with open('ioi_small.pkl', 'wb') as f:
        pickle.dump(h, f, pickle.HIGHEST_PROTOCOL)

# %%
with open('ioi_small.pkl', 'rb') as f:
        h = pickle.load(f)
# %%
# attn_results_fast = deepcopy(h.attn_results)
# mlp_results_fast = deepcopy(h.mlp_results)
# #%% [markdown]
# # Test that Arthur didn't mess up the fast caching

# use_caching = False
# h = HypothesisTree(
#     model, 
#     metric=logit_diff_io_s, 
#     dataset=ioi_dataset, 
#     orig_data=ioi_dataset.toks.long(), 
#     new_data=abc_dataset.toks.long(), 
#     threshold=0.15,  
# )
# h.eval()
# attn_results_slow = deepcopy(h.attn_results)
# mlp_results_slow = deepcopy(h.mlp_results)

# for fast_res, slow_res in zip([attn_results_fast, mlp_results_fast], [attn_results_slow, mlp_results_slow]):
#     for layer in range(fast_res.shape[0]):
#         for head in range(fast_res.shape[1]):
#             assert torch.allclose(torch.tensor(fast_res[layer, head]), torch.tensor(slow_res[layer, head]), atol=1e-3, rtol=1e-3), f"fast_res[{layer}, {head}] = {fast_res[layer, head]}, slow_res[{layer}, {head}] = {slow_res[layer, head]}"
#     for layer in range(fast_res.shape[0]):
#         assert torch.allclose(torch.tensor(fast_res[layer]), torch.tensor(slow_res[layer])), f"fast_res[{layer}] = {fast_res[layer]}, slow_res[{layer}] = {slow_res[layer]}"

# #%%
# def randomise_indices(arr: np.ndarray, indices: np.ndarray):
#     """
#     Given an array arr, shuffle the elements that occur at the indices positions between themselves
#     """

#     # get the elements that we want to shuffle
#     elements = arr[indices]

#     # shuffle the elements
#     np.random.shuffle(elements)

#     # put the shuffled elements back into the array
#     arr[indices] = elements

#     return arr

# #%%

# a = [1, 2, 3, 4, 5]
# b = [1, 2, 3]
# a = np.array(a)
# b = np.array(b)
# print(randomise_indices(a, b))
