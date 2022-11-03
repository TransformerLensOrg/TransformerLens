# %%
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from utils_circuit_discovery import get_hook_tuple, path_patching, path_patching_up_to, logit_diff_io_s
from copy import deepcopy
from ioi_utils import show_pp
from functools import partial
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
import gc
import einops
from collections import OrderedDict
from ioi_dataset import IOIDataset


from easy_transformer import EasyTransformer
from easy_transformer.experiments import get_act_hook

from ioi_dataset import (
    IOIDataset,
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
    prompt_type="mixed",
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

    
# %%
# Try out path patching at residual stream
receiver_hooks = [('blocks.11.hook_resid_post', None)]
attn_results, mlp_results = path_patching_up_to(
    model, 
    layer=12, 
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    receiver_hooks=receiver_hooks,
    positions=[ioi_dataset.word_idx['end']]
)

model.reset_hooks()
default_logit_diff = logit_diff_io_s(model, ioi_dataset)
attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(np.expand_dims(mlp_results_n, axis=0), color_continuous_scale='RdBu', color_continuous_midpoint=0)

# %%
# Name mover queries
#receiver_hooks = [('blocks.9.attn.hook_q', 9), ('blocks.9.attn.hook_q', 6), ('blocks.10.attn.hook_q', 0)]
receiver_hooks = [
    ('blocks.11.attn.hook_q', 10), 
    ('blocks.11.attn.hook_k', 10), 
    ('blocks.11.attn.hook_v', 10), ]
    #('blocks.10.attn.hook_q', 7)]
attn_results, mlp_results = path_patching_up_to(
    model, 
    layer=10, 
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    receiver_hooks=receiver_hooks,
    position=ioi_dataset.word_idx['end'])

model.reset_hooks()
default_logit_diff = logit_diff_io_s(model, ioi_dataset)
attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(mlp_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0)

# %%
# S-inhibition heads
receiver_hooks = [
    ('blocks.7.attn.hook_v', 3), 
    ('blocks.7.attn.hook_v', 9), 
    ('blocks.8.attn.hook_v', 6),
    ('blocks.8.attn.hook_v', 10)]
attn_results, mlp_results = path_patching_up_to(
    model, 
    layer=10, 
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    receiver_hooks=receiver_hooks,
    position=ioi_dataset.word_idx['S2'])

attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(np.expand_dims(mlp_results_n, axis=0), color_continuous_scale='RdBu', color_continuous_midpoint=0)

#%% [markdown] 
# Main part of the automatic circuit discovery algorithm

class Node():
    def __init__(self,
        layer: int,
        head: int,
        position: int
    ):
        self.layer = layer
        self.head = head
        self.position = position
        self.children = []
        self.parents = []

    def __repr__(self):
        return f"Node({self.layer}, {self.head})"

    def repr_long(self):
        return f"Node({self.layer}, {self.head}, {self.position}) with children {[child.__repr__() for child in self.children]}"

use_caching = True

class HypothesisTree():
    def __init__(self, model: EasyTransformer, metric: Callable, dataset, orig_data, new_data, threshold: int):
        self.model = model
        self.possible_positions = [ioi_dataset.word_idx['end']] # TODO: deal with positions
        self.node_stack = OrderedDict()
        self.populate_node_stack()
        self.current_node = self.node_stack[next(reversed(self.node_stack))] # last element
        self.root_node = self.current_node
        self.metric = metric
        self.dataset = dataset
        self.orig_data = orig_data
        self.new_data = new_data
        self.threshold = threshold
        self.default_metric = self.metric(model, dataset)
        self.orig_cache = None
        self.new_cache = None
        if use_caching:
            self.get_caches()
        self.important_nodes = []

    def populate_node_stack(self):
        for layer in range(self.model.cfg.n_layers):
            for head in list(range(self.model.cfg.n_heads)) + [None]: # includes None for mlp
                for pos in self.possible_positions:
                    node = Node(layer, head, pos)
                    self.node_stack[(layer, head, pos)] = node
        layer = self.model.cfg.n_layers
        pos = self.possible_positions[-1] # assume the last position specified is the one that we care about in the residual stream
        resid_post = Node(layer, None, pos) 
        #resid_post.hook_name = f'blocks.{layer-1}.hook_resid_post'
        self.node_stack[(layer, None, pos)] = resid_post # this represents blocks.{last}.hook_resid_post

    def get_caches(self):
        if "orig_cache" in self.__dict__.keys():
            warnings.warn("Caches already exist, overwriting")

        # save activations from orig
        self.orig_cache = {}
        self.model.reset_hooks()
        self.model.cache_all(self.orig_cache)
        _ = self.model(self.orig_data, prepend_bos=False)

        # save activations from new for senders
        self.new_cache = {}
        self.model.reset_hooks()
        self.model.cache_all(self.new_cache)
        _ = self.model(self.new_data, prepend_bos=False)

    def eval(self):
        """Process current_node, then move to next current_node"""

        _, node = self.node_stack.popitem()
        self.important_nodes.append(node)
        print("Currently evaluating", node)

        if node.layer == self.model.cfg.n_layers:
            receiver_hooks = [
                (f"blocks.{node.layer-1}.hook_resid_post", None)
            ]
        elif node.head is not None:
            receiver_hooks = [
                (f"blocks.{node.layer}.attn.hook_q", node.head),
                (f"blocks.{node.layer}.attn.hook_k", node.head),
                (f"blocks.{node.layer}.attn.hook_v", node.head)
            ]
        else:
            receiver_hooks = [
                (f"blocks.{node.layer}.hook_mlp_out", None)
            ]

        # do path patching on all nodes before node
        attn_results, mlp_results = path_patching_up_to(
            model=model, 
            layer=node.layer,
            metric=self.metric,
            dataset=self.dataset,
            orig_data=self.orig_data, 
            new_data=self.new_data, 
            receiver_hooks=receiver_hooks,
            position=node.position,
            orig_cache=self.orig_cache,
            new_cache=self.new_cache,
        ) 

        # convert to percentage
        attn_results -= self.default_metric
        attn_results /= self.default_metric
        mlp_results -= self.default_metric
        mlp_results /= self.default_metric
        self.attn_results = attn_results
        self.mlp_results = mlp_results

        show_pp(attn_results.T, title=f"attn results for {node}", xlabel="Head", ylabel="Layer")
        show_pp(mlp_results, title=f"mlp results for {node}", xlabel="Layer", ylabel="")

        # process result and mark nodes above threshold as important
        for layer in range(attn_results.shape[0]):
            for head in range(attn_results.shape[1]):
                if abs(attn_results[layer, head]) > self.threshold:
                    print("Found important head:", (layer, head), "at position", node.position)
                    self.node_stack[(layer, head, node.position)].children.append(node)
                    node.parents.append(self.node_stack[(layer, head, node.position)])
            if abs(mlp_results[layer]) > self.threshold:
                print("Found important MLP: layer", layer, "position", node.position)
                self.node_stack[(layer, None, node.position)].children.append(node)

        # update self.current_node
        while len(self.node_stack) > 0 and len(self.node_stack[next(reversed(self.node_stack))].children) == 0:
            self.node_stack.popitem()
        if len(self.node_stack) > 0:
            self.current_node = self.node_stack[next(reversed(self.node_stack))]
        else:
            self.current_node = None

    def show(self):
        edge_list = [] # TODO add weights of edges
        current_node = h.root_node
        def dfs(node):
            for child in node.parents:
                edge_list.append((node, child))
                dfs(child)
        dfs(current_node)
        dag = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
        # make plt figure fills screen
        fig = plt.figure(figsize=(12, 12))
        nx.draw_planar(
            dag,
            arrowsize=12,
            with_labels=True,
            node_size=8000,
            node_color="#ffff8f",
            linewidths=2.0,
            width=1.5,
            font_size=14,
        )

h = HypothesisTree(
    model, 
    metric=logit_diff_io_s, 
    dataset=ioi_dataset, 
    orig_data=ioi_dataset.toks.long(), 
    new_data=abc_dataset.toks.long(), 
    threshold=0.15,
)
h.eval()
attn_results_fast = deepcopy(h.attn_results)
mlp_results_fast = deepcopy(h.mlp_results)
#%% [markdown]
# Test that Arthur didn't mess up the fast caching

use_caching = False
h = HypothesisTree(
    model, 
    metric=logit_diff_io_s, 
    dataset=ioi_dataset, 
    orig_data=ioi_dataset.toks.long(), 
    new_data=abc_dataset.toks.long(), 
    threshold=0.15,
)
h.eval()
attn_results_slow = deepcopy(h.attn_results)
mlp_results_slow = deepcopy(h.mlp_results)

for fast_res, slow_res in zip([attn_results_fast, mlp_results_fast], [attn_results_slow, mlp_results_slow]):
    for layer in range(fast_res.shape[0]):
        for head in range(fast_res.shape[1]):
            assert torch.allclose(torch.tensor(fast_res[layer, head]), torch.tensor(slow_res[layer, head]), atol=1e-3, rtol=1e-3), f"fast_res[{layer}, {head}] = {fast_res[layer, head]}, slow_res[{layer}, {head}] = {slow_res[layer, head]}"
    for layer in range(fast_res.shape[0]):
        assert torch.allclose(torch.tensor(fast_res[layer]), torch.tensor(slow_res[layer])), f"fast_res[{layer}] = {fast_res[layer]}, slow_res[{layer}] = {slow_res[layer]}"