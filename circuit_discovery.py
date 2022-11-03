# %%
from utils_circuit_discovery import get_hook_tuple, path_patching, path_patching_up_to, logit_diff_io_s
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
    positions=[ioi_dataset.word_idx['end']])

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

# %%
class Node():
    def __init__(self,
        layer: int,
        head: int,
        position: int
    ):
        self.layer = layer
        self.head = head
        self.position = position
        #self.hook_name = get_hook_tuple(self.layer, self.head)[0]
        self.children = []

    def __repr__(self):
        return f"Node({self.layer}, {self.head}, {self.position})"

class HypothesisTree():
    def __init__(self, model: EasyTransformer, metric: Callable, dataset, orig_data, new_data, threshold: int):
        self.model = model
        self.possible_positions = [ioi_dataset.word_idx['end']] # TODO: deal with positions
        self.node_stack = OrderedDict()
        self.populate_node_stack()
        self.current_node = self.node_stack[next(reversed(self.node_stack))] # last element
        self.metric = metric
        self.dataset = dataset
        self.orig_data = orig_data
        self.new_data = new_data
        self.threshold = threshold
        self.default_metric = self.metric(model, dataset)
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

    def eval(self, threshold=None):
        """Process current_node, then move to next current_node"""

        if threshold is None:
            threshold = self.threshold

        _, node = self.node_stack.popitem()
        self.important_nodes.append(node)
        print(f"working on node ({node.layer}, {node.head})")

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
            position=node.position
        ) 

        # convert to percentage
        attn_results -= self.default_metric
        attn_results /= self.default_metric
        mlp_results -= self.default_metric
        mlp_results /= self.default_metric

        threshold = max(3 * attn_results.std(), 3 * mlp_results.std(), 0.01)
        print(f"{attn_results.mean()=}, {attn_results.std()=}")
        print(f"{mlp_results.mean()=}, {mlp_results.std()=}")
        show_pp(attn_results.T, title=f'direct effect on {node.layer}.{node.head}')
        show_pp(mlp_results, title=f'direct effect on {node.layer}.{node.head}')
        print(f"identified")
        # process result and mark nodes above threshold as important
        for layer in range(attn_results.shape[0]):
            for head in range(attn_results.shape[1]):
                if abs(attn_results[layer, head]) > threshold:
                    print(f"({layer}, {head})")
                    self.node_stack[(layer, head, node.position)].children.append(node)
            if abs(mlp_results[layer]) > threshold:
                print(f"mlp {layer}")
                self.node_stack[(layer, None, node.position)].children.append(node)

        # update self.current_node
        while len(self.node_stack) > 0 and len(self.node_stack[next(reversed(self.node_stack))].children) == 0:
            self.node_stack.popitem()
        if len(self.node_stack) > 0:
            self.current_node = self.node_stack[next(reversed(self.node_stack))]
        else:
            self.current_node = None

    def show(self):
        print("pretty picture")


h = HypothesisTree(
    model, 
    metric=logit_diff_io_s, 
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(), 
    new_data=abc_dataset.toks.long(), 
    threshold=0.2)

# %%
%%time
while h.current_node is not None:
    h.eval()
# %%
