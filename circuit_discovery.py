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

class Node():
    def __init__(self,
        layer: int,
        head: int,
        position: str
    ):
        self.layer = layer
        self.head = head
        assert isinstance(position, str), f"Position must be a string, not {type(position)}"
        self.position = position
        self.children = []
        self.parents = []

    def __repr__(self):
        return f"Node({self.layer}, {self.head}, {self.position})"

    def repr_long(self):
        return f"Node({self.layer}, {self.head}, {self.position}) with children {[child.__repr__() for child in self.children]}"


class HypothesisTree():
    def __init__(self, model: EasyTransformer, metric: Callable, dataset, orig_data, new_data, threshold: int, possible_positions: OrderedDict, use_caching: bool = True):
        self.model = model
        self.possible_positions = possible_positions
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
        pos = next(reversed(self.possible_positions)) # assume the last position specified is the one that we care about in the residual stream
        resid_post = Node(layer, None, pos) 
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

    def eval(self, threshold: Union[float, None] = None, verbose: bool = False, show_graphics: bool = True, auto_threshold: bool = False):
        """Process current_node, then move to next current_node"""

        if threshold is None:
            threshold = self.threshold

        _, node = self.node_stack.popitem()
        self.important_nodes.append(node)
        print("Currently evaluating", node)

        current_node_position = node.position
        for pos in self.possible_positions:
            if current_node_position != pos and node.head is None: # MLPs and the end state of the residual stream only care about the last position
                continue

            receiver_hooks = []
            if node.layer == self.model.cfg.n_layers:
                receiver_hooks.append((f"blocks.{node.layer-1}.hook_resid_post", None))
            elif node.head is None:
                receiver_hooks.append((f"blocks.{node.layer}.hook_mlp_out", None))
            else:
                receiver_hooks.append((f"blocks.{node.layer}.attn.hook_v", node.head))
                receiver_hooks.append((f"blocks.{node.layer}.attn.hook_k", node.head))
                if pos == current_node_position:
                    receiver_hooks.append((f"blocks.{node.layer}.attn.hook_q", node.head)) # similar story to above, only care about the last position

            for receiver_hook in receiver_hooks:
                if verbose:
                    print(f"Working on pos {pos}, receiver hook {receiver_hook}")
                attn_results, mlp_results = path_patching_up_to(
                    model=model, 
                    layer=node.layer,
                    metric=self.metric,
                    dataset=self.dataset,
                    orig_data=self.orig_data, 
                    new_data=self.new_data, 
                    receiver_hooks=[receiver_hook],
                    position=self.possible_positions[pos], # TODO TODO TODO I think we might need to have an "in position" (pos) as well as an "out position" (node.position)
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

                if show_graphics:
                    show_pp(attn_results.T, title=f"Attn results for {node} with receiver hook {receiver_hook}", xlabel="Head", ylabel="Layer")
                    show_pp(mlp_results, title=f"MLP results for {node} with receiver hook {receiver_hook}", xlabel="Layer", ylabel="")

                if auto_threshold:
                    threshold = max(3 * attn_results.std(), 3 * mlp_results.std(), 0.01)
                # process result and mark nodes above threshold as important
                for layer in range(attn_results.shape[0]):
                    for head in range(attn_results.shape[1]):
                        if abs(attn_results[layer, head]) > threshold:
                            print("Found important head:", (layer, head), "at position", pos)
                            score = attn_results[layer, head]
                            comp_type = receiver_hook[0].split('_')[-1] # q, k, v, out, post
                            self.node_stack[(layer, head, pos)].children.append((node, score, comp_type))
                            node.parents.append((self.node_stack[(layer, head, pos)], score, comp_type))
                    if abs(mlp_results[layer]) > threshold:
                        print("Found important MLP: layer", layer, "position", pos)
                        score = mlp_results[layer, 0]
                        comp_type = receiver_hook[0].split('_')[-1] # q, k, v, out, post
                        self.node_stack[(layer, None, pos)].children.append((node, score, comp_type))
                        node.parents.append((self.node_stack[(layer, None, pos)],  score, comp_type))

        # update self.current_node
        while len(self.node_stack) > 0 and len(self.node_stack[next(reversed(self.node_stack))].children) == 0:
            self.node_stack.popitem()
        if len(self.node_stack) > 0:
            self.current_node = self.node_stack[next(reversed(self.node_stack))]
        else:
            self.current_node = None

    def show(self, save=False):
        edge_list = [] # TODO add weights of edges
        edge_color_list = []
        color_dict = {'q': 'black', 'k': 'blue', 'v': 'green', 'out': 'red', 'post': 'red'}
        current_node = h.root_node
        G = nx.DiGraph()
        def dfs(node):
            G.add_nodes_from([(node, {'layer': node.layer})])
            for child_node, child_score, child_type in node.parents:
                G.add_edges_from([(node, child_node, 
                    {'weight': round(child_score,3)})])
                #edge_list.append((node, child_node, 
                #    {'weight': round(child_score,3), 'color': color_dict[child_type]}))
                edge_color_list.append(color_dict[child_type])
                dfs(child_node)
        dfs(current_node)
        #dag = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
        #pos = nx.planar_layout(dag)
        pos = nx.multipartite_layout(G, subset_key="layer")
        # make plt figure fills screen
        fig = plt.figure(dpi=300, figsize=(24, 24))
        nx.draw(
            G,
            pos,
            node_size=8000,
            node_color='#b0a8a7',
            linewidths=2.0,
            edge_color=edge_color_list,
            width=1.5,
            arrowsize=12
        )
        # nx.draw_networkx_nodes(
        #     dag,
        #     pos,
        #     node_size=8000,
        #     node_color="#ffff8f",
        #     linewidths=2.0,
        # )
        # nx.draw_networkx_edges(
        #     dag, 
        #     pos, 
        #     edgelist=edge_list,
        #     edge_color=edge_color_list,
        #     width=1.5,
        #     arrowsize=12)

        nx.draw_networkx_labels(G, pos, font_size=14)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        if save:
            plt.savefig('ioi_circuit.png')

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
h.eval(show_graphics=False)
while h.current_node is not None:
    h.eval(verbose=True, show_graphics=False, auto_threshold=True)
    h.show(save=True)
#h.show(save=True)

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
# # %%

# if __name__ == '__main__':
#     h.eval()
#     h.show(save=True)
#     # while h.current_node is not None:
#     #     h.eval(verbose=True, auto_threshold=True)
#     #     h.show()