from functools import partial
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import torch
from easy_transformer import EasyTransformer
from easy_transformer.experiments import get_act_hook
from easy_transformer.ioi_dataset import (
    IOIDataset,
)
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from collections import OrderedDict
from easy_transformer.ioi_utils import show_pp
import graphviz  # need both pip install graphviz and sudo apt-get install graphviz


def get_hook_tuple(layer, head_idx):
    if head_idx is None:
        return (f"blocks.{layer}.hook_mlp_out", None)
    else:
        return (f"blocks.{layer}.attn.hook_result", head_idx)


def patch_all(z, source_act, hook):
    z[:] = source_act  # make sure to slice! Otherwise objects get copied around
    return z


def patch_positions(z, source_act, hook, positions):
    if positions is None:  # same as patch_all
        raise NotImplementedError(
            "haven't implemented not specifying positions to patch"
        )
        # return source_act
    else:
        batch = z.shape[0]
        for pos in positions:
            z[torch.arange(batch), pos] = source_act[torch.arange(batch), pos]
        return z


def path_patching(
    model: EasyTransformer,
    orig_data,
    new_data,
    initial_senders=List[Tuple[int, Optional[int]]],
    receiver_to_senders: Dict[
        Tuple[str, Optional[int]], List[Tuple[int, Optional[int]]]
    ] = {},  # TODO support for token embeddings?
    position: int = 0,  # TODO extend this ...
    return_hooks: bool = False,
    freeze_mlps: bool = True,
    orig_cache=None,
    new_cache=None,
    prepend_bos=False,  # we did IOI with prepend_bos = False, but in general we think True is less sketchy. Currently EasyTransformer sometimes does one and sometimes does the other : (
) -> torch.Tensor:  # the logits
    """
    `initial_sender_hooks` is a list of (layer, head) tuples. These are the hooks that will be patched from the `new_data`
    `receiver_to_senders`: dict of (hook_name, idx) -> [(layer_idx, head_idx), ...], where head_idx None means MLP
    (because senders always have to be OUTPUTS of heads or MLPs. But receivers could be just one of Q or K or V)

    MLPs are by default considered as just another component and so are
    by default frozen when collecting acts on receivers.
    orig_data: string, torch.Tensor, or list of strings - any format that can be passed to the model directly
    new_data: same as orig_data
    max_layer: layers beyond max_layer are not frozen when collecting receiver activations
    positions: default None and patch at all positions, or a tensor specifying the positions at which to patch

    NOTE: This relies on a change to the cache_some() function in EasyTransformer/hook_points.py [we .clone() activations, unlike in neelnanda-io/EasyTransformer]
    """

    # caching...
    if orig_cache is None:
        # save activations from orig
        model.reset_hooks()
        orig_cache = {}
        model.cache_all(orig_cache)
        _ = model(orig_data, prepend_bos=False)
    initial_sender_hook_names = [
        get_hook_tuple(layer_idx, head_idx)[0]
        for layer_idx, head_idx in initial_senders
    ]
    if new_cache is None:
        # save activations from new for senders
        model.reset_hooks()
        new_cache = {}
        model.cache_some(new_cache, lambda x: x in initial_sender_hook_names)
        _ = model(new_data, prepend_bos=False)
    else:
        assert all(
            [x in new_cache for x in initial_sender_hook_names]
        ), f"Incomplete new_cache. Missing {set(initial_sender_hook_names) - set(new_cache.keys())}"

    # add the initial senders to the receiver_to_senders dict
    for layer_idx, head_idx in initial_senders:
        hook_name, head_idx = get_hook_tuple(layer_idx, head_idx)
        hook = get_act_hook(
            fn=partial(patch_positions, positions=position),
            alt_act=new_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
        )
        model.add_hook(hook_name, hook)

    # setup a way for model components to dynamically see activations from the same forward pass
    for hp in model.hook_points():
        assert (
            "model" not in hp.hook_dict or hp.hook_dict["model"] is model
        ), "Multiple models used as hook point references!"
        hp.ctx["model"] = model
    model.cache = {}  # note this cache is quite different from other caches...

    # for specifically editing the inputs from certain previous parts
    def input_activation_editor(z, hook):
        """ "Probably too many asserts, ignore them"""
        head_idx = hook.ctx["idx"]  # can be None
        N = z.shape[0]
        assert (
            len(receiver_to_senders[(hook_name, head_idx)]) > 0
        ), f"No senders for {hook_name, head_idx}, this shouldn't be attached!"
        for sender_layer_idx, sender_head_idx in receiver_to_senders[
            (hook_name, head_idx)
        ]:
            sender_hook_name, sender_head_idx = get_hook_tuple(
                sender_layer_idx, sender_head_idx
            )
            assert new_cache[sender_hook_name].shape == z.shape, (
                f"sender {sender_hook_name} has shape {new_cache[sender_hook_name].shape}, "
                f"but receiver {hook_name} has shape {z.shape}"
            )
            if head_idx is None:
                assert 3 == len(z.shape), f"hook {hook_name} has shape {z.shape}"
                z[torch.arange(N), position, :] += (
                    new_cache[sender_hook_name][torch.arange(N), position, :]
                    - orig_cache[sender_hook_name][torch.arange(N), position, :]
                )

            else:
                assert 4 == len(z.shape), f"z.shape = {z.shape}"
                z[torch.arange(N), position, head_idx:] += (
                    new_cache[sender_hook_name][torch.arange(N), position, head_idx, :]
                    - orig_cache[sender_hook_name][
                        torch.arange(N), position, head_idx, :
                    ]
                )
        return z

    # for saving and then overwriting outputs of attention and MLP layers
    def layer_output_hook(z, hook):
        hook_name = hook.ctx["hook_name"]
        hook.ctx["model"].cache[hook_name] = z.clone()  # hmm maybe CPU if debugging OOM
        assert (
            z.shape == orig_cache[hook_name].shape
        ), f"Shape mismatch: {z.shape} vs {orig_cache[hook_name].shape}"
        z[:] = orig_cache[hook_name]
        return z

    for layer_idx in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            # if this is a receiver, then compute the input activations carefully
            for letter in ["q", "k", "v"]:
                hook_name = f"blocks.{layer_idx}.attn.hook_{letter}_input"
                if (hook_name, head_idx) in receiver_to_senders:
                    model.add_hook(name=hook_name, hook=input_activation_editor)
        hook_name = f"blocks.{layer_idx}.hook_mlp_out"
        if (hook_name, None) in receiver_to_senders:
            model.add_hook(name=hook_name, hook=input_activation_editor)

        # then add the hooks that save and edit outputs
        for hook_name in [
            f"blocks.{layer_idx}.attn.hook_result",
            f"blocks.{layer_idx}.hook_mlp_out",
        ]:
            model.add_hook(
                name=hook_name,
                hook=layer_output_hook,
            )
    # don't forget hook resid post
    model.add_hook(
        name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
        hook=input_activation_editor,
    )
    logits = model(orig_data.toks.long())
    model.reset_hooks()
    return logits


# def path_patching_up_to(
#     model: EasyTransformer,
#     layer: int,
#     metric,
#     dataset,
#     orig_data,
#     new_data,
#     receiver_hooks,
#     position,
#     orig_cache=None,
#     new_cache=None,
# ):
#     model.reset_hooks()
#     attn_results = np.zeros((layer, model.cfg.n_heads))
#     mlp_results = np.zeros((layer, 1))
#     for l in tqdm(range(layer)):
#         for h in range(model.cfg.n_heads):
#             model = path_patching(
#                 model,
#                 orig_data=orig_data,
#                 new_data=new_data,
#                 senders=[(l, h)],
#                 receiver_hooks=receiver_hooks,
#                 max_layer=model.cfg.n_layers,
#                 position=position,
#                 orig_cache=orig_cache,
#                 new_cache=new_cache,
#             )
#             attn_results[l, h] = metric(model, dataset)
#             model.reset_hooks()
#         # mlp
#         model = path_patching(
#             model,
#             orig_data=orig_data,
#             new_data=new_data,
#             senders=[(l, None)],
#             receiver_hooks=receiver_hooks,
#             max_layer=model.cfg.n_layers,
#             position=position,
#             orig_cache=orig_cache,
#             new_cache=new_cache,
#         )
#         mlp_results[l] = metric(model, dataset)
#         model.reset_hooks()
#     return attn_results, mlp_results


def logit_diff_io_s(model: EasyTransformer, dataset: IOIDataset):
    N = dataset.N
    io_logits = model(dataset.toks.long())[
        torch.arange(N), dataset.word_idx["end"], dataset.io_tokenIDs
    ]
    s_logits = model(dataset.toks.long())[
        torch.arange(N), dataset.word_idx["end"], dataset.s_tokenIDs
    ]
    return (io_logits - s_logits).mean().item()


class Node:
    def __init__(self, layer: int, head: int, position: str):
        self.layer = layer
        self.head = head
        assert isinstance(
            position, str
        ), f"Position must be a string, not {type(position)}"
        self.position = position
        self.children = []
        self.parents = []

    def __repr__(self):
        return f"Node({self.layer}, {self.head}, {self.position})"

    def repr_long(self):
        return f"Node({self.layer}, {self.head}, {self.position}) with children {[child.__repr__() for child in self.children]}"

    def display(self):
        if self.layer == 12:
            return "resid out"
        elif self.head is None:
            return f"mlp{self.layer}\n{self.position}"
        else:
            return f"{self.layer}.{self.head}\n{self.position}"


class HypothesisTree:
    def __init__(
        self,
        model: EasyTransformer,
        metric: Callable,
        dataset,
        orig_data,
        new_data,
        threshold: int,
        possible_positions: OrderedDict,
        use_caching: bool = True,
        direct_paths_only: bool = False,
    ):
        self.model = model
        self.possible_positions = possible_positions
        self.node_stack = OrderedDict()
        self.populate_node_stack()
        self.current_node = self.node_stack[
            next(reversed(self.node_stack))
        ]  # last element
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
        self.direct_paths_only = direct_paths_only

    def populate_node_stack(self):
        for layer in range(self.model.cfg.n_layers):
            for head in list(range(self.model.cfg.n_heads)) + [
                None
            ]:  # includes None for mlp
                for pos in self.possible_positions:
                    node = Node(layer, head, pos)
                    self.node_stack[(layer, head, pos)] = node
        layer = self.model.cfg.n_layers
        pos = next(
            reversed(self.possible_positions)
        )  # assume the last position specified is the one that we care about in the residual stream
        resid_post = Node(layer, None, pos)
        self.node_stack[
            (layer, None, pos)
        ] = resid_post  # this represents blocks.{last}.hook_resid_post

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

    def eval(
        self,
        threshold: Union[float, None] = None,
        verbose: bool = False,
        show_graphics: bool = True,
        auto_threshold: float = 0.0,
    ):
        """Process current_node, then move to next current_node"""

        if threshold is None:
            threshold = self.threshold

        _, node = self.node_stack.popitem()
        self.important_nodes.append(node)
        print("Currently evaluating", node)

        current_node_position = node.position
        for pos in self.possible_positions:
            if (
                current_node_position != pos and node.head is None
            ):  # MLPs and the end state of the residual stream only care about the last position
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
                    receiver_hooks.append(
                        (f"blocks.{node.layer}.attn.hook_q", node.head)
                    )  # similar story to above, only care about the last position

            for receiver_hook in receiver_hooks:
                if verbose:
                    print(f"Working on pos {pos}, receiver hook {receiver_hook}")
                attn_results, mlp_results = path_patching_up_to(
                    model=self.model,
                    layer=node.layer,
                    metric=self.metric,
                    dataset=self.dataset,
                    orig_data=self.orig_data,
                    new_data=self.new_data,
                    receiver_hooks=[receiver_hook],
                    position=self.possible_positions[
                        pos
                    ],  # TODO TODO TODO I think we might need to have an "in position" (pos) as well as an "out position" (node.position)
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
                    show_pp(
                        attn_results.T,
                        title=f"Attn results for {node} with receiver hook {receiver_hook}",
                        xlabel="Head",
                        ylabel="Layer",
                    )
                    show_pp(
                        mlp_results,
                        title=f"MLP results for {node} with receiver hook {receiver_hook}",
                        xlabel="Layer",
                        ylabel="",
                    )

                if auto_threshold:
                    threshold = max(
                        auto_threshold * attn_results.std(),
                        auto_threshold * mlp_results.std(),
                        0.01,
                    )
                if verbose:
                    print(f"threshold: {threshold:.3f}")
                # process result and mark nodes above threshold as important
                for layer in range(attn_results.shape[0]):
                    for head in range(attn_results.shape[1]):
                        if abs(attn_results[layer, head]) > threshold:
                            print(
                                "Found important head:",
                                (layer, head),
                                "at position",
                                pos,
                            )
                            score = attn_results[layer, head]
                            comp_type = receiver_hook[0].split("_")[
                                -1
                            ]  # q, k, v, out, post
                            self.node_stack[(layer, head, pos)].parents.append(
                                (node, score, comp_type)
                            )
                            node.children.append(
                                (self.node_stack[(layer, head, pos)], score, comp_type)
                            )
                    if abs(mlp_results[layer]) > threshold:
                        print("Found important MLP: layer", layer, "position", pos)
                        score = mlp_results[layer, 0]
                        comp_type = receiver_hook[0].split("_")[
                            -1
                        ]  # q, k, v, out, post
                        self.node_stack[(layer, None, pos)].parents.append(
                            (node, score, comp_type)
                        )
                        node.children.append(
                            (self.node_stack[(layer, None, pos)], score, comp_type)
                        )

        # update self.current_node
        while (
            len(self.node_stack) > 0
            and len(self.node_stack[next(reversed(self.node_stack))].parents) == 0
        ):
            self.node_stack.popitem()
        if len(self.node_stack) > 0:
            self.current_node = self.node_stack[next(reversed(self.node_stack))]
        else:
            self.current_node = None

    def show(self, save_file: Optional[str] = None):
        g = graphviz.Digraph(format="png")
        g.attr("node", shape="box")
        color_dict = {
            "q": "red",
            "k": "green",
            "v": "blue",
            "out": "black",
            "post": "black",
        }
        # add each layer as a subgraph with rank=same
        for layer in range(12):
            with g.subgraph() as s:
                s.attr(rank="same")
                for node in self.important_nodes:
                    if node.layer == layer:
                        s.node(node.display())

        def scale(num: float):
            return 3 * min(1, abs(num) ** 0.4)

        for node in self.important_nodes:
            for child in node.children:
                g.edge(
                    child[0].display(),
                    node.display(),
                    color=color_dict[child[2]],
                    penwidth=str(scale(child[1])),
                    arrowsize=str(scale(child[1])),
                )
        # add invisible edges to keep layers separate
        for i in range(len(self.important_nodes) - 1):
            node1 = self.important_nodes[i]
            node2 = self.important_nodes[i + 1]
            if node1.layer != node2.layer:
                g.edge(node2.display(), node1.display(), style="invis")
        return g
