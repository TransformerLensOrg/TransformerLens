from copy import deepcopy
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


def get_comp_type(comp):
    if comp.endswith("_input"):
        return comp[-7]
    else:
        return "other"


def get_hook_tuple(layer, head_idx, comp=None, input=False, model_layers=12):
    """Very cursed"""
    """warning, built for 12 layer models"""

    if layer == -1:
        assert head_idx is None, head_idx
        assert comp is None, comp
        return ("blocks.0.hook_resid_pre", None)

    if comp is None:
        if head_idx is None:
            if layer < model_layers:
                if input:
                    return (f"blocks.{layer}.hook_resid_mid", None)
                else:
                    return (f"blocks.{layer}.hook_mlp_out", None)
            else:
                assert layer == model_layers
                return (f"blocks.{layer-1}.hook_resid_post", None)
        else:
            return (f"blocks.{layer}.attn.hook_result", head_idx)

    else:  # I think the QKV case here is quite different because this is INPUT to a component, not output
        assert comp in ["q", "k", "v"]
        assert head_idx is not None
        return (f"blocks.{layer}.attn.hook_{comp}_input", head_idx)


def patch_all(z, source_act, hook):
    z[:] = source_act  # make sure to slice! Otherwise objects get copied around
    return z


def patch_positions(z, source_act, hook, positions):
    assert isinstance(
        positions, torch.Tensor
    ), "Dropped support for everything that isn't a tensor of shape (batchsize,)"
    assert (
        source_act.shape[0] == positions.shape[0] == z.shape[0]
    ), f"Batch size mismatch {source_act.shape} {positions.shape} {z.shape}"
    batch_size = source_act.shape[0]

    z[torch.arange(batch_size), positions] = source_act[
        torch.arange(batch_size), positions
    ]
    return z


def get_datasets():
    """from unity"""
    batch_size = 1
    orig = "When John and Mary went to the store, John gave a bottle of milk to Mary"
    new = "When Alice and Bob went to the store, Charlie gave a bottle of milk to Mary"
    prompts_orig = [
        {"S": "John", "IO": "Mary", "TEMPLATE_IDX": -42, "text": orig}
    ]  # TODO make ET dataset construction not need TEMPLATE_IDX
    prompts_new = [{"S": "Alice", "IO": "Bob", "TEMPLATE_IDX": -42, "text": new}]
    prompts_new[0]["text"] = new
    dataset_orig = IOIDataset(
        N=batch_size, prompts=prompts_orig, prompt_type="mixed"
    )  # TODO make ET dataset construction not need prompt_type
    dataset_new = IOIDataset(
        N=batch_size,
        prompts=prompts_new,
        prompt_type="mixed",
        manual_word_idx=dataset_orig.word_idx,
    )
    return dataset_new, dataset_orig


def direct_path_patching(
    model: EasyTransformer,
    orig_data,
    new_data,
    receivers_to_senders: Dict[
        Tuple[str, Optional[int]], List[Tuple[str, Optional[int], str]]
    ],  # TODO support for pushing back to token embeddings?
    orig_positions,  # tensor of shape (batch_size,)
    new_positions,
    initial_receivers_to_senders: Optional[
        List[Tuple[Tuple[str, Optional[int]], Tuple[str, Optional[int], str]]]
    ] = None,  # these are the only edges where we patch from new_cache
    orig_cache=None,
    new_cache=None,
) -> EasyTransformer:
    """
    Generalisation of the path_patching from the paper, where we only consider direct effects, and never indirect follow through effects.

    `intial_receivers_to_sender` is a list of pairs representing the edges we patch the new_cache connection on.

    `receiver_to_senders`: dict of (hook_name, idx, pos) -> [(hook_name, head_idx, pos), ...]
    these define all of the edges in the graph

    NOTE: This relies on several changes to Neel's library (and RR/ET main, too)
    WARNING: this implementation is fairly cursed, mostly because it is in general hard to do these sorts of things with hooks
    """

    if initial_receivers_to_senders is None:
        initial_receivers_to_senders = []
        for receiver_hook, senders in receivers_to_senders.items():
            for sender_hook in senders:
                if (sender_hook[0], sender_hook[1]) not in receivers_to_senders:
                    initial_receivers_to_senders.append((receiver_hook, sender_hook))

    # caching...
    if orig_cache is None:
        # save activations from orig
        model.reset_hooks()
        orig_cache = {}
        model.cache_all(orig_cache)
        _ = model(orig_data, prepend_bos=False)
        model.reset_hooks()
    initial_sender_hook_names = [
        sender_hook[0] for _, sender_hook in initial_receivers_to_senders
    ]
    if new_cache is None:
        # save activations from new for senders
        model.reset_hooks()
        new_cache = {}
        model.cache_some(new_cache, lambda x: x in initial_sender_hook_names)
        _ = model(new_data, prepend_bos=False)
        model.reset_hooks()
    else:
        assert all(
            [x in new_cache for x in initial_sender_hook_names]
        ), f"Incomplete new_cache. Missing {set(initial_sender_hook_names) - set(new_cache.keys())}"
    model.reset_hooks()

    # setup a way for model components to dynamically see activations from the same forward pass
    for name, hp in model.hook_dict.items():
        assert (
            "model" not in hp.ctx or hp.ctx["model"] is model
        ), "Multiple models used as hook point references!"
        hp.ctx["model"] = model
        hp.ctx["hook_name"] = name
    model.cache = (
        {}
    )  # note this cache is quite different from other caches... it is populated and used on the same forward pass

    # for specifically editing the inputs from certain previous parts
    def input_activation_editor(
        z,
        hook,
        head_idx=None,
    ):
        """Probably too many asserts, ignore them"""
        new_z = z.clone()
        N = z.shape[0]
        hook_name = hook.ctx["hook_name"]
        assert (
            len(receivers_to_senders[(hook_name, head_idx)]) > 0
        ), f"No senders for {hook_name, head_idx}, this shouldn't be attached!"

        assert len(receivers_to_senders[(hook_name, head_idx)]) > 0, (
            receivers_to_senders,
            hook_name,
            head_idx,
        )
        for sender_hook_name, sender_hook_idx, sender_head_pos in receivers_to_senders[
            (hook_name, head_idx)
        ]:
            if (
                (hook_name, head_idx),
                (sender_hook_name, sender_hook_idx, sender_head_pos),
            ) in initial_receivers_to_senders:  # hopefully fires > once
                cache_to_use = new_cache
                positions_to_use = new_positions
            else:
                cache_to_use = hook.ctx["model"].cache
                positions_to_use = orig_positions

            # we have to do both things casewise
            if sender_hook_idx is None:
                sender_value = (
                    cache_to_use[sender_hook_name][
                        torch.arange(N), positions_to_use[sender_head_pos]
                    ]
                    - orig_cache[sender_hook_name][
                        torch.arange(N), positions_to_use[sender_head_pos]
                    ]
                )
            else:
                sender_value = (
                    cache_to_use[sender_hook_name][
                        torch.arange(N),
                        positions_to_use[sender_head_pos],
                        sender_hook_idx,
                    ]
                    - orig_cache[sender_hook_name][
                        torch.arange(N),
                        positions_to_use[sender_head_pos],
                        sender_hook_idx,
                    ]
                )

            if head_idx is None:
                assert (
                    new_z[torch.arange(N), positions_to_use[sender_head_pos], :].shape
                    == sender_value.shape
                ), f"{new_z.shape} != {sender_value.shape}"
                new_z[
                    torch.arange(N), positions_to_use[sender_head_pos]
                ] += sender_value
            else:
                assert (
                    new_z[
                        torch.arange(N), positions_to_use[sender_head_pos], head_idx
                    ].shape
                    == sender_value.shape
                ), f"{new_z[:, positions_to_use[sender_head_pos], head_idx].shape} != {sender_value.shape}, {positions_to_use[sender_head_pos].shape}"
                new_z[
                    torch.arange(N), positions_to_use[sender_head_pos], head_idx
                ] += sender_value

        return new_z

    # for saving and then overwriting outputs of attention and MLP layers
    def layer_output_hook(z, hook):
        hook_name = hook.ctx["hook_name"]
        hook.ctx["model"].cache[hook_name] = z.clone()  # hmm maybe CPU if debugging OOM
        assert (
            z.shape == orig_cache[hook_name].shape
        ), f"Shape mismatch: {z.shape} vs {orig_cache[hook_name].shape}"
        if hook_name == "blocks.0.hook_resid_pre" and not torch.allclose(
            z, orig_cache["blocks.0.hook_resid_pre"]
        ):
            a = 1
        z[:] = orig_cache[hook_name]
        return z

    # save the embeddings! they will be useful
    model.add_hook(name="blocks.0.hook_resid_pre", hook=layer_output_hook)

    for layer_idx in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            # if this is a receiver, then compute the input activations carefully
            for letter in ["q", "k", "v"]:
                hook_name = f"blocks.{layer_idx}.attn.hook_{letter}_input"
                if (hook_name, head_idx) in receivers_to_senders:
                    model.add_hook(
                        name=hook_name,
                        hook=partial(input_activation_editor, head_idx=head_idx),
                    )
        hook_name = f"blocks.{layer_idx}.hook_resid_mid"
        if (hook_name, None) in receivers_to_senders:
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

    # don't forget hook resid post (if missed, it would just be overwritten, which is pointless)
    model.add_hook(
        name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
        hook=input_activation_editor,
    )
    return model


def make_base_receiver_sender_objects(
    important_nodes,
):
    base_initial_senders = []
    base_receivers_to_senders = {}

    for receiver in important_nodes:
        hook = get_hook_tuple(receiver.layer, receiver.head, input=True)

        for sender_child, _, comp in receiver.children:
            if comp in ["v", "k", "q"]:
                qkv_hook = get_hook_tuple(receiver.layer, receiver.head, comp)
                if qkv_hook not in base_receivers_to_senders:
                    base_receivers_to_senders[qkv_hook] = []
                sender_hook = get_hook_tuple(sender_child.layer, sender_child.head)
                base_receivers_to_senders[qkv_hook].append(
                    (sender_hook[0], sender_hook[1], sender_child.position)
                )

            else:
                if hook not in base_receivers_to_senders:
                    base_receivers_to_senders[hook] = []
                sender_hook = get_hook_tuple(sender_child.layer, sender_child.head)
                base_receivers_to_senders[hook].append(
                    (sender_hook[0], sender_hook[1], sender_child.position)
                )

    return base_receivers_to_senders


def direct_path_patching_up_to(
    model: EasyTransformer,
    receiver_hook,  # this is a tuple of (hook_name, head_idx)
    important_nodes,
    metric: Callable[[EasyTransformer, Any], float],
    dataset,
    orig_data,
    cur_position,
    new_data,
    orig_positions,
    new_positions,
    orig_cache=None,
    new_cache=None,
):
    """New version of path_patching_up_to
    we are going to convert important_nodes into the receivers_to_senders format"""

    # now construct the arguments (in each path patching run, we will edit these a bit)

    base_receivers_to_senders = make_base_receiver_sender_objects(
        important_nodes,
    )

    receiver_hook_layer = int(receiver_hook[0].split(".")[1])
    model.reset_hooks()
    attn_layer_shape = receiver_hook_layer + (1 if receiver_hook[1] is None else 0)
    mlp_layer_shape = receiver_hook_layer
    attn_results = torch.zeros((attn_layer_shape, model.cfg.n_heads))
    mlp_results = torch.zeros((mlp_layer_shape), 1)
    for l in tqdm(range(attn_layer_shape)):
        for h in range(model.cfg.n_heads):
            receivers_to_senders = deepcopy(base_receivers_to_senders)
            if receiver_hook not in receivers_to_senders:
                receivers_to_senders[receiver_hook] = []
            sender_hook = get_hook_tuple(l, h)
            receivers_to_senders[receiver_hook].append(
                (sender_hook[0], sender_hook[1], cur_position)
            )

            model = direct_path_patching(
                model=model,
                orig_data=orig_data,
                new_data=new_data,
                initial_receivers_to_senders=[
                    (receiver_hook, (sender_hook[0], sender_hook[1], cur_position))
                ],
                receivers_to_senders=receivers_to_senders,
                orig_positions=orig_positions,
                new_positions=new_positions,
                orig_cache=orig_cache,
                new_cache=new_cache,
            )
            attn_results[l, h] = metric(model, dataset)
            model.reset_hooks()
        # mlp
        if l < mlp_layer_shape:
            receivers_to_senders = deepcopy(base_receivers_to_senders)
            if receiver_hook not in receivers_to_senders:
                receivers_to_senders[receiver_hook] = []
            sender_hook = get_hook_tuple(l, None)
            receivers_to_senders[receiver_hook].append(
                (sender_hook[0], sender_hook[1], cur_position)
            )
            cur_logits = direct_path_patching(
                model=model,
                orig_data=orig_data,
                new_data=new_data,
                initial_receivers_to_senders=[
                    (receiver_hook, (sender_hook[0], sender_hook[1], cur_position))
                ],
                receivers_to_senders=receivers_to_senders,
                orig_positions=orig_positions,
                new_positions=new_positions,
                orig_cache=orig_cache,
                new_cache=new_cache,
            )
            mlp_results[l] = metric(cur_logits, dataset)
            model.reset_hooks()

    # finally see the patch from embeds
    receivers_to_senders = deepcopy(base_receivers_to_senders)
    sender_hook = ("blocks.0.hook_resid_pre", None)
    if receiver_hook not in receivers_to_senders:
        receivers_to_senders[receiver_hook] = []
    receivers_to_senders[receiver_hook].append(
        (sender_hook[0], sender_hook[1], cur_position)
    )
    initial_receivers_to_senders = [
        (receiver_hook, (sender_hook[0], sender_hook[1], cur_position))
    ]
    model.reset_hooks()
    cur_logits = direct_path_patching(
        model=model,
        orig_data=orig_data,
        new_data=new_data,
        initial_receivers_to_senders=initial_receivers_to_senders,
        receivers_to_senders=receivers_to_senders,
        orig_positions=orig_positions,
        new_positions=new_positions,
        orig_cache=orig_cache,
        new_cache=new_cache,
    )
    embed_results = torch.tensor(metric(cur_logits, dataset))
    model.reset_hooks()

    return (
        attn_results.cpu().detach(),
        mlp_results.cpu().detach(),
        embed_results.cpu().detach(),
    )


def logit_diff_io_s(model: EasyTransformer, dataset: IOIDataset):
    N = dataset.N
    logits = model(dataset.toks.long())
    io_logits = logits[torch.arange(N), dataset.word_idx["end"], dataset.io_tokenIDs]
    s_logits = logits[torch.arange(N), dataset.word_idx["end"], dataset.s_tokenIDs]
    return (io_logits - s_logits).mean().item()


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


class Node:
    def __init__(self, layer: int, head: int, position: str, resid_out: bool = False):
        self.layer = layer
        self.head = head
        assert isinstance(
            position, str
        ), f"Position must be a string, not {type(position)}"
        self.position = position
        self.children = []
        self.parents = []
        self.resid_out = resid_out

    def __repr__(self):
        return f"Node({self.layer}, {self.head}, {self.position})"

    def repr_long(self):
        return f"Node({self.layer}, {self.head}, {self.position}) with children {[child.__repr__() for child in self.children]}"

    def display(self):
        if self.resid_out:
            return "resid out"
        elif self.layer == -1:
            return f"Embed\n{self.position}"
        elif self.head is None:
            return f"mlp{self.layer}\n{self.position}"
        else:
            return f"{self.layer}.{self.head}\n{self.position}"


class Circuit:
    def __init__(
        self,
        model: EasyTransformer,
        metric: Callable[[EasyTransformer, Any], float],
        orig_data,
        new_data,
        threshold: int,
        orig_positions: OrderedDict,
        new_positions: OrderedDict,
        use_caching: bool = True,
        dataset=None,
    ):
        model.reset_hooks()
        self.model = model
        self.orig_positions = orig_positions
        self.new_positions = new_positions
        assert list(orig_positions.keys()) == list(
            new_positions.keys()
        ), "Number and order of keys should be the same ... for now"
        self.node_stack = OrderedDict()
        self.populate_node_stack()
        self.current_node = self.node_stack[
            next(reversed(self.node_stack))
        ]  # last element TODO make a method or something for this
        self.root_node = self.current_node
        self.metric = metric
        self.dataset = dataset
        self.orig_data = orig_data
        self.new_data = new_data
        self.threshold = threshold
        self.default_metric = self.metric(model, dataset)
        assert not torch.allclose(
            torch.tensor(self.default_metric),
            torch.zeros_like(torch.tensor(self.default_metric)),
        ), "Default metric should not be zero"
        self.orig_cache = None
        self.new_cache = None
        if use_caching:
            self.get_caches()
        self.important_nodes = []
        self.finished = False

    def populate_node_stack(self):
        for pos in self.orig_positions:
            node = Node(-1, None, pos)  # represents the embedding
            self.node_stack[(-1, None, pos)] = node

        for layer in range(self.model.cfg.n_layers):
            for head in list(range(self.model.cfg.n_heads)) + [
                None
            ]:  # includes None for mlp
                for pos in self.orig_positions:
                    node = Node(layer, head, pos)
                    self.node_stack[(layer, head, pos)] = node
        layer = self.model.cfg.n_layers
        pos = next(
            reversed(self.orig_positions)
        )  # assume the last position specified is the one that we care about in the residual stream
        resid_post = Node(layer, None, pos, resid_out=True)
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
        """DIRECT PATH PATCHING VERSION"""

        if threshold is None:
            threshold = self.threshold

        _, node = self.node_stack.popitem()
        self.important_nodes.append(node)
        print("Currently evaluating", node)

        current_node_position = node.position
        for pos in self.orig_positions:
            if (
                current_node_position != pos and node.head is None
            ):  # MLPs and the end state of the residual stream only care about the last position
                continue

            receiver_hooks = []
            if node.layer == -1:
                continue  # nothing before this
            elif node.layer == self.model.cfg.n_layers:
                receiver_hooks.append((f"blocks.{node.layer-1}.hook_resid_post", None))
            elif node.head is None:
                receiver_hooks.append((f"blocks.{node.layer}.hook_resid_mid", None))
            else:
                receiver_hooks.append(
                    (f"blocks.{node.layer}.attn.hook_v_input", node.head)
                )
                receiver_hooks.append(
                    (f"blocks.{node.layer}.attn.hook_k_input", node.head)
                )
                if pos == current_node_position:
                    receiver_hooks.append(
                        (f"blocks.{node.layer}.attn.hook_q_input", node.head)
                    )  # similar story to above, only care about the last position

            for receiver_hook in receiver_hooks:
                # if verbose:
                print(f"Working on pos {pos}, receiver hook {receiver_hook}")

                attn_results, mlp_results, embed_results = direct_path_patching_up_to(
                    model=self.model,
                    receiver_hook=receiver_hook,
                    important_nodes=self.important_nodes,
                    metric=self.metric,
                    dataset=self.dataset,
                    orig_data=self.orig_data,
                    new_data=self.new_data,
                    orig_positions=self.orig_positions,
                    new_positions=self.new_positions,
                    cur_position=pos,
                    orig_cache=self.orig_cache,
                    new_cache=self.new_cache,
                )

                self.attn_results = attn_results.clone()
                self.mlp_results = mlp_results.clone()
                self.embed_results = torch.tensor(embed_results).clone()
                # convert to percentage
                attn_results -= self.default_metric
                attn_results /= self.default_metric
                mlp_results -= self.default_metric
                mlp_results /= self.default_metric
                embed_results -= self.default_metric
                embed_results /= self.default_metric

                if show_graphics:
                    show_pp(
                        attn_results,
                        title=f"{node} {receiver_hook} {pos}",
                        xlabel="Head",
                        ylabel="Layer",
                    )
                    show_pp(
                        mlp_results,
                        title=f"MLP results for {node} with receiver hook {receiver_hook} position {pos}",
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
                for layer in range(
                    attn_results.shape[0]
                ):  # TODO seems to be able to put 9.6 in the things that affect 9.6...why
                    for head in range(attn_results.shape[1]):
                        if abs(attn_results[layer, head]) > threshold:
                            print(
                                "Found important head:",
                                (layer, head),
                                "at position",
                                pos,
                            )
                            score = attn_results[layer, head]
                            comp_type = get_comp_type(receiver_hook[0])
                            self.node_stack[(layer, head, pos)].parents.append(
                                (node, score, comp_type)
                            )
                            node.children.append(
                                (self.node_stack[(layer, head, pos)], score, comp_type)
                            )
                    if (
                        layer < mlp_results.shape[0]
                        and abs(mlp_results[layer]) > threshold
                    ):
                        print("Found important MLP: layer", layer, "position", pos)
                        score = mlp_results[layer, 0]
                        comp_type = get_comp_type(receiver_hook[0])
                        self.node_stack[
                            (layer, None, pos)
                        ].parents.append(  # TODO fix the MLP thing with GPT-NEO
                            (node, score, comp_type)
                        )
                        node.children.append(
                            (self.node_stack[(layer, None, pos)], score, comp_type)
                        )
                # deal with the embedding layer too
                if abs(embed_results) > threshold:
                    print("Found important embedding layer at position", pos)
                    score = embed_results
                    comp_type = get_comp_type(receiver_hook[0])
                    self.node_stack[
                        (-1, None, pos)
                    ].parents.append(  # TODO fix the MLP thing with GPT-NEO
                        (node, score, comp_type)
                    )
                    node.children.append(
                        (self.node_stack[(-1, None, pos)], score, comp_type)
                    )
            if current_node_position == pos:
                break

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
            "other": "black",
        }
        # add each layer as a subgraph with rank=same
        for layer in range(-1, self.model.cfg.n_layers):
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

    def get_extracted_model(self, safe: bool = True) -> EasyTransformer:
        """Return the EasyTransformer model with the extracted subgraph"""
        if safe and self.current_node is not None:
            raise RuntimeError(
                "Cannot extract model while there are still nodes to explore"
            )


def evaluate_circuit(h, dataset):
    if h.current_node is not None:
        raise NotImplementedError("Make circuit full")

    receivers_to_senders = make_base_receiver_sender_objects(h.important_nodes)

    initial_receivers_to_senders: List[
        Tuple[Tuple[str, Optional[int]], Tuple[str, Optional[int], str]]
    ] = []
    for node in h.important_nodes:
        for child, _, _2 in node.children:
            if child.layer == -1:
                initial_receivers_to_senders.append(
                    (
                        ("blocks.0.hook_resid_pre", None),
                        ("blocks.0.hook_resid_pre", None, node.position),
                    )
                )
    assert (
        len(initial_receivers_to_senders) > 0
    ), "Need at least one embedding present!!!"

    initial_receivers_to_senders = list(set(initial_receivers_to_senders))

    for pos in h.orig_positions:
        assert torch.allclose(
            h.orig_positions[pos], h.new_positions[pos]
        ), "Data must be the same for all positions"

    model = direct_path_patching(
        model=h.model,
        orig_data=h.new_data,  # NOTE these are different
        new_data=h.orig_data,
        initial_receivers_to_senders=initial_receivers_to_senders,
        receivers_to_senders=receivers_to_senders,
        orig_positions=h.orig_positions,  # tensor of shape (batch_size,)
        new_positions=h.new_positions,
        orig_cache=None,
        new_cache=None,
    )
    return h.metric(model, dataset)


def path_patching(
    model,
    D_new,
    D_orig,
    sender_heads,
    receiver_hooks,
    positions=["end"],
    return_hooks=False,
    extra_hooks=[],  # when we call reset hooks, we may want to add some extra hooks after this, add these here
    freeze_mlps=False,  # recall in IOI paper we consider these "vital model components"
    have_internal_interactions=False,
):
    """
    TODO probably scrap later - very old path patching!!!
    Patch in the effect of `sender_heads` on `receiver_hooks` only
    (though MLPs are "ignored" if `freeze_mlps` is False so are slight confounders in this case - see Appendix B of https://arxiv.org/pdf/2211.00593.pdf)
    """

    def patch_positions(z, source_act, hook, positions=["end"], verbose=False):
        for pos in positions:
            z[torch.arange(D_orig.N), D_orig.word_idx[pos]] = source_act[
                torch.arange(D_new.N), D_new.word_idx[pos]
            ]
        return z

    # process arguments
    sender_hooks = []
    for layer, head_idx in sender_heads:
        if head_idx is None:
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]

    # Forward pass A (in https://arxiv.org/pdf/2211.00593.pdf)
    sender_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_some(
        sender_cache, lambda x: x in sender_hook_names, suppress_warning=True
    )
    source_logits = model(D_new.toks.long())

    # Forward pass B
    target_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_all(target_cache, suppress_warning=True)
    target_logits = model(D_orig.toks.long())

    # Forward pass C
    # Cache the receiver hooks
    # (adding these hooks first means we save values BEFORE they are overwritten)
    receiver_cache = {}
    model.reset_hooks()
    model.cache_some(
        receiver_cache,
        lambda x: x in receiver_hook_names,
        suppress_warning=True,
        verbose=False,
    )

    # "Freeze" intermediate heads to their D_orig values
    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)

                if have_internal_interactions and hook_name in receiver_hook_names:
                    continue

                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                    name=hook_name,
                )
                model.add_hook(hook_name, hook)

        if freeze_mlps:
            hook_name = f"blocks.{layer}.hook_mlp_out"
            hook = get_act_hook(
                patch_all,
                alt_act=target_cache[hook_name],
                idx=None,
                dim=None,
                name=hook_name,
            )
            model.add_hook(hook_name, hook)

    for hook in extra_hooks:
        model.add_hook(*hook)

    # These hooks will overwrite the freezing, for the sender heads
    for hook_name, head_idx in sender_hooks:
        assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name]), (
            hook_name,
            head_idx,
        )
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=sender_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        model.add_hook(hook_name, hook)
    receiver_logits = model(D_orig.toks.long())

    # Add (or return) all the hooks needed for forward pass D
    model.reset_hooks()
    hooks = []
    for hook in extra_hooks:
        hooks.append(hook)

    for hook_name, head_idx in receiver_hooks:
        for pos in positions:
            if torch.allclose(
                receiver_cache[hook_name][torch.arange(D_orig.N), D_orig.word_idx[pos]],
                target_cache[hook_name][torch.arange(D_orig.N), D_orig.word_idx[pos]],
            ):
                warnings.warn("Torch all close for {}".format(hook_name))
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=receiver_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        hooks.append((hook_name, hook))

    model.reset_hooks()
    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model