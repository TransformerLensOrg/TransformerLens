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


def get_hook_tuple(layer, head_idx, comp=None, input=False):
    """Very cursed"""
    """warning, only built for 12 layer models"""
    if comp is None:
        if head_idx is None:
            if layer < 12:
                if input:
                    return (f"blocks.{layer}.hook_resid_mid", None)
                else:
                    return (f"blocks.{layer}.hook_mlp_out", None)
            else:
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
    if positions is None:  # same as patch_all
        raise NotImplementedError(
            "haven't implemented not specifying positions to patch"
        )
        # return source_act
    else:
        batch = z.shape[0]
        cur_positions = torch.tensor(positions)
        if len(cur_positions.shape) == 0:
            cur_positions = cur_positions.unsqueeze(0)
        for pos in cur_positions:
            z[torch.arange(batch), pos] = source_act[torch.arange(batch), pos]
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


def old_path_patching(
    model: EasyTransformer,
    orig_data,
    new_data,
    senders: List[Tuple],
    receiver_hooks: List[Tuple],
    max_layer: Union[int, None] = None,
    position: int = 0,
    return_hooks: bool = False,
    freeze_mlps: bool = True,
    orig_cache=None,
    new_cache=None,
    prepend_bos=False,  # we did IOI with prepend_bos = False, but in general we think True is less sketchy. Currently EasyTransformer sometimes does one and sometimes does the other : (
):
    """TODO: synchronize"""
    """MLPs are by default considered as just another component and so are
    by default frozen when collecting acts on receivers.
    orig_data: string, torch.Tensor, or list of strings - any format that can be passed to the model directly
    new_data: same as orig_data
    senders: list of tuples (layer, head) for attention heads and (layer, None) for mlps
    receiver_hooks: list of tuples (hook_name, head) for attn heads and (hook_name, None) for mlps
    max_layer: layers beyond max_layer are not frozen when collecting receiver activations
    positions: default None and patch at all positions, or a tensor specifying the positions at which to patch
    NOTE: This relies on a change to the cache_some() function in EasyTransformer/hook_points.py [we .clone() activations, unlike in neelnanda-io/EasyTransformer]
    """
    if max_layer is None:
        max_layer = model.cfg.n_layers
    assert max_layer <= model.cfg.n_layers

    if orig_cache is None:
        # save activations from orig
        orig_cache = {}
        model.reset_hooks()
        model.cache_all(orig_cache)
        _ = model(orig_data, prepend_bos=False)

    # process senders
    sender_hooks = []
    for layer, head in senders:
        if head is not None:  # (layer, head) for attention heads
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head))
        else:  # (layer, None) for mlps
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))
    sender_hook_names = [x[0] for x in sender_hooks]

    if new_cache is None:
        # save activations from new for senders
        model.reset_hooks()
        new_cache = {}
        model.cache_some(new_cache, lambda x: x in sender_hook_names)
        _ = model(new_data, prepend_bos=False)
    else:
        assert all(
            [x in new_cache for x in sender_hook_names]
        ), f"Difference between new_cache and senders: {set(sender_hook_names) - set(new_cache.keys())}"

    # set up receiver cache
    model.reset_hooks()
    receiver_hook_names = [x[0] for x in receiver_hooks]
    receiver_cache = {}
    model.cache_some(receiver_cache, lambda x: x in receiver_hook_names)

    # configure hooks for freezing activations
    for layer in range(max_layer):
        # heads
        for head in range(model.cfg.n_heads):
            # if (layer, head) in senders:
            #     continue
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)
                hook = get_act_hook(
                    patch_all,
                    alt_act=orig_cache[hook_name],
                    idx=head,
                    dim=2,
                )
                model.add_hook(hook_name, hook)
        # mlp
        if freeze_mlps:
            hook_name = f"blocks.{layer}.hook_mlp_out"
            hook = get_act_hook(
                patch_all,
                alt_act=orig_cache[hook_name],
                idx=None,
                dim=None,
            )
            model.add_hook(hook_name, hook)

    # for senders, add new hook to patching in new acts
    for hook_name, head in sender_hooks:
        # assert not torch.allclose(orig_cache[hook_name], new_cache[hook_name]), (hook_name, head)
        hook = get_act_hook(
            partial(patch_positions, positions=[position]),
            alt_act=new_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        model.add_hook(hook_name, hook)

    # forward pass on orig, where patch in new acts for senders and orig acts for the rest
    # and save activations on receivers
    _ = model(orig_data, prepend_bos=False)
    model.reset_hooks()

    # add hooks for final forward pass on orig, where we patch in hybrid acts for receivers
    hooks = []
    for hook_name, head in receiver_hooks:
        # assert not torch.allclose(orig_cache[hook_name], receiver_cache[hook_name])
        hook = get_act_hook(
            partial(patch_positions, positions=[position]),
            alt_act=receiver_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        hooks.append((hook_name, hook))

    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model


def direct_path_patching(
    model: EasyTransformer,
    orig_data,
    new_data,
    initial_receivers_to_senders: List[
        Tuple[Tuple[str, Optional[int]], Tuple[int, Optional[int], str]]
    ],  # these are the only edges where we patch from new_cache
    receivers_to_senders: Dict[
        Tuple[str, Optional[int]], List[Tuple[int, Optional[int], str]]
    ],  # TODO support for pushing back to token embeddings?
    orig_positions,
    new_positions,
    orig_cache=None,
    new_cache=None,
) -> EasyTransformer:
    """
    Generalisation of the path_patching from the paper, where we only consider direct effects, and never indirect follow through effects.

    `intial_receivers_to_sender` is a list of pairs representing the edges we patch the new_cache connection on.

    `receiver_to_senders`: dict of (hook_name, idx, pos) -> [(layer_idx, head_idx, pos), ...]
    these define all of the edges in the graph

    NOTE: This relies on several changes to Neel's library (and RR/ET main, too)

    WARNING: this implementation is fairly cursed, mostly because it is in general hard to do these sorts of things with hooks
    """

    # caching...
    if orig_cache is None:
        # save activations from orig
        model.reset_hooks()
        orig_cache = {}
        model.cache_all(orig_cache)
        _ = model(orig_data, prepend_bos=False)
        model.reset_hooks()
    initial_sender_hook_names = [
        get_hook_tuple(item[1][0], item[1][1])[0]
        for item in initial_receivers_to_senders
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
        """ "Probably too many asserts, ignore them"""
        new_z = z.clone()
        N = z.shape[0]
        hook_name = hook.ctx["hook_name"]
        assert (
            len(receivers_to_senders[(hook_name, head_idx)]) > 0
        ), f"No senders for {hook_name, head_idx}, this shouldn't be attached!"

        # print(hook_name, head_idx, receivers_to_senders)

        assert len(receivers_to_senders[(hook_name, head_idx)]) > 0, (
            receivers_to_senders,
            hook_name,
            head_idx,
        )
        for sender_layer_idx, sender_head_idx, sender_head_pos in receivers_to_senders[
            (hook_name, head_idx)
        ]:
            sender_hook_name, sender_head_idx = get_hook_tuple(
                sender_layer_idx, sender_head_idx
            )

            if (
                (hook_name, head_idx),
                (sender_layer_idx, sender_head_idx, sender_head_pos),
            ) in initial_receivers_to_senders:  # hopefully fires > once
                cache_to_use = new_cache
                positions_to_use = new_positions
            else:
                cache_to_use = hook.ctx["model"].cache
                positions_to_use = orig_positions

            # we have to do both things casewise
            if sender_head_idx is None:
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
                        sender_head_idx,
                    ]
                    - orig_cache[sender_hook_name][
                        torch.arange(N),
                        positions_to_use[sender_head_pos],
                        sender_head_idx,
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
        z[:] = orig_cache[hook_name]
        return z

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


def direct_path_patching_up_to(
    model: EasyTransformer,
    receiver_hook,  # this is a tuple of (hook_name, head_idx)
    important_nodes,
    metric: Callable[[torch.Tensor, IOIDataset], float],  # NOTE: different to before
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
    base_initial_senders = []
    base_receivers_to_senders = {}

    for receiver in important_nodes:
        hook = get_hook_tuple(receiver.layer, receiver.head, input=True)

        if hook == receiver_hook:
            assert (
                len(receiver.children) == 0
            ), "Receiver_hook should not have children; we're going to find them here"

        elif len(receiver.children) == 0:
            continue

        else:  # patch, through the paths
            for sender_child, _, comp in receiver.children:
                if comp in ["v", "k", "q"]:
                    qkv_hook = get_hook_tuple(receiver.layer, receiver.head, comp)
                    if qkv_hook not in base_receivers_to_senders:
                        base_receivers_to_senders[qkv_hook] = []
                    warnings.warn(
                        "I think we need finer grained position control: this will just be the "
                    )
                    base_receivers_to_senders[qkv_hook].append(
                        (sender_child.layer, sender_child.head, sender_child.position)
                    )

                else:
                    if hook not in base_receivers_to_senders:
                        base_receivers_to_senders[hook] = []
                    base_receivers_to_senders[hook].append(
                        (sender_child.layer, sender_child.head, sender_child.position)
                    )

    receiver_hook_layer = int(receiver_hook[0].split(".")[1])
    model.reset_hooks()
    attn_layer_shape = receiver_hook_layer + (1 if receiver_hook[1] is None else 0)
    mlp_layer_shape = receiver_hook_layer
    attn_results = torch.zeros((attn_layer_shape, model.cfg.n_heads))
    mlp_results = torch.zeros((mlp_layer_shape), 1)
    for l in tqdm(range(attn_layer_shape)):
        for h in range(model.cfg.n_heads):
            senders = deepcopy(base_initial_senders)
            senders.append((l, h))
            receivers_to_senders = deepcopy(base_receivers_to_senders)
            if receiver_hook not in receivers_to_senders:
                receivers_to_senders[receiver_hook] = []
            receivers_to_senders[receiver_hook].append((l, h, cur_position))

            model = direct_path_patching(
                model=model,
                orig_data=orig_data,
                new_data=new_data,
                initial_receivers_to_senders=[(receiver_hook, (l, h, cur_position))],
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
            senders = deepcopy(base_initial_senders)
            senders.append((l, None))
            receivers_to_senders = deepcopy(base_receivers_to_senders)
            if receiver_hook not in receivers_to_senders:
                receivers_to_senders[receiver_hook] = []
            receivers_to_senders[receiver_hook].append((l, None, cur_position))
            cur_logits = direct_path_patching(
                model=model,
                orig_data=orig_data,
                new_data=new_data,
                initial_receivers_to_senders=[(receiver_hook, (l, None, cur_position))],
                receivers_to_senders=receivers_to_senders,
                orig_positions=orig_positions,
                new_positions=new_positions,
                orig_cache=orig_cache,
                new_cache=new_cache,
            )
            mlp_results[l] = metric(cur_logits, dataset)
            model.reset_hooks()

    return attn_results.cpu().detach(), mlp_results.cpu().detach()


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
        orig_data,
        new_data,
        threshold: int,
        orig_positions: OrderedDict,
        new_positions: OrderedDict,
        dataset=None,
        use_caching: bool = True,
        direct_paths_only: bool = False,
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
                for pos in self.orig_positions:
                    node = Node(layer, head, pos)
                    self.node_stack[(layer, head, pos)] = node
        layer = self.model.cfg.n_layers
        pos = next(
            reversed(self.orig_positions)
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
        """A handler."""
        if self.direct_paths_only:
            print("1")
            self.eval_new(
                threshold=threshold,
                verbose=verbose,
                show_graphics=show_graphics,
                auto_threshold=auto_threshold,
            )
        else:
            print("2")
            self.eval_old(
                threshold=threshold,
                verbose=verbose,
                show_graphics=show_graphics,
                auto_threshold=auto_threshold,
            )

    def eval_new(
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
            if node.layer == self.model.cfg.n_layers:
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

                attn_results, mlp_results = direct_path_patching_up_to(
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
                # convert to percentage
                attn_results -= self.default_metric
                attn_results /= self.default_metric
                mlp_results -= self.default_metric
                mlp_results /= self.default_metric

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
                            comp_type = receiver_hook[0].split("_")[
                                -1
                            ]  # q, k, v, out, post
                            if (
                                comp_type == "input"
                            ):  # TODO Arthur does a hacky thing here to make printing nicer... fix
                                for letter in "qkv":
                                    if f"_{letter}_" in receiver_hook[0]:
                                        comp_type = letter
                                assert comp_type in [
                                    "q",
                                    "k",
                                    "v",
                                ], f"{comp_type}, {receiver_hook[0]}"
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
                        comp_type = receiver_hook[0].split("_")[
                            -1
                        ]  # q, k, v, out, post
                        if comp_type == "input":  # oh rip I edited things
                            for letter in "qkv":
                                if f"_{letter}_" in receiver_hook[0]:
                                    comp_type = letter
                            assert comp_type in [
                                "q",
                                "k",
                                "v",
                            ], f"{comp_type}, {receiver_hook[0]}"
                        self.node_stack[
                            (layer, None, pos)
                        ].parents.append(  # TODO fix the MLP thing with GPT-NEO
                            (node, score, comp_type)
                        )
                        node.children.append(
                            (self.node_stack[(layer, None, pos)], score, comp_type)
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

    def eval_old(
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
        for pos in self.orig_positions:
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

                (attn_results, mlp_results,) = old_path_patching_up_to(
                    model=self.model,
                    layer=node.layer,
                    metric=self.metric,
                    dataset=self.dataset,
                    orig_data=self.orig_data,
                    new_data=self.new_data,
                    receiver_hooks=[receiver_hook],
                    position=self.orig_positions[
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
                        attn_results,
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
            "mid": "black",  # new thing for MLPs
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


def old_path_patching_up_to(
    model: EasyTransformer,
    layer: int,
    metric,
    dataset,
    orig_data,
    new_data,
    receiver_hooks,
    position,
    orig_cache=None,
    new_cache=None,
):
    model.reset_hooks()
    attn_results = np.zeros((layer, model.cfg.n_heads))
    mlp_results = np.zeros((layer, 1))
    for l in tqdm(range(layer)):
        for h in range(model.cfg.n_heads):
            model = old_path_patching(
                model,
                orig_data=orig_data,
                new_data=new_data,
                senders=[(l, h)],
                receiver_hooks=receiver_hooks,
                max_layer=model.cfg.n_layers,
                position=position,
                orig_cache=orig_cache,
                new_cache=new_cache,
            )
            attn_results[l, h] = metric(model, dataset)
            model.reset_hooks()
        # mlp
        model = old_path_patching(
            model,
            orig_data=orig_data,
            new_data=new_data,
            senders=[(l, None)],
            receiver_hooks=receiver_hooks,
            max_layer=model.cfg.n_layers,
            position=position,
            orig_cache=orig_cache,
            new_cache=new_cache,
        )
        mlp_results[l] = metric(model, dataset)
        model.reset_hooks()
    return attn_results, mlp_results
