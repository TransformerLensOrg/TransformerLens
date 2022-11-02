# %%
from functools import partial
import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
import gc
import einops
from collections import OrderedDict

from easy_transformer import EasyTransformer
from easy_transformer.experiments import get_act_hook

# %%
model_name = "gpt2"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']
model = EasyTransformer.from_pretrained(model_name)
if torch.cuda.is_available():
    model.to("cuda")
model.set_use_attn_result(True)

# %%

def patch_all(z, source_act, hook):
    return source_act

def patch_positions(z, source_act, hook, positions):
    if positions is None: # same as patch_all
        raise NotImplementedError("haven't implemented not specifying positions to patch")
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
    senders: List[Tuple], 
    receiver_hooks: List[Tuple], 
    max_layer: Union[int, None] = None, 
    positions: Union[torch.Tensor, None] = None,
    return_hooks: bool = False,
    freeze_mlps: bool = True):
    """ mlps are by default considered as just another component and so are
        by default frozen when collecting acts on receivers. 
        orig_data: string, torch.Tensor, or list of strings - any format that can be passed to the model directly
        new_data: same as orig_data
        senders: list of tuples (layer, head) for attention heads and (layer, None) for mlps
        receiver_hooks: list of tuples (hook_name, head) for attn heads and (hook_name, None) for mlps
        max_layer: layers beyond max_layer are not frozen when collecting receiver activations
        positions: default None and patch at all positions, or a tensor specifying the positions at which to patch

        NOTE: This relies on a change to the cache_some() function in EasyTransformer/hook_points.py.
    """
    if max_layer is None:
        max_layer = model.cfg.n_layers
    assert max_layer <= model.cfg.n_layers

    # save activations from orig
    orig_cache = {}
    model.cache_all(orig_cache)
    _ = model(orig_data)
    model.reset_hooks()

    # process senders
    sender_hooks = []
    for layer, head in senders:
        if head is not None: # (layer, head) for attention heads
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head))
        else: # (layer, None) for mlps
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))
    sender_hook_names = [x[0] for x in sender_hooks]

    # save activations from new for senders
    new_cache = {}
    model.cache_some(new_cache, lambda x: x in sender_hook_names)
    _ = model(new_data)
    model.reset_hooks()

    # set up receiver cache
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
                'blocks.{}.attn.hook_q',
                'blocks.{}.attn.hook_k',
                'blocks.{}.attn.hook_v',
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
            hook_name = f'blocks.{layer}.hook_mlp_out'
            hook = get_act_hook(
                patch_all,
                alt_act=orig_cache[hook_name],
                idx=None,
                dim=None,
            )
            model.add_hook(hook_name, hook)
    
    # for senders, add new hook to patching in new acts
    for hook_name, head in sender_hooks:
        #assert not torch.allclose(orig_cache[hook_name], new_cache[hook_name]), (hook_name, head)
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=new_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        model.add_hook(hook_name, hook)
    
    # forward pass on orig, where patch in new acts for senders and orig acts for the rest
    # and save activations on receivers
    _ = model(orig_data)
    model.reset_hooks()

    # add hooks for final forward pass on orig, where we patch in hybrid acts for receivers
    hooks = []
    for hook_name, head in receiver_hooks:
        #assert not torch.allclose(orig_cache[hook_name], receiver_cache[hook_name])
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
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

def path_patching2(
    model: EasyTransformer, 
    orig_data, 
    new_data, 
    senders: List[Tuple], 
    receiver_hooks: List[Tuple], 
    max_layer: Union[int, None] = None, 
    positions: Union[torch.Tensor, None] = None,
    return_hooks: bool = False,
    freeze_mlps: bool = True):
    """ mlps are by default considered as just another component and so are
        by default frozen when collecting acts on receivers. 
        orig_data: string, torch.Tensor, or list of strings - any format that can be passed to the model directly
        new_data: same as orig_data
        senders: list of tuples (layer, head) for attention heads and (layer, None) for mlps
        receiver_hooks: list of tuples (hook_name, head) for attn heads and (hook_name, None) for mlps
        max_layer: layers beyond max_layer are not frozen when collecting receiver activations
        positions: default None and patch at all positions, or a tensor specifying the positions at which to patch
    """
    if max_layer is None:
        max_layer = model.cfg.n_layers
    assert max_layer <= model.cfg.n_layers

    # save activations from orig
    orig_cache = {}
    model.cache_all(orig_cache)
    _ = model(orig_data)
    model.reset_hooks()

    # process senders
    sender_hooks = []
    for layer, head in senders:
        if head is not None: # (layer, head) for attention heads
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head))
        else: # (layer, None) for mlps
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))
    sender_hook_names = [x[0] for x in sender_hooks]

    # save activations from new for senders
    new_cache = {}
    model.cache_some(new_cache, lambda x: x in sender_hook_names)
    _ = model(new_data)
    model.reset_hooks()

    # set up receiver cache
    receiver_hook_names = [x[0] for x in receiver_hooks]
    receiver_cache = {}

    # collect receiver acts for each receiver separately
    for hook_name, head in receiver_hooks:
        model.cache_some(receiver_cache, lambda x: x == hook_name)
        receiver_layer = int(hook_name.split('.')[1]) # assumes hook names are 'blocks.n.xxx'
        # configure hooks for freezing activations
        for layer in range(min(max_layer, receiver_layer)):
            # heads
            for head in range(model.cfg.n_heads):
                # if (layer, head) in senders:
                #     continue
                for hook_template in [
                    'blocks.{}.attn.hook_q',
                    'blocks.{}.attn.hook_k',
                    'blocks.{}.attn.hook_v',
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
                hook_name = f'blocks.{layer}.hook_mlp_out'
                hook = get_act_hook(
                    patch_all,
                    alt_act=orig_cache[hook_name],
                    idx=None,
                    dim=None,
                )
                model.add_hook(hook_name, hook)
    
        # for senders, add new hook to patching in new acts
        for hook_name, head in sender_hooks:
            #assert not torch.allclose(orig_cache[hook_name], new_cache[hook_name]), (hook_name, head)
            hook = get_act_hook(
                partial(patch_positions, positions=positions),
                alt_act=new_cache[hook_name],
                idx=head,
                dim=2 if head is not None else None,
            )
            model.add_hook(hook_name, hook)
    
        # forward pass on orig, where patch in new acts for senders and orig acts for the rest
        # and save activations on receivers
        _ = model(orig_data)
        model.reset_hooks() # should this be here??

    #print(receiver_cache.keys())
    # add hooks for final forward pass on orig, where we patch in hybrid acts for receivers
    hooks = []
    for hook_name, head in receiver_hooks:
        #assert not torch.allclose(orig_cache[hook_name], receiver_cache[hook_name])
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
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

# %%
orig = "When John and Mary went to the store, John gave a bottle of milk to Mary."
new = "When John and Mary went to the store, Charlie gave a bottle of milk to Mary."
#new = "A completely different gibberish sentence blalablabladfghjkoiuytrdfg"

logit = model(orig)[0,16,5335]

# model = path_patching(
#     model, 
#     orig, 
#     new, 
#     [(8, 6)],
#     [('blocks.9.attn.hook_q', 9)],
#     12,
#     positions=[torch.tensor([16])],
# )

model = path_patching(
    model, 
    orig, 
    new, 
    [(5, 5)],
    [('blocks.8.attn.hook_v', 6)],
    12,
    positions=[torch.tensor([16])],
)

# model = path_patching(
#     model, 
#     orig, 
#     new, 
#     [(9, 9)],
#     [('blocks.11.hook_resid_post', None)],
#     12,
#     positions=[torch.tensor([16])],
# )

new_logit = model(orig)[0,16,5335]
model.reset_hooks()
print(logit, new_logit)

# %%
from ioi_dataset import (
    IOIDataset,
)

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

def path_patching_up_to(
    model: EasyTransformer, 
    layer: int,
    metric, 
    dataset,
    orig_data, 
    new_data, 
    receiver_hooks,
    positions
):
    attn_results = np.zeros((layer, model.cfg.n_heads))
    mlp_results = np.zeros(layer)
    for l in range(layer):
        for h in range(model.cfg.n_heads):
            model = path_patching(
                model,
                orig_data=orig_data,
                new_data=new_data,
                senders=[(l, h)],
                receiver_hooks=receiver_hooks,
                max_layer=model.cfg.n_layers,
                positions=positions
            )
            attn_results[l, h] = metric(model, dataset)
            model.reset_hooks()
        # mlp
        model = path_patching(
            model,
            orig_data=orig_data,
            new_data=new_data,
            senders=[(l, None)],
            receiver_hooks=receiver_hooks,
            max_layer=model.cfg.n_layers,
            positions=positions
        )
        mlp_results[l] = metric(model, dataset)
        model.reset_hooks()
    return attn_results, mlp_results

def logit_diff_io_s(model: EasyTransformer, dataset: IOIDataset):
    N = dataset.N
    io_logits = model(dataset.toks.long())[torch.arange(N), dataset.word_idx['end'], dataset.io_tokenIDs]
    s_logits = model(dataset.toks.long())[torch.arange(N), dataset.word_idx['end'], dataset.s_tokenIDs]
    return (io_logits - s_logits).mean().item()

# %%
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

# %%
model.reset_hooks()
default_logit_diff = logit_diff_io_s(model, ioi_dataset)
attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(np.expand_dims(mlp_results_n, axis=0), color_continuous_scale='RdBu', color_continuous_midpoint=0)

# %%
receiver_hooks = [('blocks.9.attn.hook_q', 9)]#, ('blocks.9.attn.hook_q', 6), ('blocks.10.attn.hook_q', 0)]
attn_results, mlp_results = path_patching_up_to(
    model, 
    layer=10, 
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    receiver_hooks=receiver_hooks,
    positions=[ioi_dataset.word_idx['end']])
# %%
model.reset_hooks()
default_logit_diff = logit_diff_io_s(model, ioi_dataset)
attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(np.expand_dims(mlp_results_n, axis=0), color_continuous_scale='RdBu', color_continuous_midpoint=0)

# %%
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
    positions=ioi_dataset.word_idx['S2'])
# %%
attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(np.expand_dims(mlp_results_n, axis=0), color_continuous_scale='RdBu', color_continuous_midpoint=0)

# %%
class HypothesisTree():
    def __init__(self, model):
        self.model = model
        self.node_stack = OrderedDict() # keys are model components, values are True for important ones and False for unimportant
        self.populate_node_stack()
        self.current_node = 

    def populate_node_stack(self):
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                self.node_stack[(layer, head)] = False
            self.node_stack[(layer, None)] = False
        self.node_stack[(self.model.cfg.n_layers, None)] = True # this represents blocks.{last}.hook_resid_post


    def eval(self):
        # 
        pass

    def show(self):
        print("pretty picture")
    
# %%

base_hypothesis = HypothesisTree()
next_hypotheses = base_hypothesis.expand(1)
good_hypos = []
for next_hypo in next_hypotheses:
    score = next_hypo.eval() # path patching
    if score > threshold:
        good_hypos.append(next_hypo)
base_hypothesis = HypothesisTree(good_hypos)

# %%

# start at blocks.11.hook_resid_post
# results = path_patching_up_to(11)
# for h
threshold = 0.2

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

for layer in range(12):
    for head in range(12):
        if attn_results[layer, head] > threshold:
            receiver_hooks = [
                (f'blocks.{layer}.attn.hook_q', head),
                (f'blocks.{layer}.attn.hook_k', head),
                (f'blocks.{layer}.attn.hook_v', head)]
            new_attn_results, new_mlp_results = path_patching_up_to(
                model,
                layer=layer,
                metric=logit_diff_io_s,
                dataset=ioi_dataset,
                orig_data=ioi_dataset.toks.long(),
                new_data=abc_dataset.toks.long(),
                receiver_hooks=receiver_hooks,
                positions=[ioi_dataset.word_idx['end']]
            )

def discover_circuit(..., start_point: Tuple):
    return new_receiver_hooks

discover_circuit("blocks.11.hook_resid_post") -> [name movers]