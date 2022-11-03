from functools import partial
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
from easy_transformer import EasyTransformer
from easy_transformer.experiments import get_act_hook
from ioi_dataset import (
    IOIDataset,
)

def get_hook_tuple(layer, head_idx):
    if head_idx is None:
        return (f"blocks.{layer}.hook_mlp_out", None)
    else:
        return (f"blocks.{layer}.attn.hook_result", head_idx)

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
    position: int = 0,
    return_hooks: bool = False,
    freeze_mlps: bool = True,
    orig_cache = None,

):
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

    model.reset_hooks()
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
            partial(patch_positions, positions=[position]),
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

def path_patching_up_to(
    model: EasyTransformer, 
    layer: int,
    metric, 
    dataset,
    orig_data, 
    new_data, 
    receiver_hooks,
    position
):
    model.reset_hooks()
    attn_results = np.zeros((layer, model.cfg.n_heads))
    mlp_results = np.zeros((layer,1))
    for l in tqdm(range(layer)):
        for h in range(model.cfg.n_heads):
            model = path_patching(
                model,
                orig_data=orig_data,
                new_data=new_data,
                senders=[(l, h)],
                receiver_hooks=receiver_hooks,
                max_layer=model.cfg.n_layers,
                position=position
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
            position=position
        )
        mlp_results[l] = metric(model, dataset)
        model.reset_hooks()
    return attn_results, mlp_results

def logit_diff_io_s(model: EasyTransformer, dataset: IOIDataset):
    N = dataset.N
    io_logits = model(dataset.toks.long())[torch.arange(N), dataset.word_idx['end'], dataset.io_tokenIDs]
    s_logits = model(dataset.toks.long())[torch.arange(N), dataset.word_idx['end'], dataset.s_tokenIDs]
    return (io_logits - s_logits).mean().item()