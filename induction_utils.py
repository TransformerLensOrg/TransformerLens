from contextlib import suppress
import warnings
from functools import partial
from easy_transformer import EasyTransformer
import plotly.graph_objects as go
import numpy as np
from numpy import sin, cos, pi
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
import gc
import einops
from easy_transformer.experiments import get_act_hook
from ioi_utils import e
from ioi_dataset import IOIDataset
from ioi_circuit_extraction import do_circuit_extraction


def path_patching_attribution(
    model,
    tokens,
    patch_tokens,
    start_token,
    end_token,
    sender_heads,
    receiver_hooks,
    verbose=False,
    return_hooks=False,
    extra_hooks=[],  # when we call reset hooks, we may want to add some extra hooks after this, add these here
    freeze_mlps=False,  # recall in IOI paper we consider these "vital model components"
    have_internal_interactions=False,
    device="cuda",
):
    """
    Do path patching in order to see which heads matter the most
    for directly writing the correct answer (see loss change)

    """

    def patch_all(z, source_act, hook):
        # z[start_token:end_token] = source_act[start_token:end_token]
        z = source_act
        return z

    # see path patching in ioi utils
    sender_hooks = []

    for layer, head_idx in sender_heads:
        if head_idx is None:
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]

    sender_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_some(
        sender_cache,
        lambda x: x in sender_hook_names,
        suppress_warning=True,
        device=device,
    )
    source_logits, source_loss = model(
        patch_tokens, return_type="both", loss_return_per_token=True
    ).values()

    target_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_some(
        target_cache,
        lambda x: (
            "attn.hook_q" in x
            or "hook_mlp_out" in x
            or "attn.hook_k" in x
            or "attn.hook_v" in x
        ),
        suppress_warning=True,
        device=device,
    )
    target_logits, target_loss = model(
        tokens, return_type="both", loss_return_per_token=True
    ).values()

    # measure the receiver heads' values
    receiver_cache = {}
    model.reset_hooks()
    model.cache_some(
        receiver_cache,
        lambda x: x in receiver_hook_names,
        suppress_warning=True,
        verbose=False,
        device=device,
    )

    # for all the Q, K, V things
    for layer in range(12):
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
        # ughhh, think that this is what we want, this should override the QKV above
        model.add_hook(*hook)

    # we can override the hooks above for the sender heads, though
    for hook_name, head_idx in sender_hooks:

        hook = get_act_hook(
            patch_all,
            alt_act=sender_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        model.add_hook(hook_name, hook)
    receiver_logits, receiver_loss = model(
        tokens, return_type="both", loss_return_per_token=True
    ).values()

    # receiver_cache stuff ...
    # patch these values in
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(
            *hook
        )  # ehh probably doesn't actually matter cos end thing hooked

    hooks = []
    for hook_name, head_idx in receiver_hooks:
        # if torch.allclose(receiver_cache[hook_name], target_cache[hook_name]):
        #     assert False, (hook_name, head_idx)
        hook = get_act_hook(
            patch_all,
            alt_act=receiver_cache[hook_name].clone(),
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        hooks.append((hook_name, hook))

    for obj in ["receiver_cache", "target_cache", "sender_cache"]:
        e(obj)

    model.reset_hooks()
    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model
