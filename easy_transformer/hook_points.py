# Import stuff
import logging
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm

import random
import time

from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc
import collections
import copy

# import comet_ml
import itertools

# A helper class to get access to intermediate activations (inspired by Garcon)
# It's a dummy module that is the identity function by default
# I can wrap any intermediate activation in a HookPoint and get a convenient
# way to add PyTorch hooks
class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
        self.ctx = {}

        # A variable giving the hook's name (from the perspective of the root
        # module) - this is set by the root module at setup.
        self.name = None

    def add_hook(self, hook, dir="fwd"):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)

        if dir == "fwd":
            def full_hook(module, module_input, module_output):
                return hook(module_output, hook=self)
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == "bwd":
            # For a backwards hook, module_output is a tuple of (grad,) - I don't know why.
            def full_hook(module, module_input, module_output):
                return hook(module_output[0], hook=self)
            handle = self.register_full_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir="fwd"):
        if (dir == "fwd") or (dir == "both"):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir == "bwd") or (dir == "both"):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def clear_context(self):
        del self.ctx
        self.ctx = {}

    def forward(self, x):
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on EasyTransformer
        # If it doesn't have this form, raises an error -
        split_name = self.name.split(".")
        return int(split_name[1])


class HookedRootModule(nn.Module):
    # A class building on nn.Module to interface nicely with HookPoints
    # Allows you to name each hook, remove hooks, cache every activation/gradient, etc
    def __init__(self, *args):
        super().__init__()
        self.is_caching = False
    def setup(self):
        # Setup function - this needs to be run in __init__ AFTER defining all
        # layers
        # Add a parameter to each module giving its name
        # Build a dictionary mapping a module name to the module
        self.mod_dict = {}
        self.hook_dict = {}
        for name, module in self.named_modules():
            module.name = name
            self.mod_dict[name] = module
            if "HookPoint" in str(type(module)):
                self.hook_dict[name] = module

    def hook_points(self):
        return self.hook_dict.values()

    def remove_all_hook_fns(self, direction="both"):
        for hp in self.hook_points():
            hp.remove_hooks(direction)

    def clear_contexts(self):
        for hp in self.hook_points():
            hp.clear_context()

    def reset_hooks(self, clear_contexts=True, direction="both"):
        if clear_contexts:
            self.clear_contexts()
        self.remove_all_hook_fns(direction)
        self.is_caching = False

    def cache_all(self, cache, incl_bwd=False, device=None, remove_batch_dim=False):
        # Caches all activations wrapped in a HookPoint
        # Remove batch dim is a utility for single batch inputs that removes the batch 
        # dimension from the cached activations - use ONLY for batches of size 1
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cache_some(cache, lambda x: True, incl_bwd=incl_bwd, device=device, remove_batch_dim=remove_batch_dim)

    def cache_some(self, cache, names: Callable[[str], bool], incl_bwd=False, device=None, remove_batch_dim=False):
        """Cache a list of hook provided by names, Boolean function on names"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_caching = True
        def save_hook(tensor, hook):
            if remove_batch_dim:
                cache[hook.name] = tensor.detach().to(device)[0]
            else:
                cache[hook.name] = tensor.detach().to(device)

        def save_hook_back(tensor, hook):
            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor[0].detach().to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor[0].detach().to(device)
        for name, hp in self.hook_dict.items():
            if names(name):
                hp.add_hook(save_hook, "fwd")
                if incl_bwd:
                    hp.add_hook(save_hook_back, "bwd")

    def add_hook(self, name, hook, dir="fwd"):
        if type(name) == str:
            self.mod_dict[name].add_hook(hook, dir=dir)
        else:
            # Otherwise, name is a Boolean function on names
            for hook_name, hp in self.hook_dict.items():
                if name(hook_name):
                    hp.add_hook(hook, dir=dir)

    def run_with_hooks(
        self, *args, fwd_hooks=[], bwd_hooks=[], reset_hooks_start=True, reset_hooks_end=True, clear_contexts=False, **kwargs
    ):
        """
        fwd_hooks: A list of (name, hook), where name is either the name of
        a hook point or a Boolean function on hook names and hook is the
        function to add to that hook point, or the hook whose names evaluate
        to True respectively. Ditto bwd_hooks
        reset_hooks_start (bool): If True, all prior hooks are removed at the start
        reset_hooks_end (bool): If True, all hooks are removed at the end (ie,
        including those added in this run)
        clear_contexts (bool): If True, clears hook contexts whenever hooks are reset
        Note that if we want to use backward hooks, we need to set
        reset_hooks_end to be False, so the backward hooks are still there - this function only runs a forward pass.
        """
        if reset_hooks_start:
            if self.is_caching:
                logging.warning("Caching is on, but hooks are being reset")
            self.reset_hooks(clear_contexts)
        for name, hook in fwd_hooks:
            if type(name) == str:
                self.mod_dict[name].add_hook(hook, dir="fwd")
            else:
                # Otherwise, name is a Boolean function on names
                for hook_name, hp in self.hook_dict.items():
                    if name(hook_name):
                        hp.add_hook(hook, dir="fwd")
        for name, hook in bwd_hooks:
            if type(name) == str:
                self.mod_dict[name].add_hook(hook, dir="bwd")
            else:
                # Otherwise, name is a Boolean function on names
                for hook_name, hp in self.hook_dict:
                    if name(hook_name):
                        hp.add_hook(hook, dir="bwd")
        out = self.forward(*args, **kwargs)
        if reset_hooks_end:
            if len(bwd_hooks) > 0:
                logging.warning("WARNING: Hooks were reset at the end of run_with_hooks while backward hooks were set. This removes the backward hooks before a backward pass can occur")
            self.reset_hooks(clear_contexts)
        return out
