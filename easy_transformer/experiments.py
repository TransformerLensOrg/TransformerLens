# Ablation implem
# Import stuff
from typing import Callable, Union, List, Tuple
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

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.utils import gelu_new, to_numpy, get_corner, print_gpu_mem, get_sample_from_dataset
from easy_transformer.EasyTransformer import EasyTransformer


class ExperimentMetric:
    def __init__(
        self,
        metric: Callable[[EasyTransformer, List[str]], torch.Tensor],
        dataset: List[str],
        scalar_metric=True,
        relative_metric=True,
    ):
        self.relative_metric = relative_metric
        self.metric = metric  # metric can return any tensor shape. Can call run_with_hook with reset_hooks_start=False
        self.scalar_metric = scalar_metric
        self.baseline = None  # metric without ablation
        self.dataset = dataset
        self.shape = None

    def set_baseline(self, model):
        model.reset_hooks()
        base_metric = self.metric(model, self.dataset)
        self.baseline = base_metric
        self.shape = base_metric.shape

    def compute_metric(self, model):
        assert (self.baseline is not None) or not (self.relative_metric), "Baseline has not been set in relative mean"
        out = self.metric(model, self.dataset)
        if self.scalar_metric:
            assert len(out.shape) == 0, "Output of scalar metric has shape of length > 0"
        self.shape = out.shape
        if self.relative_metric:
            out = (out / self.baseline) - 1
        return out


class ExperimentConfig:
    def __init__(
        self,
        target_module: str = "attn_head",
        layers: Union[Tuple[int, int], str] = "all",
        heads: Union[List[int], str] = "all",
        verbose: bool = False,
        head_circuit: str = "z",
    ):
        assert target_module in ["mlp", "attn_layer", "attn_head"]
        assert head_circuit in ["z", "q", "v", "k", "attn", "attn_scores", "result"]

        self.target_module = target_module
        self.head_circuit = head_circuit
        self.layers = layers
        self.heads = heads
        self.dataset = None
        self.verbose = verbose

        self.beg_layer = None  # layers where the ablation begins and ends
        self.end_layer = None

    def adapt_to_model(self, model: EasyTransformer):
        """Return a new experiment config that fits the model."""
        model_cfg = self.copy()
        if self.target_module == "attn_head":
            if self.heads == "all":
                model_cfg.heads = list(range(model.cfg.n_heads))

        if self.layers == "all":
            model_cfg.beg_layer = 0
            model_cfg.end_layer = model.cfg.n_layers
        else:
            model_cfg.beg_layer, model_cfg.end_layer = self.layers
        return model_cfg

    def copy(self):
        copy = self.__class__()
        for name, attr in vars(self).items():
            if type(attr) == list:
                setattr(copy, name, attr.copy())
            else:
                setattr(copy, name, attr)
        return copy

    def __str__(self):
        str_print = f"--- {self.__class__.__name__}: ---\n"
        for name, attr in vars(self).items():
            attr = getattr(self, name)
            attr_str = f"* {name}: "

            if name == "mean_dataset" and self.compute_means and attr is not None:
                attr_str += get_sample_from_dataset(self.mean_dataset)
            elif name == "dataset" and attr is not None:
                attr_str += get_sample_from_dataset(self.dataset)
            else:
                attr_str += str(attr)
            attr_str += "\n"
            str_print += attr_str
        return str_print

    def __repr__(self):
        return self.__str__()


def zero_fn(z, hk):
    return torch.zeros(z.shape)


def cst_fn(z, cst, hook):
    return cst


def neg_fn(z, hk):
    return -z


class AblationConfig(ExperimentConfig):
    def __init__(
        self,
        abl_type: str = "zero",
        mean_dataset: List[str] = None,
        cache_means: bool = True,
        abl_fn: Callable[[torch.tensor, torch.tensor, HookPoint], torch.tensor] = None,
        **kwargs,
    ):
        assert abl_type in ["mean", "zero", "neg", "custom"]
        assert not (abl_type == "custom" and abl_fn is None), "You must specify you ablation function"
        super().__init__(**kwargs)

        self.abl_type = abl_type
        self.mean_dataset = mean_dataset
        self.dataset = None
        self.cache_means = cache_means
        self.compute_means = abl_type == "mean" or abl_type == "custom"
        self.abl_fn = abl_fn

        if abl_type == "zero":
            self.abl_fn = zero_fn
        if abl_type == "neg":
            self.abl_fn = neg_fn
        if abl_type == "mean":
            self.abl_fn = cst_fn


class PatchingConfig(ExperimentConfig):
    """Configuration for patching activations from the source dataset to the target dataset"""

    def __init__(
        self,
        source_dataset: List[str] = None,
        target_dataset: List[str] = None,
        patch_fn: Callable[[torch.tensor, torch.tensor, HookPoint], torch.tensor] = None,
        cache_act: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.cache_act = cache_act  # if we should cache activation. Take more GPU memory but faster to run
        self.patch_fn = patch_fn
        if patch_fn is None:  # default patch_fn
            self.patch_fn = cst_fn


# TODO : loss metric, zero ablation, mean ablation
# TODO : add different direction for the mean, add tokens, change type of
# datasets, make deterministic


class EasyExperiment:
    """A virtual class to interatively apply hooks to layers or heads. The children class only needs to define the methods
    get_hook"""

    def __init__(self, model: EasyTransformer, config: ExperimentConfig, metric: ExperimentMetric):
        self.model = model
        self.metric = metric
        self.cfg = config.adapt_to_model(model)
        self.cfg.dataset = self.metric.dataset

    def run_experiment(self):
        self.metric.set_baseline(self.model)
        results = torch.empty(self.get_result_shape())
        if self.cfg.verbose:
            print(self.cfg)

        self.metric.set_baseline(self.model)
        results = torch.empty(self.get_result_shape())
        for layer in tqdm(range(self.cfg.beg_layer, self.cfg.end_layer)):
            if self.cfg.target_module == "attn_head":
                for head in self.cfg.heads:
                    hook = self.get_hook(layer, head)
                    results[layer, head] = self.compute_metric(hook).cpu().detach()
            else:
                hook = self.get_hook(layer)
                results[layer] = self.compute_metric(hook).cpu().detach()
        self.model.reset_hooks()
        if len(results.shape) < 2:
            results = results.unsqueeze(0)  # to make sure that we can always easily plot the results
        return results

    def get_result_shape(self):
        if self.cfg.target_module == "attn_head":
            return (self.cfg.end_layer - self.cfg.beg_layer, len(self.cfg.heads)) + self.metric.shape
        else:
            return (self.cfg.end_layer - self.cfg.beg_layer,) + self.metric.shape

    def compute_metric(self, abl_hook):
        self.model.reset_hooks()
        hk_name, hk = abl_hook
        self.model.add_hook(hk_name, hk)
        return self.metric.compute_metric(self.model)

    def get_target(self, layer, head):
        if head is not None:
            hook_name = f"blocks.{layer}.attn.hook_{self.cfg.head_circuit}"
            dim = (
                1 if "hook_attn" in hook_name else 2
            )  # hook_attn and hook_attn_scores are [batch,nb_head,seq_len, seq_len] and the other activation of head (z, q, v,k) are [batch, seq_len, nb_head, head_dim]
        else:
            if self.cfg.target_module == "mlp":
                hook_name = f"blocks.{layer}.mlp.hook_post"
            else:
                hook_name = f"blocks.{layer}.hook_attn_out"
            dim = None  # all the activation dimensions are ablated
        return hook_name, dim


class EasyAblation(EasyExperiment):
    """
    Run an ablation experiment according to the config object
    Pass semantic_indices not None to average across different index positions
    (probably limited used currently, see test_experiments for one usage)
    """

    def __init__(self, model: EasyTransformer, config: AblationConfig, metric: ExperimentMetric, semantic_indices=None):
        super().__init__(model, config, metric)
        assert "AblationConfig" in str(type(config))
        assert not (
            (semantic_indices is not None) and (config.head_circuit in ["hook_attn_scores", "hook_attn"])
        )  # not implemented (surely not very useful)
        self.semantic_indices = semantic_indices
        if self.cfg.mean_dataset is None and config.compute_means:
            self.cfg.mean_dataset = self.metric.dataset
        if self.cfg.cache_means and self.cfg.compute_means:
            self.get_all_mean()

    def run_ablation(self):
        return self.run_experiment()

    def get_hook(self, layer, head=None):
        # If the target is a layer, head is None.
        hook_name, dim = self.get_target(layer, head)
        mean = None
        if self.cfg.compute_means:
            if self.cfg.cache_means:
                mean = self.mean_cache[hook_name]
            else:
                mean = self.get_mean(hook_name)

        abl_hook = get_act_hook(self.cfg.abl_fn, mean, head, dim=dim)
        return (hook_name, abl_hook)

    def get_all_mean(self):
        cache = {}
        self.model.reset_hooks()
        self.model.cache_all(cache)
        self.model(self.cfg.mean_dataset)
        self.mean_cache = {}
        for hk in cache.keys():
            if "blocks" in hk:
                self.mean_cache[hk] = self.compute_mean(cache[hk], hk)

    def get_mean(self, hook_name):
        cache = {}

        def cache_hook(z, hook):
            cache[hook_name] = z.detach().to("cuda")

        self.model.reset_hooks()
        self.model.run_with_hooks(self.cfg.mean_dataset, fwd_hooks=[(hook_name, cache_hook)])
        return self.compute_mean(cache[hook_name], hook_name)

    def compute_mean(self, z, hk_name):
        mean = torch.mean(z, dim=0, keepdim=False).detach().clone()  # we compute the mean along the batch dim
        mean = einops.repeat(mean, "... -> s ...", s=z.shape[0])
        if self.semantic_indices is None or "hook_attn" in hk_name:
            return mean
        dataset_length = len(self.cfg.mean_dataset)
        for semantic_symbol, semantic_indices in self.semantic_indices.items():
            mean[list(range(dataset_length)), semantic_indices] = einops.repeat(
                torch.mean(z[list(range(dataset_length)), semantic_indices], dim=0, keepdim=False).clone(),
                "... -> s ...",
                s=dataset_length,  # instead of the mean constant accross position, for semantic indices, when do semantic ablations
            )
        return mean


class EasyPatching(EasyExperiment):
    def __init__(self, model: EasyTransformer, config: PatchingConfig, metric: ExperimentMetric):
        super().__init__(model, config, metric)
        assert "PatchingConfig" in str(type(config))
        if self.cfg.cache_act:
            self.get_all_act()

    def run_patching(self):
        return self.run_experiment()

    def get_hook(self, layer, head=None):
        # If the target is a layer, head is None.
        hook_name, dim = self.get_target(layer, head)
        if self.cfg.cache_act:
            act = self.act_cache[hook_name]  # activation on the source dataset
        else:
            act = self.get_act(hook_name)

        hook = get_act_hook(self.cfg.patch_fn, act, head, dim=dim)
        return (hook_name, hook)

    def get_all_act(self):
        self.act_cache = {}
        self.model.reset_hooks()
        self.model.cache_all(self.act_cache)
        self.model(self.cfg.source_dataset)

    def get_act(self, hook_name):
        cache = {}

        def cache_hook(z, hook):
            cache[hook_name] = z.detach().to("cuda")

        self.model.reset_hooks()
        self.model.run_with_hooks(self.cfg.source_dataset, fwd_hooks=[(hook_name, cache_hook)])
        return cache[hook_name]


def get_act_hook(fn, alt_act=None, idx=None, dim=None):
    """Return an hook that modify the activation on the fly. alt_act (Alternative activations) is a tensor of the same shape of the z.
    E.g. It can be the mean activation or the activations on other dataset."""
    if alt_act is not None:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim

            if dim is None:  # mean and z have the same shape, the mean is constant along the batch dimension
                return fn(z, alt_act, hook)
            if dim == 0:
                z[idx] = fn(z[idx], alt_act[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], alt_act[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], alt_act[:, :, idx], hook)
            return z

    else:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            if dim is None:
                return fn(z, hook)
            if dim == 0:
                z[idx] = fn(z[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], hook)
            return z

    return custom_hook
