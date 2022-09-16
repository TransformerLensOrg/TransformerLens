# Import stuff
from typing import Callable, Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm

from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    print_gpu_mem,
    get_sample_from_dataset,
)
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
)


def test_semantic_ablation():
    """
    Compute semantic ablation
    in a manual way, and then 
    in the experiments.py way and check that they agree
    """

    # so we don't have to add the IOI dataset object to this library...
    ioi_text_prompts = [
        "Then, Christina and Samantha were working at the grocery store. Samantha decided to give a kiss to Christina",
        "Then, Samantha and Christina were working at the grocery store. Christina decided to give a kiss to Samantha",
        "When Timothy and Dustin got a kiss at the grocery store, Dustin decided to give it to Timothy",
        "When Dustin and Timothy got a kiss at the grocery store, Timothy decided to give it to Dustin",
    ]
    ioi_io_ids = [33673, 34778, 22283, 37616]
    ioi_s_ids = [34778, 33673, 37616, 22283]
    ioi_end_idx = [18, 18, 17, 17]
    semantic_indices = {"IO": [2, 2, 1, 1], "S": [4, 4, 3, 3], "S2": [12, 12, 12, 12]}
    L = len(ioi_text_prompts)

    def logit_diff(model, text_prompts):
        """Difference between the IO and the S logits (at the "to" token)"""
        logits = model(text_prompts).detach()
        IO_logits = logits[torch.arange(len(text_prompts)), ioi_end_idx, ioi_io_ids]
        S_logits = logits[torch.arange(len(text_prompts)), ioi_end_idx, ioi_s_ids]
        return (IO_logits - S_logits).mean().detach().cpu()

    model = EasyTransformer("gpt2", use_attn_result=True)
    if torch.cuda.is_available():
        model.to("cuda")

    # compute in the proper way
    metric = ExperimentMetric(
        metric=logit_diff, dataset=ioi_text_prompts, relative_metric=True
    )
    config = AblationConfig(
        abl_type="mean",
        mean_dataset=ioi_text_prompts,
        target_module="attn_head",
        head_circuit="result",
        cache_means=True,
        verbose=True,
    )
    abl = EasyAblation(model, config, metric, semantic_indices=semantic_indices)
    result = abl.run_ablation()

    # compute in a manual way
    model.reset_hooks()
    cache = {}
    model.cache_all(cache)
    logits = model(ioi_text_prompts)
    io_logits = logits[list(range(L)), ioi_end_idx, ioi_io_ids]
    s_logits = logits[list(range(L)), ioi_end_idx, ioi_s_ids]
    diff_logits = io_logits - s_logits
    avg_logits = diff_logits.mean()
    max_seq_length = cache["hook_embed"].shape[1]
    assert list(cache["hook_embed"].shape) == [
        L,
        max_seq_length,
        model.cfg.d_model,
    ], cache["hook_embed"].shape
    average_activations = {}
    for key in cache.keys():
        if "attn.hook_result" not in key:
            continue
        tens = cache[key].detach().cpu()
        avg_tens = torch.mean(tens, dim=0, keepdim=False)
        cache[key] = einops.repeat(avg_tens, "... -> s ...", s=L)

        for thing in ["IO", "S", "S2"]:
            thing_average = (
                tens[list(range(L)), semantic_indices[thing], :, :]
                .detach()
                .cpu()
                .mean(dim=0)
            )
            cache[key][
                list(range(L)), semantic_indices[thing], :, :
            ] = thing_average.clone()
    diffs = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    diffs += avg_logits.item()
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            new_val = (
                cache[f"blocks.{layer}.attn.hook_result"][:, :, head, :]
                .detach()
                .clone()
            )

            def ablate_my_head(x, hook):
                x[:, :, head, :] = new_val
                return x

            model.reset_hooks()
            new_logits = model.run_with_hooks(
                ioi_text_prompts,
                fwd_hooks=[(f"blocks.{layer}.attn.hook_result", ablate_my_head)],
            )

            new_io_logits = new_logits[list(range(L)), ioi_end_idx, ioi_io_ids]
            new_s_logits = new_logits[list(range(L)), ioi_end_idx, ioi_s_ids]
            new_diff_logits = new_io_logits - new_s_logits
            new_avg_logits = new_diff_logits.mean()
            diffs[layer][head] /= new_avg_logits.item()
    diffs -= 1.0

    assert torch.allclose(
        diffs, result, rtol=1e-4, atol=1e-4
    ), f"{get_corner(diffs)}, {get_corner(result)}"


if __name__ == "__main__":
    test_semantic_ablation()
