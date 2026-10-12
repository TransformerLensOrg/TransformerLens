"""Direct path patching (after Wang et al. 2022): add only head A's output delta to a later head
B's q/k/v, linearised through the corrupted run's frozen ln1 scale (inexact even with LN folded)."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal, cast

import torch
from jaxtyping import Float
from tqdm.auto import tqdm

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_protocol import (
    TransformerLensModel,
    TransformerLensModelWithWeights,
)


def _check_fold_ln(model: Any) -> None:
    """Warn if the model's LayerNorm weights have not been folded in."""
    try:
        ln1 = model.blocks[0].ln1
        # .weight on TransformerBridge (wraps the HF module); .w on legacy TL components.
        w = getattr(ln1, "w", None)
        if w is None:
            w = getattr(ln1, "weight", None)
        if w is not None and getattr(getattr(model, "cfg", None), "rmsnorm_uses_offset", False):
            # Offset-RMS (Gemma) stores w as an offset from 1, so the folded identity is w = 0.
            w = 1.0 + w
        if w is not None and not torch.allclose(w, torch.ones_like(w), atol=1e-3):
            warnings.warn(
                "get_act_patch_direct_path is most accurate when LayerNorm parameters "
                "are folded into the weight matrices: call "
                "model.process_weights(fold_ln=True) on the TransformerBridge. "
                "Results may be inaccurate with unfolded LayerNorm.",
                UserWarning,
                stacklevel=3,
            )
    except (AttributeError, TypeError):
        pass  # non-standard model: LN weights can't be inspected, so don't block the run


def _make_direct_path_hook(
    delta_resid: Float[torch.Tensor, "batch pos d_model"],
    dst_head: int,
    W_component: Float[torch.Tensor, "d_model d_head"],
    ln_scale_name: str,
    corrupted_cache: ActivationCache,
) -> Callable:
    """Hook adding the linearised ``delta_resid`` to ``dst_head``'s q, k or v."""

    def hook_fn(
        value: Float[torch.Tensor, "batch pos n_heads d_head"],
        hook,  # unused; required by the hook signature
    ) -> Float[torch.Tensor, "batch pos n_heads d_head"]:
        # Frozen ln1 scale [batch, pos, 1]; error grows as delta shifts the residual norm.
        ln_scale = corrupted_cache[ln_scale_name]
        delta = (delta_resid / ln_scale) @ W_component

        if value.requires_grad:
            value = value.clone()
        value[:, :, dst_head, :] = value[:, :, dst_head, :] + delta
        return value

    return hook_fn


def get_act_patch_direct_path(
    model: TransformerLensModel,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
    src_layer: int,
    src_head: int,
    component: Literal["q", "k", "v"] = "q",
    verbose: bool = True,
) -> Float[torch.Tensor, "n_layers n_heads"]:
    """Patch head (src_layer, src_head) directly into each later head's ``component``.
    Returns the metric per ``[dst_layer, dst_head]``; entries with dst_layer <= src_layer stay 0."""
    _check_fold_ln(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = torch.zeros(n_layers, n_heads, device=model.cfg.device)

    # Rebuilt from hook_z @ W_O: hook_result is only cached when cfg.use_hook_result is set.
    src_z_name = f"blocks.{src_layer}.attn.hook_z"
    # Static-only cast: nn.Module submodules are invisible to getattr_static-based isinstance.
    weights_model = cast(TransformerLensModelWithWeights, model)
    W_O = weights_model.blocks[src_layer].attn.W_O

    def _head_result(cache, h):
        z = cache[src_z_name][:, :, h, :]
        return z @ W_O[h]

    delta_resid = _head_result(clean_cache, src_head) - _head_result(corrupted_cache, src_head)

    _comp_map = {
        "q": lambda attn: attn.W_Q,
        "k": lambda attn: attn.W_K,
        "v": lambda attn: attn.W_V,
    }
    _hook_name_map = {
        "q": lambda lb: f"blocks.{lb}.attn.hook_q",
        "k": lambda lb: f"blocks.{lb}.attn.hook_k",
        "v": lambda lb: f"blocks.{lb}.attn.hook_v",
    }
    W_all = _comp_map[component]
    hook_name_fn = _hook_name_map[component]

    dst_pairs = [(lb, hb) for lb in range(src_layer + 1, n_layers) for hb in range(n_heads)]

    for dst_layer, dst_head in tqdm(
        dst_pairs,
        desc=f"Direct path patch ({src_layer},{src_head}) → * [{component}]",
        disable=not verbose,
    ):
        ln_scale_name = f"blocks.{dst_layer}.ln1.hook_scale"
        W_comp = W_all(weights_model.blocks[dst_layer].attn)[dst_head]

        hook_fn = _make_direct_path_hook(
            delta_resid=delta_resid,
            dst_head=dst_head,
            W_component=W_comp,
            ln_scale_name=ln_scale_name,
            corrupted_cache=corrupted_cache,
        )

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name_fn(dst_layer), hook_fn)],
        )

        results[dst_layer, dst_head] = patching_metric(patched_logits).item()

    return results


def get_act_patch_direct_path_all_sources(
    model: TransformerLensModel,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
    component: Literal["q", "k", "v"] = "q",
    verbose: bool = True,
) -> Float[torch.Tensor, "n_layers n_heads n_layers n_heads"]:
    """Every source head into every later head: ``result[sl, sh, dl, dh]``, 0 where dl <= sl.
    One forward pass per (source, destination) pair; prefer per-source calls on large models."""
    _check_fold_ln(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = torch.zeros(n_layers, n_heads, n_layers, n_heads, device=model.cfg.device)

    src_pairs = [(sl, sh) for sl in range(n_layers) for sh in range(n_heads)]
    for src_layer, src_head in tqdm(
        src_pairs,
        desc=f"Direct path patch — all sources [{component}]",
        disable=not verbose,
    ):
        results[src_layer, src_head] = get_act_patch_direct_path(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            patching_metric=patching_metric,
            src_layer=src_layer,
            src_head=src_head,
            component=component,
            verbose=False,
        )

    return results
