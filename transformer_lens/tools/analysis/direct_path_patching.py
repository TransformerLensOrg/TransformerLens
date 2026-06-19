"""Direct Path Patching.

Implements direct path patching — a finer-grained variant of activation patching
introduced for circuit analysis.

Background
----------
Standard activation patching (see patching.py) replaces an activation at a given
layer/position with its value from a clean run, and measures how much the model's
output shifts. But patching the *residual stream* affects ALL downstream components,
making it hard to isolate the direct information flow between two specific heads.

Direct path patching isolates the path A → B: it patches *only* the contribution of
source head A (at layer src_layer) into the input of destination head B (at layer
dst_layer > src_layer), leaving every other component's view of A's output unchanged.

The linear approximation used here (following Neel Nanda's description in issue #111)
is:

    delta_resid = clean_A_result - corrupted_A_result   # [batch, pos, d_model]
    delta_q     = (delta_resid / ln1_scale) @ W_Q[hb]  # [batch, pos, d_head]
    patched_q   = corrupted_q + delta_q

This is exact under linear layer norm (no learned offset changes the scale
in a way that matters for the perturbation), and matches the gradient-based
approximation used in attribution patching.

Usage
-----
    # 1. Cache clean and corrupted activations
    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # 2. Define your metric (same as activation patching)
    def metric(logits):
        return logit_diff(logits, ...)

    # 3. Sweep all (dst_layer, dst_head) pairs for a fixed source head
    results = get_act_patch_direct_path(
        model, corrupted_tokens, clean_cache, corrupted_cache,
        metric, src_layer=9, src_head=9,
        component="q",   # patch into Q; also supports "k", "v"
    )
    # results.shape == (n_layers, n_heads)
    # results[dst_layer, dst_head] = metric when A→B path is patched

References
----------
- Neel Nanda, TransformerLens issue #111 (2022)
- Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object
  Identification in GPT-2 small" (2022)
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal, Union

import torch
from jaxtyping import Float
from tqdm.auto import tqdm

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_fold_ln(model: Union["HookedTransformer", "TransformerBridge"]) -> None:
    """Warn if the model's LayerNorm weights have not been folded in.

    HookedTransformer stores the learned scale as ``.w``; TransformerBridge wraps
    the original HuggingFace module, which stores it as ``.weight``.  We check
    both so the guard works for either system.
    """
    try:
        ln1 = model.blocks[0].ln1  # type: ignore[index]
        # .w  → HookedTransformer; .weight → TransformerBridge (wraps HF module)
        w = getattr(ln1, "w", None)
        if w is None:
            w = getattr(ln1, "weight", None)
        if w is not None and not torch.allclose(w, torch.ones_like(w), atol=1e-3):
            warnings.warn(
                "get_act_patch_direct_path is most accurate when LayerNorm parameters "
                "are folded into the weight matrices. "
                "For HookedTransformer: pass fold_ln=True to from_pretrained, or call "
                "model.process_weights_(). "
                "For TransformerBridge: call model.process_weights(fold_ln=True). "
                "Results may be inaccurate with unfolded LayerNorm.",
                UserWarning,
                stacklevel=3,
            )
    except (AttributeError, TypeError):
        pass  # non-standard model — cannot inspect LN weights, proceed


# ---------------------------------------------------------------------------
# Core hook factory
# ---------------------------------------------------------------------------


def _make_direct_path_hook(
    delta_resid: Float[torch.Tensor, "batch pos d_model"],
    dst_head: int,
    W_component: Float[torch.Tensor, "d_model d_head"],
    ln_scale_name: str,
    corrupted_cache: ActivationCache,
    component: Literal["q", "k", "v"],
) -> Callable:
    """Return a hook function that adds the linearised delta to one head's Q, K, or V.

    Parameters
    ----------
    delta_resid:
        (clean_A_result - corrupted_A_result), shape [batch, pos, d_model].
    dst_head:
        Index of the destination attention head to patch.
    W_component:
        The weight matrix for the component being patched:
        W_Q[dst_head], W_K[dst_head], or W_V[dst_head].
        Shape [d_model, d_head].
    ln_scale_name:
        Cache key for the layer-norm scale at the destination layer,
        e.g. "blocks.3.ln1.hook_scale".
    corrupted_cache:
        Cache from the corrupted forward pass (used to look up ln1 scale).
    component:
        One of "q", "k", "v" — determines which QKV tensor is hooked.
    """

    def hook_fn(
        value: Float[torch.Tensor, "batch pos n_heads d_head"],
        hook,  # HookPoint, unused but required by TransformerLens
    ) -> Float[torch.Tensor, "batch pos n_heads d_head"]:
        # ln scale: [batch, pos, 1]
        ln_scale = corrupted_cache[ln_scale_name]  # [batch, pos, 1]

        # Linearised delta in query/key/value space
        # delta_resid: [batch, pos, d_model]
        # W_component: [d_model, d_head]
        delta = (delta_resid / ln_scale) @ W_component  # [batch, pos, d_head]

        if value.requires_grad:
            value = value.clone()
        value[:, :, dst_head, :] = value[:, :, dst_head, :] + delta
        return value

    return hook_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_act_patch_direct_path(
    model: Union[HookedTransformer, TransformerBridge],
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
    src_layer: int,
    src_head: int,
    component: Literal["q", "k", "v"] = "q",
    verbose: bool = True,
) -> Float[torch.Tensor, "n_layers n_heads"]:
    """Sweep direct path patches from one source head to all downstream heads.

    For every destination head B = (dst_layer, dst_head) where dst_layer > src_layer,
    patch the contribution of source head A = (src_layer, src_head) into B's query
    (or key / value) input, and record the patching metric.

    The patch is a linear approximation:

        delta_resid  = clean_A_result - corrupted_A_result   [batch, pos, d_model]
        delta_B_comp = (delta_resid / ln1_scale) @ W_comp[dst_head]

    where W_comp is W_Q, W_K, or W_V according to `component`.

    Parameters
    ----------
    model:
        A HookedTransformer or TransformerBridge instance.
    corrupted_tokens:
        Token IDs for the corrupted input, shape [batch, seq_len].
    clean_cache:
        Cached activations from the clean (unpatched) run.
    corrupted_cache:
        Cached activations from the corrupted run (needed for ln1 scale).
    patching_metric:
        A function mapping the model's logits tensor to a scalar.
    src_layer:
        Layer index of the source attention head.
    src_head:
        Head index of the source attention head.
    component:
        Which input to patch at the destination head — "q" (default), "k", or "v".
    verbose:
        Whether to show a tqdm progress bar.

    Returns
    -------
    results : Float[Tensor, "n_layers n_heads"]
        results[dst_layer, dst_head] is the patching metric when the direct path
        A → B is patched in.  Entries for dst_layer <= src_layer are left as 0.0
        (no causal path from A to those layers).
    """
    _check_fold_ln(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = torch.zeros(n_layers, n_heads, device=model.cfg.device)

    # Residual stream delta from source head A.
    #
    # hook_result (per-head residual contribution) requires cfg.use_hook_result=True
    # and is not in the default cache. We compute it instead from hook_z and W_O,
    # which are always available:
    #   result_h = z[:, :, h, :] @ W_O[h]   shape [batch, pos, d_model]
    src_z_name = f"blocks.{src_layer}.attn.hook_z"
    W_O = model.blocks[src_layer].attn.W_O  # type: ignore[union-attr]  # [n_heads, d_head, d_model]

    def _head_result(cache, h):
        z = cache[src_z_name][:, :, h, :]  # [batch, pos, d_head]
        return z @ W_O[h]  # type: ignore[index]  # [batch, pos, d_model]

    delta_resid = _head_result(clean_cache, src_head) - _head_result(corrupted_cache, src_head)
    # shape: [batch, pos, d_model]

    # Weight matrix for the component being patched
    _comp_map = {
        "q": lambda attn: attn.W_Q,  # [n_heads, d_model, d_head]
        "k": lambda attn: attn.W_K,
        "v": lambda attn: attn.W_V,
    }
    _hook_name_map = {
        "q": lambda lb: f"blocks.{lb}.attn.hook_q",
        "k": lambda lb: f"blocks.{lb}.attn.hook_k",
        "v": lambda lb: f"blocks.{lb}.attn.hook_v",
    }
    W_all = _comp_map[component]  # callable: attn → [n_heads, d_model, d_head]
    hook_name_fn = _hook_name_map[component]

    dst_pairs = [(lb, hb) for lb in range(src_layer + 1, n_layers) for hb in range(n_heads)]

    for dst_layer, dst_head in tqdm(
        dst_pairs,
        desc=f"Direct path patch ({src_layer},{src_head}) → * [{component}]",
        disable=not verbose,
    ):
        ln_scale_name = f"blocks.{dst_layer}.ln1.hook_scale"
        W_comp = W_all(model.blocks[dst_layer].attn)[dst_head]  # type: ignore[index]  # [d_model, d_head]

        hook_fn = _make_direct_path_hook(
            delta_resid=delta_resid,
            dst_head=dst_head,
            W_component=W_comp,
            ln_scale_name=ln_scale_name,
            corrupted_cache=corrupted_cache,
            component=component,
        )

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name_fn(dst_layer), hook_fn)],
        )

        results[dst_layer, dst_head] = patching_metric(patched_logits).item()

    return results


def get_act_patch_direct_path_all_sources(
    model: Union[HookedTransformer, TransformerBridge],
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    patching_metric: Callable[[torch.Tensor], torch.Tensor],
    component: Literal["q", "k", "v"] = "q",
    verbose: bool = True,
) -> Float[torch.Tensor, "n_layers n_heads n_layers n_heads"]:
    """Full sweep: all (src_layer, src_head) → (dst_layer, dst_head) direct paths.

    Returns a 4-D tensor of shape [n_layers, n_heads, n_layers, n_heads].
    result[sl, sh, dl, dh] = patching metric when head (sl,sh)'s output is
    patched directly into head (dl,dh)'s query/key/value input.

    Entries where dl <= sl are 0 (no causal path).

    This runs O(n_layers * n_heads * n_layers * n_heads) forward passes and is
    intended for small models or targeted sub-sweeps.  For large models prefer
    calling get_act_patch_direct_path per source head.
    """
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
