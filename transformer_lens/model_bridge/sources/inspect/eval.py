"""Inspect solver for capturing TransformerLens activations during an eval.

``capture_activations([...])`` is a solver you add to a Task's solver chain to harvest
activations alongside a behavioral eval (with a ``tl_bridge``-served model). Full
activations go to a per-sample side artifact (``output_dir/<sample_id>.npz``) for
probing/SAE; a compact ``reduce(...)`` summary lands in the sample store so
``inspect_ai.analysis.samples_df`` can correlate activation features with scores.

Imports ``inspect_ai`` at module load (like ``provider.py``); the package ``__init__``
exposes ``capture_activations`` lazily so importing the package stays inspect_ai-free.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from . import hooks, wire


def _default_reduce(activations: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Per-hook L2 norm + shape — small, JSON-able, queryable in samples_df."""
    return {
        name: {"l2": float(np.linalg.norm(arr)), "shape": list(arr.shape)}
        for name, arr in activations.items()
    }


@solver
def capture_activations(
    capture: Sequence[str],
    output_dir: str = "tl_activations",
    reduce: Optional[Callable[[Mapping[str, np.ndarray]], dict[str, Any]]] = None,
    store_key: str = "tl_activations",
) -> Solver:
    """Capture ``capture`` hooks for each sample's current messages.

    Writes full activations to ``output_dir/<sample_id>.npz`` and a ``reduce(...)`` summary
    (default: per-hook L2 + shape) to the sample store under ``store_key`` (+ ``_path``).
    Requires a ``tl_bridge``-served model; raises if the model returns no activations.
    Place before ``generate()`` to capture the prompt, after it to include the completion.
    """
    # Resolve at construction so a multi-eval run from different CWDs doesn't scatter
    # artifacts (the solver runs later, possibly under a different working directory).
    output_dir = os.path.abspath(output_dir)
    reduce_fn = reduce or _default_reduce
    # TL hook names → provider wire keys (and back, to key the saved arrays by hook name).
    name_by_wire = {}
    for name in capture:
        resolved = hooks.resolve(name)
        if resolved is None:
            raise ValueError(f"capture_activations: {name!r} is not a fireable hook name.")
        name_by_wire[hooks.wire_key(*resolved)] = name
    wire_keys = list(name_by_wire)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # get_model() (no args) is the eval's active model; state.model is just its name.
        output = await get_model().generate(
            state.messages,
            config=GenerateConfig(
                extra_body={"extra_args": {"capture": wire_keys, "return_logits": False}}
            ),
        )
        decoded = wire.decode_activations(getattr(output, "metadata", None), wire_keys)
        if not decoded:
            raise RuntimeError(
                "capture_activations got no activations back — the eval model must be a "
                "tl_bridge provider (e.g. model='tl_bridge/gpt2')."
            )
        activations = {name_by_wire[wk]: arr for wk, arr in decoded.items()}
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{state.sample_id}.npz")
        np.savez_compressed(path, **activations)
        store().set(store_key, reduce_fn(activations))
        store().set(f"{store_key}_path", path)
        return state

    return solve


def turn_activations(sample: Any) -> list[dict[str, np.ndarray]]:
    """Per-turn activations from an eval sample's model events, for a provider booted with
    ``capture=[...]`` (e.g. ``model_args={"capture": [...]}``). Returns one dict per model
    generation, in turn order — the activations of an agentic/multi-turn rollout. Every
    array gets a leading batch dim: boundaries are ``(1, seq, d_model)``, head-split
    q/k/v/z ``(1, seq, heads, d_head)``.
    """
    turns = []
    for event in getattr(sample, "events", []) or []:
        metadata = getattr(getattr(event, "output", None), "metadata", None)
        if not metadata or "activations" not in metadata:
            continue
        decoded = wire.decode_activations(metadata, list(metadata["activations"]))
        named = {}
        for wk, arr in decoded.items():
            name = hooks.name_from_wire_key(wk)
            if name is None:
                continue
            # Rank-aware batch dim (mirrors driver._assemble_captures): boundary kinds
            # arrive rank-2, head-split kinds rank-3 — unsqueeze exactly once either way.
            batchless = hooks.WIRE_BATCHLESS_NDIM.get(wk.partition(":")[2], 2)
            named[name] = arr[np.newaxis, ...] if arr.ndim == batchless else arr
        if named:
            turns.append(named)
    return turns


def activations_column(store_key: str = "tl_activations", name: Optional[str] = None) -> Any:
    """A ``samples_df`` column surfacing :func:`capture_activations`'s reduction, for
    correlating activation features with scores::

        df = samples_df(logs, columns=[*SampleSummary, activations_column()])
    """
    from inspect_ai.analysis import SampleColumn

    return SampleColumn(name or store_key, path=lambda s: s.store.get(store_key), full=True)


__all__ = ["activations_column", "capture_activations", "turn_activations"]
