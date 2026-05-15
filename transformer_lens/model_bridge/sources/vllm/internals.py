"""Single chokepoint for vLLM internal API access.

vLLM rearranges its internal class paths every 4-6 weeks. Centralize every
``llm.llm_engine.…`` walk here so version drift is patched in one place.
"""
from __future__ import annotations

from typing import Any


def extract_inner_model(llm: Any) -> Any:
    """Return the worker's loaded ``nn.Module`` (e.g. ``LlamaForCausalLM``).

    Path is V1 engine, single-worker (TP=PP=1). Multi-worker is a v2 concern.
    """
    try:
        return llm.llm_engine.engine_core.model_executor.driver_worker.model_runner.model
    except AttributeError as e:
        raise RuntimeError(
            "Could not locate the inner model under llm.llm_engine.engine_core.…"
            ". vLLM may have moved it; update extract_inner_model() to match."
        ) from e


def extract_hf_config(llm: Any) -> Any:
    """Return the HF config that vLLM loaded the model from."""
    try:
        return llm.llm_engine.model_config.hf_config
    except AttributeError as e:
        raise RuntimeError(
            "Could not locate hf_config under llm.llm_engine.model_config. "
            "vLLM may have moved it; update extract_hf_config() to match."
        ) from e


def walk_dot_path(root: Any, dot_path: str) -> Any:
    """Walk a dot-path with integer-segment indexing support."""
    target = root
    for seg in dot_path.split("."):
        target = target[int(seg)] if seg.isdigit() else getattr(target, seg)
    return target
