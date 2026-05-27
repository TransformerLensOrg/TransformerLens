"""Single chokepoint for vLLM internal API access.

vLLM rearranges its internal class paths every 4-6 weeks. Centralize every
``llm.llm_engine.…`` walk here so version drift is patched in one place.

**Validated against ``vllm==0.20.2``** (also the version pinned in
``demos/vLLM_Bridge_Integration_Test.ipynb``). The patched-load-model path in
``plugin.py`` and the ``hf_config`` walk below have been confirmed on that
release; newer releases may move attributes — re-validate before bumping.
"""
from __future__ import annotations

from typing import Any


def extract_hf_config(llm: Any) -> Any:
    """Return the HF config that vLLM loaded the model from."""
    try:
        return llm.llm_engine.model_config.hf_config
    except AttributeError as e:
        raise RuntimeError(
            "Could not locate hf_config under llm.llm_engine.model_config. "
            "vLLM may have moved it; update extract_hf_config() to match."
        ) from e


def segment_by_request(model_runner: Any) -> Any:
    """Return ``(query_start_loc_cpu, req_ids)`` for the current forward.

    Per-request token boundaries (``query_start_loc``) live on the per-layer
    attention metadata in the active forward context, not on a stable engine
    field, so this is only valid called from *inside* a forward (e.g. a layer
    hook). ``req_ids`` maps each batch row to a request id and is NOT guaranteed
    to be submission order — callers must join on it, not on position.

    Returns ``(None, req_ids)`` if no metadata carries ``query_start_loc`` (e.g.
    hybrid attention backends); the caller falls back to a single-request slice.
    """
    from vllm.forward_context import get_forward_context

    req_ids = list(model_runner.input_batch.req_ids)
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, list):  # dual-batch-overlap returns a list of dicts
        attn_metadata = attn_metadata[0]
    if attn_metadata is None:
        return None, req_ids
    for meta in attn_metadata.values():
        qsl = getattr(meta, "query_start_loc", None)
        if qsl is not None:
            return qsl.detach().cpu(), req_ids
    return None, req_ids
