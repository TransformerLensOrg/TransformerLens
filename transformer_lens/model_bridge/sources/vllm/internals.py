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


def verify_hook_coverage(llm: Any) -> None:
    """Raise if any configured capture hook installed on NO rank.

    Per-rank absence is legal (pipeline-parallel ranks own layer subsets), so
    hook installation skips missing modules instead of raising — this boot-time
    check restores the fail-loud contract: a hook absent everywhere is a broken
    overlay dot-path and would otherwise read back as silent zeros.
    """
    absent_per_rank = llm.collective_rpc("tl_absent_hooks")
    if not absent_per_rank:
        return
    absent_everywhere = set(absent_per_rank[0])
    for rank_absent in absent_per_rank[1:]:
        absent_everywhere &= set(rank_absent)
    if absent_everywhere:
        raise RuntimeError(
            f"Capture hooks failed to install on every worker: {sorted(absent_everywhere)}. "
            "The overlay's dot-paths don't match this model's module tree."
        )


# Cumulative per-request query offsets: FlashAttention/Triton name them
# query_start_loc, FlashInfer names them qo_indptr. Request i = rows i:i+1.
_QUERY_OFFSET_ATTRS = ("query_start_loc", "qo_indptr")


def _to_cpu_offsets(buf: Any) -> Any:
    """Coerce a CpuGpuBuffer (``.cpu``/``.np``/``.gpu``) or tensor to a CPU tensor."""
    import torch

    for accessor in ("cpu", "np", "gpu"):
        data = getattr(buf, accessor, None)
        if data is not None and hasattr(data, "__len__"):
            return torch.as_tensor(data)
    return torch.as_tensor(buf) if hasattr(buf, "shape") else None


def segment_by_request(model_runner: Any) -> Any:
    """Return ``(query_offsets_cpu, req_ids)``; request i = rows offsets[i]:offsets[i+1].

    Only valid inside a forward. ``req_ids`` is row order, NOT submission order —
    join on it. Reads ``model_runner.query_start_loc`` (backend-agnostic; the
    runner builds it before any attention backend, whereas FlashInfer buries its
    offsets in an opaque C++ wrapper), falling back to attn metadata for backends
    that surface them directly. ``(None, req_ids)`` ⇒ caller single-slices.
    """
    req_ids = list(model_runner.input_batch.req_ids)
    n = len(req_ids)

    qsl = getattr(model_runner, "query_start_loc", None)
    if qsl is not None:
        offsets = _to_cpu_offsets(qsl)
        if offsets is not None and len(offsets) >= n + 1:  # buffer is padded to max batch
            return offsets[: n + 1].detach().cpu(), req_ids

    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, list):  # dual-batch-overlap returns a list of dicts
        attn_metadata = attn_metadata[0]
    if isinstance(attn_metadata, dict):
        for meta in attn_metadata.values():
            for attr in _QUERY_OFFSET_ATTRS:
                off = getattr(meta, attr, None)
                if off is not None:
                    return off.detach().cpu(), req_ids
    return None, req_ids
