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
