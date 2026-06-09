"""Skip doctest-modules collection of inspect-provider files when ``inspect_ai`` is not
installed (it's an optional extra). Each file does a top-level ``from inspect_ai.model
import …`` for the ``ModelAPI`` base + helper types — guarding that import per-file would
bury the dependency contract; one collect-time skip here keeps the providers' source clean
and lets ``make docstring-test`` run cleanly without the ``inspect`` extra."""
from __future__ import annotations

from importlib import util as _importlib_util

collect_ignore_glob = (
    []
    if _importlib_util.find_spec("inspect_ai") is not None
    else [
        "_provider_base.py",
        "eval.py",
        "sglang_provider.py",
        "transformers_provider.py",
        "vllm_provider.py",
    ]
)
