"""Inspect driver: turn an ``inspect_ai``-served model into a TransformerLens bridge.

We ship our own HF-backed provider (``provider.py``) and a torch-free, provider-
agnostic consumer (``driver.py``) wire-compatible with vllm-lens.

Maintainer note: this package is named ``inspect``, shadowing the stdlib module.
Only ever import the stdlib ``inspect`` via an absolute import from outside this
package; never write a bare ``import inspect`` inside these modules.
"""
from __future__ import annotations

from .source import boot_inspect

__all__ = ["boot_inspect"]
