"""Intervention op vocabulary shared by the driver (spec validation) and the
worker extension (spec → buffer translation).

Adding a new op requires (a) listing it here, (b) handling it in
``worker_extension._apply_intervention``.
"""
from __future__ import annotations

SUPPORTED_OPS = frozenset({"suppress", "scale", "add", "set"})
