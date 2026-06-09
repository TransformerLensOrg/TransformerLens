"""SGLang plugin entry point. Registered under
``[project.entry-points."sglang.srt.plugins"]``; ``load_plugins()`` fires this
in every spawned worker subprocess and in the driver process, so the Scheduler
class patch survives spawn without a monkey-patch in the parent."""
from __future__ import annotations


def register() -> None:
    """Bind the tl_* hook methods onto ``Scheduler``. Safe no-op if SGLang isn't
    importable. Idempotent — double-call doesn't double-wrap."""
    try:
        from sglang.srt.managers.scheduler import (  # type: ignore[import-not-found]
            Scheduler,
        )
    except ImportError:
        return

    if getattr(Scheduler, "_tl_methods_installed", False):
        return

    from . import hooks

    for method_name in hooks.SCHEDULER_METHODS:
        setattr(Scheduler, method_name, getattr(hooks, method_name))
    Scheduler._tl_methods_installed = True
