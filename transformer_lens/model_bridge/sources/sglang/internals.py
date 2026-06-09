"""Single chokepoint for SGLang internal API access. SGLang's internals churn
(1-3 week releases); centralize every ``engine.…`` walk here so version drift
is patched in one place. Validated against ``sglang==0.5.12.post1``."""
from __future__ import annotations

from typing import Any

# Hard pin lower bound; release notes indicate API stabilization at 0.5.12.
_MIN_TESTED_VERSION = "0.5.12"


def assert_sglang_supported() -> None:
    """Fail fast at boot when sglang is missing or older than the validated pin."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("sglang")
    except PackageNotFoundError as e:
        raise RuntimeError(
            "sglang is not installed. See demos/SGLang_Bridge_Integration_Test.ipynb for install."
        ) from e
    # Coarse major.minor compare — catches the 0.4.x → 0.5.x split that moved
    # ModelRunner.initialize()'s hook-install point.
    installed_parts = tuple(int(p) for p in installed.split(".")[:2] if p.isdigit())
    min_parts = tuple(int(p) for p in _MIN_TESTED_VERSION.split(".")[:2])
    if installed_parts < min_parts:
        raise RuntimeError(f"sglang>={_MIN_TESTED_VERSION} required (installed: {installed}).")


def extract_hf_config(engine: Any) -> Any:
    """HF config the engine loaded; tries the documented path then a refactor fallback."""
    try:
        return engine.tokenizer_manager.model_config.hf_config
    except AttributeError as e:
        try:
            return engine.scheduler.tp_worker.model_runner.model_config.hf_config
        except AttributeError:
            raise RuntimeError(
                "Could not locate hf_config; update extract_hf_config() for this sglang version."
            ) from e


def model_runner_class() -> Any:
    """Resolve ``ModelRunner`` for the ``load_model`` monkey-patch. Lazy import."""
    from sglang.srt.model_executor.model_runner import (
        ModelRunner,  # type: ignore[import-not-found]
    )

    return ModelRunner


def rpc_classes() -> Any:
    """Resolve ``(RpcReqInput, RpcReqOutput)`` for cross-process method dispatch."""
    from sglang.srt.managers.io_struct import (  # type: ignore[import-not-found]
        RpcReqInput,
        RpcReqOutput,
    )

    return RpcReqInput, RpcReqOutput
