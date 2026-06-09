"""Single chokepoint for SGLang internal API access. Validated against ``sglang==0.5.12.post1``."""
from __future__ import annotations

from typing import Any

_MIN_TESTED_VERSION = "0.5.12"


def assert_sglang_supported() -> None:
    """Fail fast when sglang is missing or older than 0.5.12 — ``forward_hooks``,
    the plugin entry-point group, and the generic RPC dispatcher all landed
    there. A 0.5.x install < .12 would fail confusingly inside ``Engine(...)``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("sglang")
    except PackageNotFoundError as e:
        raise RuntimeError(
            "sglang is not installed. See demos/SGLang_Bridge_Integration_Test.ipynb for install."
        ) from e
    if _parse_version(installed) < _parse_version(_MIN_TESTED_VERSION):
        raise RuntimeError(f"sglang>={_MIN_TESTED_VERSION} required (installed: {installed}).")


def _parse_version(v: str) -> tuple[int, int, int]:
    """``"0.5.12.post1"`` / ``"0.5.12rc1"`` → ``(0, 5, 12)``. Tuple-compares safely."""
    parts: list[int] = []
    for chunk in v.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def extract_hf_config(engine: Any) -> Any:
    """HF config the engine loaded. ``tokenizer_manager`` lives on the driver
    process; ``engine.scheduler`` doesn't (it's in a subprocess)."""
    try:
        return engine.tokenizer_manager.model_config.hf_config
    except AttributeError as e:
        raise RuntimeError(
            "Could not locate hf_config under engine.tokenizer_manager.model_config; "
            "update extract_hf_config() for this sglang version."
        ) from e
