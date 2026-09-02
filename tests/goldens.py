"""Loader for the HookedTransformer golden fixtures (the post-deletion oracle).

Goldens are captured by ``scripts/capture_ht_goldens.py`` while HookedTransformer
still exists, and hosted on a HF Hub dataset repo pinned by revision. Tests that
certify the compatibility-mode contract load their reference tensors from here
instead of constructing a live HookedTransformer.

Resolution order:
    1. ``TL_GOLDENS_DIR`` env var — a local capture directory (development,
       or CI jobs that regenerate goldens in-place).
    2. HF Hub ``snapshot_download`` of ``GOLDENS_REPO_ID`` at ``GOLDENS_REVISION``
       (cached by huggingface_hub; requires network on first use).

Tests should gate on ``goldens_available()`` (skip, don't fail, when neither
source is reachable) so the golden tier degrades like other network-dependent
tiers.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import torch

# Set once the dataset repo exists; pin to a specific commit revision so golden
# updates are explicit, reviewed events (never a moving branch).
GOLDENS_REPO_ID = "lars4776/TL-Goldens"
GOLDENS_REVISION: str | None = "130db21b365cbe286a5473c3267725733eefd085"

_ENV_VAR = "TL_GOLDENS_DIR"


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


@functools.lru_cache(maxsize=1)
def resolve_goldens_dir() -> Path | None:
    """Locate the goldens root, or None if unavailable."""
    local = os.environ.get(_ENV_VAR)
    if local:
        path = Path(local)
        return path if path.is_dir() else None
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=GOLDENS_REPO_ID,
                repo_type="dataset",
                revision=GOLDENS_REVISION,
            )
        )
    except Exception:
        return None


def goldens_available(model: str | None = None, config: str | None = None) -> bool:
    """True if the goldens root (and optionally a specific cell) is present."""
    root = resolve_goldens_dir()
    if root is None:
        return False
    if model is None:
        return True
    cell = root / _model_dir_name(model) / (config or "")
    return (cell / "provenance.json").exists() if config else cell.is_dir()


def golden_path(model: str, config: str) -> Path:
    """Directory of one (model, processing-config) golden cell. Raises if absent."""
    root = resolve_goldens_dir()
    if root is None:
        raise FileNotFoundError(
            f"Goldens unavailable: set {_ENV_VAR} to a local capture directory or "
            f"ensure {GOLDENS_REPO_ID!r} is reachable on the HF Hub."
        )
    cell = root / _model_dir_name(model) / config
    if not (cell / "provenance.json").exists():
        raise FileNotFoundError(f"No golden cell at {cell} (missing provenance.json).")
    return cell


def load_golden_tensors(model: str, config: str, name: str) -> dict[str, torch.Tensor]:
    """Load one safetensors artifact (e.g. 'views', 'activations', 'state_dict')."""
    from safetensors.torch import load_file

    return load_file(str(golden_path(model, config) / f"{name}.safetensors"))


def load_golden_json(model: str, config: str, name: str) -> Any:
    """Load one JSON artifact (e.g. 'hook_manifest', 'scalars', 'provenance')."""
    return json.loads((golden_path(model, config) / f"{name}.json").read_text())


class GoldenCell:
    """Lazy accessor for one (model, processing-config) golden cell.

    Conftest fixtures hand these to tests so each test loads only the
    artifacts it asserts on (activation snapshots are the big ones).
    """

    def __init__(self, model: str, config: str) -> None:
        self.model = model
        self.config = config
        self.path = golden_path(model, config)  # raises early if the cell is absent

    def has(self, name: str) -> bool:
        """Whether this cell carries the named safetensors group (older dataset
        revisions may predate a group's introduction)."""
        return (self.path / f"{name}.safetensors").is_file()

    def tensors(self, name: str) -> dict[str, torch.Tensor]:
        return load_golden_tensors(self.model, self.config, name)

    def json(self, name: str) -> Any:
        return load_golden_json(self.model, self.config, name)

    @property
    def hook_manifest(self) -> dict[str, Any]:
        return self.json("hook_manifest")

    @property
    def scalars(self) -> dict[str, Any]:
        return self.json("scalars")

    @property
    def provenance(self) -> dict[str, Any]:
        return self.json("provenance")
