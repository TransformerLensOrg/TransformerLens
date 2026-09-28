"""Import-time warning hygiene after the legacy Hooked* stack was removed.

The per-class deprecation-warning tests are gone with the classes they covered
(4.0 deletion). What survives: importing the package must stay warning-clean,
and the KEPT HookedRootModule must NOT warn (it was never part of the removal).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import transformer_lens
from transformer_lens import _REMOVED_IN_4_0

PROJECT_ROOT = Path(__file__).parents[2]


def test_importing_transformer_lens_emits_no_deprecation_warning():
    code = "\n".join(
        [
            "import warnings",
            "warnings.filterwarnings(",
            "    'error',",
            "    category=DeprecationWarning,",
            "    module=r'^transformer_lens(?:\\.|$)',",
            ")",
            "import transformer_lens",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_hooked_root_module_is_not_deprecated():
    """HookedRootModule (with HookPoint) is KEPT infrastructure — the supported
    way to hook arbitrary nn.Modules — and must construct without any
    DeprecationWarning."""
    import warnings as w

    from transformer_lens import HookedRootModule

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        HookedRootModule()
    assert not [x for x in caught if issubclass(x.category, DeprecationWarning)]


@pytest.mark.parametrize("name", sorted(_REMOVED_IN_4_0))
def test_removed_name_carries_migration_pointer_via_attribute(name):
    with pytest.raises((AttributeError, ImportError), match="removed in TransformerLens 4.0"):
        getattr(transformer_lens, name)


@pytest.mark.parametrize("name", sorted(_REMOVED_IN_4_0))
def test_removed_name_carries_migration_pointer_via_from_import(name):
    """`from transformer_lens import X` is the spelling downstream packages use.
    CPython's from-import machinery discards an AttributeError from __getattr__ and
    raises its own unchained "cannot import name" ImportError, dropping the pointer;
    it only survives if __getattr__ raises ImportError. Both exception types are
    accepted so this pins the reach-the-pointer property, not the resolution."""
    with pytest.raises((AttributeError, ImportError), match="removed in TransformerLens 4.0"):
        exec(f"from transformer_lens import {name}")


def test_removed_hook_points_reexport_carries_pointer_via_attribute():
    import transformer_lens.hook_points as hook_points

    with pytest.raises((AttributeError, ImportError), match="removed in TransformerLens 4.0"):
        getattr(hook_points, "HookedRootModule")


def test_removed_hook_points_reexport_carries_pointer_via_from_import():
    with pytest.raises((AttributeError, ImportError), match="removed in TransformerLens 4.0"):
        exec("from transformer_lens.hook_points import HookedRootModule")
