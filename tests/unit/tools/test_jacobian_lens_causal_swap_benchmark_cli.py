"""Unit tests for the causal-swap benchmark's artifact-generation CLI.

Model-free: these pin the double-execution fix and the argument parser, so no model is
loaded.
"""

import subprocess
import sys

import pytest

from transformer_lens.tools.analysis.jacobian_lens import DEFAULT_K
from transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark_cli import (
    _DEFAULT_ARTIFACT_PATH,
    _DEFAULT_CONTROL_SEEDS,
    _build_parser,
    _parse_seed_list,
)

_CLI_MODULE = "transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark_cli"


def test_cli_module_is_not_imported_by_the_package_init() -> None:
    # The analysis package's __init__ imports the library module, so pointing the documented
    # `python -m` command at that module executes it twice and runpy warns. The CLI module
    # must stay out of the package import graph for the documented command to run once.
    # Checked in a fresh interpreter: importing the CLI here would mask the regression.
    probe = (
        "import sys; import transformer_lens.tools.analysis; "
        f"raise SystemExit(1 if {_CLI_MODULE!r} in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, (
        "importing transformer_lens.tools.analysis pulled in the CLI module, which would "
        f"make `python -m {_CLI_MODULE}` execute twice; stderr: {result.stderr}"
    )


def test_cli_main_parses_defaults() -> None:
    args = _build_parser().parse_args([])
    assert args.output == _DEFAULT_ARTIFACT_PATH
    assert args.control_tolerance == 0.1
    assert args.control_seeds == _DEFAULT_CONTROL_SEEDS
    assert args.alpha == 1.0
    assert args.k == DEFAULT_K


def test_cli_main_parses_overrides() -> None:
    args = _build_parser().parse_args(
        [
            "--output",
            "out.json",
            "--control-tolerance",
            "0.25",
            "--control-seeds",
            "7,8",
            "--alpha",
            "0.5",
            "--k",
            "4",
        ]
    )
    assert args.output.name == "out.json"
    assert args.control_tolerance == 0.25
    assert args.control_seeds == (7, 8)
    assert args.alpha == 0.5
    assert args.k == 4


def test_parse_seed_list_accepts_comma_separated_seeds() -> None:
    assert _parse_seed_list("0,1,2") == (0, 1, 2)
    assert _parse_seed_list(" 3 , 4 ") == (3, 4)


def test_parse_seed_list_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one control seed"):
        _parse_seed_list(",")
