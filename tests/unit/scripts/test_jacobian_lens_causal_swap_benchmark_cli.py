"""Model-free tests for the causal-swap benchmark script's argument parser."""

import pytest

from scripts.jacobian_lens_causal_swap_benchmark import (
    _DEFAULT_ARTIFACT_PATH,
    _DEFAULT_CONTROL_SEEDS,
    _build_parser,
    _parse_seed_list,
)
from transformer_lens.tools.analysis.jacobian_lens import DEFAULT_K


def test_cli_main_parses_defaults() -> None:
    args = _build_parser().parse_args([])
    assert args.output == _DEFAULT_ARTIFACT_PATH
    # A bare run must regenerate the frozen artifact the demo notebook loads.
    assert _DEFAULT_ARTIFACT_PATH.is_file()
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
