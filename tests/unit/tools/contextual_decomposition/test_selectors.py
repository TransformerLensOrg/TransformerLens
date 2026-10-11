"""Tensor-only source/target validation and per-example projection oracles."""

import pytest
import torch
from jaxtyping import TypeCheckError

from transformer_lens.tools.analysis.contextual_decomposition.rules import (
    _CDConstituents,
)
from transformer_lens.tools.analysis.contextual_decomposition.selectors import (
    _CDSource,
    _CDTarget,
    _select_target,
    _validate_source_target,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer": -1},
        {"layer": True},
        {"layer": 0, "kind": "unknown"},
        {"layer": 0, "kind": "head", "head": 0},
        {"layer": 0, "head": 0},
        {"layer": 0, "positions": ()},
        {"layer": 0, "positions": (1, 1)},
        {"layer": 0, "positions": (-1,)},
        {"layer": 0, "positions": (True,)},
    ],
)
def test_invalid_sources(kwargs):
    with pytest.raises((ValueError, TypeCheckError)):
        _CDSource(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "internal"},
        {"kind": "internal", "hook_name": "blocks.0.hook_out", "token_id": 1},
        {"kind": "logit", "position": 1},
        {"kind": "logit", "position": 1, "token_id": 1, "negative_token_id": 2},
        {"kind": "logit_difference", "position": 1, "token_id": 1},
        {"kind": "logit", "position": -1, "token_id": 1},
        {"kind": "logit", "position": 1, "token_id": True},
    ],
)
def test_invalid_targets(kwargs):
    with pytest.raises(ValueError):
        _CDTarget(**kwargs)


def test_per_example_logit_projection_is_independent_and_copies_storage():
    beta = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    gamma = -beta * 0.5
    parts = _CDConstituents(beta, gamma)
    target = _CDTarget(
        "logit_difference", position=(0, 2), token_id=(3, 1), negative_token_id=(1, 0)
    )
    result = _select_target(parts, target)
    torch.testing.assert_close(result.beta, torch.tensor([2.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(result.gamma, torch.tensor([-1.0, -0.5], dtype=torch.float64))
    result.beta.zero_()
    assert beta[0, 0, 3] == 3


def test_source_position_and_target_head_ranges():
    args = dict(batch=2, sequence=3, n_layers=2, n_heads=2, vocab_size=4)
    with pytest.raises(ValueError, match="source position"):
        _validate_source_target(_CDSource(0, positions=(3,)), None, **args)
    with pytest.raises(ValueError, match="target head"):
        _validate_source_target(
            _CDSource(0),
            _CDTarget("internal", hook_name="blocks.1.attn.q.hook_out", head=2),
            **args,
        )
