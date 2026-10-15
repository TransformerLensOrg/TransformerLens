"""Unit tests for the validators shared by the analysis tools."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transformer_lens.tools.analysis._model_state import (
    hybrid_block_types,
    require_int,
    to_token_ids,
)


@pytest.mark.parametrize("value", [True, False, 1.0, "3", None])
def test_require_int_refuses_bools_and_non_integers(value: object) -> None:
    with pytest.raises(ValueError, match="k must be an integer, got"):
        require_int(value, "k")


def test_require_int_accepts_inclusive_bounds_and_numpy_integers() -> None:
    assert require_int(1, "k") == 1
    assert require_int(3, "k", maximum=3) == 3
    result = require_int(np.int64(4), "k")
    assert result == 4 and type(result) is int


@pytest.mark.parametrize(
    ("kwargs", "value", "message"),
    [
        ({}, 0, "k must be a positive integer, got 0"),
        ({"minimum": 0}, -1, "k must be a nonnegative integer, got -1"),
        ({"minimum": 2}, 1, "k must be an integer at least 2, got 1"),
        ({"maximum": 3}, 4, "k must be between 1 and 3, got 4"),
    ],
)
def test_require_int_names_the_violated_range(
    kwargs: dict[str, int], value: int, message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        require_int(value, "k", **kwargs)


@pytest.mark.parametrize("value", [True, 0])
def test_require_int_message_replaces_both_failures(value: object) -> None:
    with pytest.raises(ValueError, match="^ks must hold integers$"):
        require_int(value, "k", message="ks must hold integers")


def test_hybrid_block_types_are_sorted_distinct_and_default_to_none() -> None:
    model = SimpleNamespace(
        layer_types=lambda: ["mamba+mlp", "attn+mlp", "linear_attn", "mamba+mlp"]
    )
    assert hybrid_block_types(model) == ["linear_attn", "mamba+mlp"]
    assert hybrid_block_types(SimpleNamespace()) == []


def _vocab_model(d_vocab: int = 10) -> SimpleNamespace:
    return SimpleNamespace(W_U=torch.zeros(4, d_vocab), to_single_token=len)


@pytest.mark.parametrize("token", [1.9, np.float64(2.0), torch.tensor(2.0), None])
def test_to_token_ids_refuses_non_integer_ids(token: object) -> None:
    with pytest.raises(TypeError, match="strings or integer token ids, got"):
        to_token_ids(_vocab_model(), [3, token])


def test_to_token_ids_accepts_every_integer_spelling() -> None:
    ids = to_token_ids(_vocab_model(), ["abc", np.int64(3), torch.tensor(3, dtype=torch.int32)])
    assert ids == [3, 3, 3] and all(type(token_id) is int for token_id in ids)


@pytest.mark.parametrize(
    "token", [3, np.int64(3), torch.tensor(3), torch.tensor(3, dtype=torch.uint8)]
)
def test_to_token_ids_resolves_a_bare_id_like_a_one_element_sequence(token: object) -> None:
    assert to_token_ids(_vocab_model(), token) == to_token_ids(_vocab_model(), [token]) == [3]


@pytest.mark.parametrize("token", [True, torch.tensor(2.0), torch.tensor(True)])
def test_to_token_ids_refuses_a_bare_non_integer_id(token: object) -> None:
    with pytest.raises((TypeError, ValueError), match="integer token id"):
        to_token_ids(_vocab_model(), token)
