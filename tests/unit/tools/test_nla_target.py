"""Local provenance and token-preparation checks without model loading."""

from types import SimpleNamespace

import pytest
import torch

from transformer_lens.tools.analysis._nla_artifacts import ArtifactRef, NLAPairConfig
from transformer_lens.tools.analysis._nla_contracts import SourceProvenance
from transformer_lens.tools.analysis._nla_target import (
    prepare_tokens,
    target_provenance,
)


@pytest.fixture
def pair():
    return NLAPairConfig(
        ArtifactRef("example/target", "a" * 40),
        ArtifactRef("example/av", "b" * 40),
        ArtifactRef("example/ar", "c" * 40),
        "qwen2",
        4,
        1,
        3,
    )


@pytest.fixture
def model():
    return SimpleNamespace(
        original_model=SimpleNamespace(config=SimpleNamespace(vocab_size=10)),
        cfg=SimpleNamespace(n_ctx=4, device="cpu", default_prepend_bos=True),
        tokenizer=SimpleNamespace(pad_token_id=9, eos_token_id=8),
    )


def test_matching_external_loading_evidence(model, pair):
    config = model.original_model.config
    config._name_or_path = pair.target.repo_id
    config._commit_hash = pair.target.revision
    assert target_provenance(model, pair) is SourceProvenance.PINNED_LOAD


def test_owned_loading_record(model, pair):
    model._nla_target_artifact = pair.target
    assert target_provenance(model, pair) is SourceProvenance.PINNED_LOAD


def test_unknown_provenance_requires_attestation(model, pair):
    with pytest.raises(ValueError, match="unknown"):
        target_provenance(model, pair)
    with pytest.warns(UserWarning, match="unverified"):
        assert (
            target_provenance(model, pair, allow_unverified_target=True)
            is SourceProvenance.CALLER_ATTESTED
        )


@pytest.mark.parametrize(
    "field,value", [("_name_or_path", "example/wrong"), ("_commit_hash", "d" * 40)]
)
def test_attestation_cannot_bypass_known_mismatch(model, pair, field, value):
    setattr(model.original_model.config, field, value)
    with pytest.raises(ValueError, match="does not match"):
        target_provenance(model, pair, allow_unverified_target=True)


def test_local_unknown_identity_can_be_attested(model, pair, tmp_path):
    model.original_model.config._name_or_path = str(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        target_provenance(model, pair)
    with pytest.warns(UserWarning, match="unverified"):
        assert (
            target_provenance(model, pair, allow_unverified_target=True)
            is SourceProvenance.CALLER_ATTESTED
        )


def test_recorded_identity_mismatch(model, pair):
    model._nla_target_artifact = pair.ar
    with pytest.raises(ValueError, match="recorded"):
        target_provenance(model, pair)


@pytest.mark.parametrize(
    "tokens", [torch.tensor([1, 2]), torch.tensor([[1, 2]], dtype=torch.int32)]
)
def test_explicit_tokens(model, tokens):
    result, policy = prepare_tokens(model, tokens)
    assert result.shape == (1, 2) and result.dtype == torch.int64
    assert "caller" in policy


@pytest.mark.parametrize(
    "tokens",
    [
        torch.tensor([]),
        torch.ones(2, 2, dtype=torch.int64),
        torch.ones(1, 1, 2, dtype=torch.int64),
        torch.tensor([-1]),
        torch.tensor([10]),
        torch.tensor([9]),
        torch.ones(5, dtype=torch.int64),
    ],
)
def test_invalid_tokens(model, tokens):
    with pytest.raises(ValueError):
        prepare_tokens(model, tokens)


def test_explicit_bos_override_rejected(model):
    with pytest.raises(ValueError, match="explicit"):
        prepare_tokens(model, torch.tensor([1]), prepend_bos=True)


def test_bos_policy_no_truncation_and_single_tokenization(model):
    calls = []

    def tokenize(text, *, prepend_bos, truncate):
        calls.append((text, prepend_bos, truncate))
        return torch.tensor([[1, 2]])

    model.to_tokens = tokenize
    tokens, policy = prepare_tokens(model, "hello", prepend_bos=False)
    assert calls == [("hello", False, False)]
    assert "prepend_bos=False" in policy and tokens.tolist() == [[1, 2]]


def test_empty_prompt(model):
    with pytest.raises(ValueError, match="nonempty"):
        prepare_tokens(model, " ")
