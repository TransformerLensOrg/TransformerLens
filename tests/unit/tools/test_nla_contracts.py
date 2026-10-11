"""Synthetic result, parsing, scoring, and length contract checks."""

import math

import pytest
import torch
from jaxtyping import TypeCheckError

from transformer_lens.tools.analysis._nla_artifacts import ArtifactRef
from transformer_lens.tools.analysis._nla_contracts import (
    ActivationProvenance,
    ExplanationStatus,
    NLAActivation,
    NLAExplanation,
    NLAReconstruction,
    NLASamplingConfig,
    SourceProvenance,
    detached_vector,
    parse_explanation,
    score_reconstruction,
    validate_context_length,
    validate_reconstruction_text,
)


@pytest.fixture
def provenance():
    return ActivationProvenance("synthetic", SourceProvenance.UNKNOWN_VECTOR)


@pytest.fixture
def artifact():
    return ArtifactRef("example/ar", "a" * 40)


def test_loading_evidence_is_not_live_weight_integrity(artifact):
    value = ActivationProvenance(
        "synthetic", SourceProvenance.PINNED_LOAD, artifact, 20, 0, "raw tokens, no BOS"
    )
    assert value.source_verified is True
    assert (
        ActivationProvenance("synthetic", SourceProvenance.CALLER_ATTESTED).source_verified is False
    )
    with pytest.raises(ValueError, match="requires"):
        ActivationProvenance("synthetic", SourceProvenance.PINNED_LOAD)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"artifact_name": ""},
        {"source": "pinned_load"},
        {"position": -1},
        {"layer": True},
        {"basis": "processed"},
        {"tokenization_policy": ""},
    ],
)
def test_invalid_provenance(kwargs):
    args = {"artifact_name": "synthetic", "source": SourceProvenance.UNKNOWN_VECTOR}
    args.update(kwargs)
    with pytest.raises((ValueError, TypeCheckError)):
        ActivationProvenance(**args)


@pytest.mark.parametrize("kind", ["activation", "reconstruction"])
def test_result_detaches_and_owns_storage(kind, provenance, artifact):
    source = torch.arange(1.0, 5.0, requires_grad=True)
    vector = source * 2
    result = (
        NLAActivation(vector, provenance)
        if kind == "activation"
        else NLAReconstruction(vector, artifact)
    )
    assert result.vector.requires_grad is False and result.vector.grad_fn is None
    assert result.vector.data_ptr() != vector.data_ptr()
    vector.detach().fill_(99)
    torch.testing.assert_close(result.vector, torch.tensor([2.0, 4.0, 6.0, 8.0]))


@pytest.mark.parametrize(
    "vector",
    [
        torch.empty(0),
        torch.ones(2, 2),
        torch.zeros(3),
        torch.ones(2, dtype=torch.int64),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([1j]),
    ],
)
def test_invalid_vectors(vector, provenance, artifact):
    with pytest.raises(ValueError):
        detached_vector(vector)
    with pytest.raises(ValueError):
        NLAActivation(vector, provenance)
    with pytest.raises(ValueError):
        NLAReconstruction(vector, artifact)


@pytest.mark.parametrize(
    "raw,status,text",
    [
        (
            "prefix <explanation>  first\nsecond  </explanation> suffix",
            ExplanationStatus.SUCCESS,
            "first\nsecond",
        ),
        ("no tags", ExplanationStatus.MISSING, None),
        ("<explanation></explanation>", ExplanationStatus.EMPTY, None),
        ("<explanation> \n </explanation>", ExplanationStatus.EMPTY, None),
        ("<explanation>x", ExplanationStatus.INCOMPLETE, None),
        ("x</explanation>", ExplanationStatus.INCOMPLETE, None),
        ("</explanation><explanation>x</explanation>", ExplanationStatus.INCOMPLETE, None),
        (
            "<explanation>a</explanation><explanation>b</explanation>",
            ExplanationStatus.MULTIPLE,
            None,
        ),
        (
            "<explanation>a<explanation>b</explanation></explanation>",
            ExplanationStatus.NESTED,
            None,
        ),
        ("<Explanation>x</Explanation>", ExplanationStatus.MALFORMED, None),
        ("<explanation attr='x'>x</explanation>", ExplanationStatus.MALFORMED, None),
        ("<explanation x", ExplanationStatus.MALFORMED, None),
        ("<explanation>a</explanation><explanation>", ExplanationStatus.INCOMPLETE, None),
    ],
)
def test_parser(raw, status, text, provenance):
    result = parse_explanation(raw, (1, 2), provenance)
    assert result.raw_text == raw and result.token_ids == (1, 2)
    assert result.parse_status is status and result.text == text
    assert result.provenance is provenance


def test_reconstruction_preserves_authored_whitespace():
    text = "  authored\n  explanation  "
    assert validate_reconstruction_text(text) == text
    assert validate_reconstruction_text("An untagged full response cannot be detected semantically")
    assert validate_reconstruction_text("<text>an ordinary tag</text>")


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        " \n",
        "<explanation>x</explanation>",
        "<explanation",
        "</ explanation>",
        "<EXPLANATION >x",
    ],
)
def test_invalid_reconstruction_text(text):
    with pytest.raises(ValueError):
        validate_reconstruction_text(text)


def test_explanation_result_invariants(provenance):
    with pytest.raises(ValueError):
        NLAExplanation("text", "raw", (), ExplanationStatus.MISSING, provenance)
    with pytest.raises(ValueError):
        NLAExplanation(None, "raw", (), ExplanationStatus.SUCCESS, provenance)
    with pytest.raises(ValueError):
        NLAExplanation(" text ", "raw", (), ExplanationStatus.SUCCESS, provenance)
    with pytest.raises(ValueError):
        parse_explanation("<explanation>x</explanation>", (-1,), provenance)


@pytest.mark.parametrize(
    "b,cosine,mse", [([1.0, 0.0], 1.0, 0.0), ([0.0, 1.0], 0.0, 2.0), ([-1.0, 0.0], -1.0, 4.0)]
)
def test_metric_identities(b, cosine, mse, provenance, artifact):
    result = score_reconstruction(
        NLAActivation(torch.tensor([2.0, 0.0]), provenance),
        NLAReconstruction(torch.tensor(b), artifact),
    )
    assert result.cosine == pytest.approx(cosine)
    assert result.normalized_mse == pytest.approx(mse)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_metric_matches_normalized_coordinate_mse(dtype):
    a = torch.tensor([1.0, 2.0, -3.0, 4.0], dtype=dtype)
    b = torch.tensor([-2.0, 3.0, 4.0, 1.0], dtype=dtype)
    result = score_reconstruction(a, b)
    a64, b64 = a.double(), b.double()
    scale = math.sqrt(a.numel())
    expected = ((scale * a64 / a64.norm() - scale * b64 / b64.norm()) ** 2).mean().item()
    assert result.normalized_mse == pytest.approx(expected, abs=1e-14)
    assert 0 <= result.normalized_mse <= 4


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_finite_extremes_do_not_overflow_norms(scale):
    a = torch.tensor([scale, scale], dtype=torch.float64)
    result = score_reconstruction(a, a)
    assert result.cosine == pytest.approx(1.0)
    assert result.normalized_mse == pytest.approx(0.0, abs=1e-14)


def test_metric_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimensions"):
        score_reconstruction(torch.ones(2), torch.ones(3))


@pytest.mark.parametrize(
    "vector", [torch.zeros(2), torch.tensor([float("nan")]), torch.tensor([float("inf")])]
)
def test_metric_rejects_invalid_input(vector):
    with pytest.raises(ValueError):
        score_reconstruction(vector, torch.ones(2))
    with pytest.raises(ValueError):
        score_reconstruction(torch.ones(2), vector)


def test_sampling_defaults_and_greedy():
    defaults = NLASamplingConfig()
    assert defaults.max_new_tokens == 512 and defaults.temperature == 1
    assert defaults.do_sample is True and defaults.use_past_kv_cache is True
    assert NLASamplingConfig(do_sample=False, temperature=0).do_sample is False
    assert NLASamplingConfig(use_past_kv_cache=False).use_past_kv_cache is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": -1},
        {"max_new_tokens": True},
        {"max_new_tokens": 1.5},
        {"temperature": 0},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": True},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_p": float("nan")},
        {"top_p": True},
        {"top_k": -1},
        {"top_k": True},
        {"do_sample": 1},
        {"use_past_kv_cache": 0},
        {"do_sample": False, "top_p": 0.9},
        {"do_sample": False, "top_k": 10},
        {"temperature": 10**400},
    ],
)
def test_invalid_sampling(kwargs):
    with pytest.raises((ValueError, TypeCheckError)):
        NLASamplingConfig(**kwargs)


def test_context_exact_boundaries():
    validate_context_length(4, 4)
    validate_context_length(1, 4, 3)
    sampling = NLASamplingConfig(max_new_tokens=3)
    sampling.validate_prompt(1, 4)
    with pytest.raises(ValueError, match="exceeds"):
        sampling.validate_prompt(2, 4)


@pytest.mark.parametrize(
    "args",
    [
        (0, 4, 0),
        (1, 0, 0),
        (True, 4, 0),
        (1, False, 0),
        (1, 4, -1),
        (1, 4, True),
        (5, 4, 0),
        (1, 4, 4),
    ],
)
def test_invalid_lengths(args):
    with pytest.raises(ValueError):
        validate_context_length(*args)
