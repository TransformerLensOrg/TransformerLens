"""Exact synthetic checks of generation-aware embedding replacement."""

import pytest
import torch

from transformer_lens.tools.analysis._nla_artifacts import ArtifactRef, NLAPairConfig
from transformer_lens.tools.analysis._nla_verbalizer import (
    VerbalizerRuntime,
    embedding_overwrite,
    validate_activation_provenance,
)


@pytest.mark.parametrize("cached", [True, False])
def test_exact_prefill_locality(cached):
    vector = torch.tensor([3.0, 4.0])
    hook = embedding_overwrite(vector, 10.0, 1, 3, cached=cached)
    value = torch.arange(6.0).reshape(1, 3, 2)
    original = value.clone()
    result = hook(value, None)
    torch.testing.assert_close(result[0, 1], torch.tensor([6.0, 8.0]))
    torch.testing.assert_close(result[0, 0], original[0, 0])
    torch.testing.assert_close(result[0, 2], original[0, 2])
    torch.testing.assert_close(value, original)
    assert result.data_ptr() != value.data_ptr()


def test_cached_one_token_prefill_and_decode():
    hook = embedding_overwrite(torch.tensor([1.0, 0.0]), 5.0, 0, 1, cached=True)
    value = torch.ones(1, 1, 2)
    torch.testing.assert_close(hook(value, None)[0, 0], torch.tensor([5.0, 0.0]))
    assert hook(value, None) is value


def test_uncached_reapplies_at_original_position():
    hook = embedding_overwrite(torch.tensor([1.0, 0.0]), 5.0, 1, 2, cached=False)
    hook(torch.ones(1, 2, 2), None)
    value = torch.ones(1, 4, 2)
    result = hook(value, None)
    torch.testing.assert_close(result[0, 1], torch.tensor([5.0, 0.0]))
    torch.testing.assert_close(result[0, 2:], value[0, 2:])


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        embedding_overwrite(torch.ones(2), scale, 0, 1, cached=True)


def test_wrong_prefill_shape():
    hook = embedding_overwrite(torch.ones(2), 1.0, 0, 2, cached=True)
    with pytest.raises(ValueError, match="prefill"):
        hook(torch.ones(1, 1, 2), None)


def test_runtime_is_lazy_and_rejects_concurrent_operations():
    pair = NLAPairConfig(
        ArtifactRef("example/target", "a" * 40),
        ArtifactRef("example/av", "b" * 40),
        ArtifactRef("example/ar", "c" * 40),
        "qwen2",
        2,
        0,
        1,
    )
    runtime = VerbalizerRuntime(pair, "synthetic", local_files_only=True)
    assert runtime._model is None
    runtime.unload()
    assert runtime._lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            runtime.verbalize(torch.ones(2))
        with pytest.raises(RuntimeError, match="during"):
            runtime.unload()
    finally:
        runtime._lock.release()
    runtime.unload()


@pytest.mark.parametrize("source", ["pinned_load", "caller_attested"])
@pytest.mark.parametrize("mismatch", ["target", "revision", "layer"])
def test_structured_provenance_mismatches_rejected(source, mismatch):
    from dataclasses import replace

    from transformer_lens.tools.analysis._nla_contracts import (
        ActivationProvenance,
        NLAActivation,
        SourceProvenance,
    )

    pair = NLAPairConfig(
        ArtifactRef("example/target", "a" * 40),
        ArtifactRef("example/av", "b" * 40),
        ArtifactRef("example/ar", "c" * 40),
        "qwen2",
        2,
        1,
        3,
    )
    provenance = ActivationProvenance(
        "synthetic", SourceProvenance(source), pair.target, 1, 0, "tokens"
    )
    if mismatch == "target":
        provenance = replace(provenance, target=pair.ar)
    elif mismatch == "revision":
        provenance = replace(provenance, target=ArtifactRef(pair.target.repo_id, "d" * 40))
    else:
        provenance = replace(provenance, layer=0)
    activation = NLAActivation(torch.ones(2), provenance)
    with pytest.raises(ValueError, match="match"):
        validate_activation_provenance(activation, pair, "synthetic")
    runtime = VerbalizerRuntime(pair, "synthetic", local_files_only=True)
    with pytest.raises(ValueError, match="match"):
        runtime.verbalize(activation)
    assert runtime._model is None


def test_reference_normalization_operation_order():
    vector = torch.linspace(-1.0, 1.0, 3584)
    expected = vector / (vector.float().norm().clamp_min(1e-12) / 150).to(vector.dtype)
    actual = embedding_overwrite(vector, 150.0, 0, 1, cached=True)(torch.zeros(1, 1, 3584), None)[
        0, 0
    ]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_extreme_input_fallback_is_scale_safe(magnitude):
    vector = torch.tensor([magnitude, magnitude], dtype=torch.float64)
    result = embedding_overwrite(vector, 5.0, 0, 1, cached=True)(torch.zeros(1, 1, 2), None)[0, 0]
    assert torch.isfinite(result).all()
    assert result.norm().item() == pytest.approx(5.0)


def test_injection_representability():
    hook = embedding_overwrite(torch.ones(2), 1e10, 0, 1, cached=True)
    with pytest.raises(ValueError, match="representable"):
        hook(torch.ones(1, 1, 2, dtype=torch.float16), None)
