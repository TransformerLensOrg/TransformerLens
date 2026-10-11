"""Exact synthetic checks of generation-aware embedding replacement."""

import pytest
import torch

from transformer_lens.tools.analysis._nla_artifacts import ArtifactRef, NLAPairConfig
from transformer_lens.tools.analysis._nla_verbalizer import (
    VerbalizerRuntime,
    embedding_overwrite,
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


def test_injection_representability():
    hook = embedding_overwrite(torch.ones(2), 1e10, 0, 1, cached=True)
    with pytest.raises(ValueError, match="representable"):
        hook(torch.ones(1, 1, 2, dtype=torch.float16), None)
