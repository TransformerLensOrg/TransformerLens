"""SVDInterpreter on TransformerBridge.

The expected tensors below are frozen from a HookedTransformer/bridge
cross-check on distilgpt2 (verified equal at ~1e-7 before freezing) — they are
inline goldens for the legacy numerics.
"""

import jaxtyping
import pytest
import torch

from transformer_lens import SVDInterpreter
from transformer_lens.model_bridge import TransformerBridge

# Get TypeCheckError from jaxtyping module (it may be re-exported from typeguard)
TypeCheckError = getattr(jaxtyping, "TypeCheckError", None)
if TypeCheckError is None:
    # Fallback to typeguard
    from typeguard import TypeCheckError

MODEL = "distilgpt2"
VECTOR_TYPES = ["OV", "w_in", "w_out"]
ATOL = 2e-4  # Absolute tolerance for float comparisons

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
    bridge.enable_compatibility_mode()
    return bridge


@pytest.fixture(scope="module")
def unfolded_model():
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
    bridge.enable_compatibility_mode(fold_ln=False)
    return bridge


@pytest.fixture(scope="module")
def second_model():
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode()
    return bridge


expected_OV_match = torch.Tensor(
    [[[0.8836, 0.7932, 0.9862, 0.7931]], [[0.8590, 0.7715, 0.8447, 0.7476]]]
)

expected_w_in_match = torch.Tensor(
    [[[0.7469, 0.9686, 0.9909, 0.9245]], [[0.7416, 0.9593, 0.9858, 0.9175]]]
)

expected_w_in_unfolded_match = torch.Tensor(
    [[[0.8242, 0.9659, 0.8883, 0.7610]], [[0.7901, 0.9006, 0.8422, 0.7588]]]
)

expected_w_out_match = torch.Tensor(
    [[[0.3489, 0.3684, 0.1473, 0.3475]], [[0.3428, 0.3474, 0.1360, 0.3038]]]
)

# Successes


def test_svd_interpreter(model):
    svd_interpreter = SVDInterpreter(model)
    ov = svd_interpreter.get_singular_vectors(
        "OV", num_vectors=4, layer_index=0, head_index=0
    ).abs()
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", num_vectors=4, layer_index=0, head_index=0
    ).abs()
    w_out = svd_interpreter.get_singular_vectors(
        "w_out", num_vectors=4, layer_index=0, head_index=0
    ).abs()

    ov, w_in, w_out = (
        ov.topk(2, dim=0).values,
        w_in.topk(2, dim=0).values,
        w_out.topk(2, dim=0).values,
    )
    assert ov.shape == w_in.shape == w_out.shape == expected_OV_match.shape
    assert torch.allclose(ov.cpu(), expected_OV_match, atol=ATOL)
    assert torch.allclose(w_in.cpu(), expected_w_in_match, atol=ATOL)
    assert torch.allclose(w_out.cpu(), expected_w_out_match, atol=ATOL)


def test_w_in_when_fold_ln_is_false(unfolded_model):
    svd_interpreter = SVDInterpreter(unfolded_model)
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", num_vectors=4, layer_index=0, head_index=0
    ).abs()
    w_in = w_in.topk(2, dim=0).values
    assert torch.allclose(w_in.cpu(), expected_w_in_unfolded_match, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_layers(model):
    svd_interpreter = SVDInterpreter(model)
    ov = svd_interpreter.get_singular_vectors(
        "OV", layer_index=1, num_vectors=4, head_index=0
    ).abs()
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", layer_index=1, num_vectors=4, head_index=0
    ).abs()
    w_out = svd_interpreter.get_singular_vectors(
        "w_out", layer_index=1, num_vectors=4, head_index=0
    ).abs()

    ov, w_in, w_out = (
        ov.topk(2, dim=0).values,
        w_in.topk(2, dim=0).values,
        w_out.topk(2, dim=0).values,
    )
    assert ov.shape == w_in.shape == w_out.shape == expected_OV_match.shape
    assert not torch.allclose(ov.cpu(), expected_OV_match, atol=ATOL)
    assert not torch.allclose(w_in.cpu(), expected_w_in_match, atol=ATOL)
    assert not torch.allclose(w_out.cpu(), expected_w_out_match, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_models(second_model):
    svd_interpreter = SVDInterpreter(second_model)
    ov = svd_interpreter.get_singular_vectors(
        "OV", layer_index=1, num_vectors=4, head_index=0
    ).abs()
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", layer_index=1, num_vectors=4, head_index=0
    ).abs()
    w_out = svd_interpreter.get_singular_vectors(
        "w_out", layer_index=1, num_vectors=4, head_index=0
    ).abs()

    ov, w_in, w_out = (
        ov.topk(2, dim=0).values,
        w_in.topk(2, dim=0).values,
        w_out.topk(2, dim=0).values,
    )
    assert not torch.allclose(ov.cpu(), expected_OV_match, atol=ATOL)
    assert not torch.allclose(w_in.cpu(), expected_w_in_match, atol=ATOL)
    assert not torch.allclose(w_out.cpu(), expected_w_out_match, atol=ATOL)


# Failures


def test_svd_interpreter_fails_on_invalid_vector_type(model):
    svd_interpreter = SVDInterpreter(model)
    # Type checking can be done by jaxtyping (TypeCheckError) or beartype (BeartypeCallHintParamViolation)
    # Catch by checking the exception type name since jaxtyping may wrap typeguard's exception
    with pytest.raises(Exception) as exc_info:
        svd_interpreter.get_singular_vectors("test", layer_index=0, num_vectors=4, head_index=0)
    # Verify it's a type checking error (from jaxtyping, typeguard, or beartype)
    exc_name = type(exc_info.value).__name__
    assert "TypeCheckError" in exc_name or "Beartype" in exc_name
    assert "type-check" in str(exc_info.value).lower() or "vector_type" in str(exc_info.value)


def test_svd_interpreter_fails_on_not_passing_required_head_index(model):
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors("OV", layer_index=0, num_vectors=4)
    assert str(e.value) == "Head index optional only for w_in and w_out, got OV"


def test_svd_interpreter_fails_on_invalid_layer_index(model):
    svd_interpreter = SVDInterpreter(model)
    for vector in VECTOR_TYPES:
        with pytest.raises(AssertionError) as e:
            svd_interpreter.get_singular_vectors(vector, layer_index=6, num_vectors=4, head_index=0)
        assert str(e.value) == "Layer index must be between 0 and 5 but got 6"


def test_svd_interpreter_fails_on_invalid_head_index(model):
    # Only OV uses head index.
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors("OV", layer_index=0, num_vectors=4, head_index=12)
    assert str(e.value) == "Head index must be between 0 and 11 but got 12"
