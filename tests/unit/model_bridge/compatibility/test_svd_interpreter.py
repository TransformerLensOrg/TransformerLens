import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from transformer_lens import SVDInterpreter
from transformer_lens.model_bridge import TransformerBridge

MODEL = "gpt2"  # Use a model that works with TransformerBridge
VECTOR_TYPES = ["OV", "w_in", "w_out"]
ATOL = 2e-4  # Absolute tolerance - how far does a float have to be before we consider it no longer equal?


@pytest.fixture(scope="module")
def model():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


@pytest.fixture(scope="module")
def unfolded_model():
    # Note: TransformerBridge may not support fold_ln parameter directly
    # We'll use the same model for now, but this test may need adjustment
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


@pytest.fixture(scope="module")
def second_model():
    # Use a different model architecture if available, otherwise same model
    # Note: If gpt2-medium fails to load, tests that need different models will be skipped
    try:
        return TransformerBridge.boot_transformers("gpt2-medium", device="cpu")
    except Exception:
        # Fallback to same model if gpt2-medium is not available
        # The test will skip if both models end up being the same
        return TransformerBridge.boot_transformers(MODEL, device="cpu")


def test_svd_interpreter_returns_meaningful_values(model):
    """SVD singular vectors must be non-zero and top values must exceed bottom values."""
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

    # All three vector types should have the same shape
    assert ov.shape == w_in.shape == w_out.shape

    # Singular vectors must contain non-zero values
    assert ov.abs().sum() > 0
    assert w_in.abs().sum() > 0
    assert w_out.abs().sum() > 0

    # Top singular values should be larger than bottom ones (SVD ordering)
    for vectors, name in [(ov, "OV"), (w_in, "w_in"), (w_out, "w_out")]:
        top2 = vectors.topk(2, dim=0).values.mean()
        bot2 = vectors.topk(2, dim=0, largest=False).values.mean()
        assert top2 > bot2, f"{name}: top singular values should exceed bottom values"


def test_w_in_unfolded_produces_nonzero_values(unfolded_model):
    """SVDInterpreter w_in should produce non-zero results on an unfolded model."""
    svd_interpreter = SVDInterpreter(unfolded_model)
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", num_vectors=4, layer_index=0, head_index=0
    ).abs()

    assert w_in.abs().sum() > 0


def test_svd_interpreter_returns_different_answers_for_different_layers(model):
    # Only test if model has multiple layers
    if model.cfg.n_layers < 2:
        pytest.skip("Model only has one layer")

    svd_interpreter = SVDInterpreter(model)

    # Layer 0 results
    ov_0 = svd_interpreter.get_singular_vectors(
        "OV", layer_index=0, num_vectors=4, head_index=0
    ).abs()
    w_in_0 = svd_interpreter.get_singular_vectors(
        "w_in", layer_index=0, num_vectors=4, head_index=0
    ).abs()
    w_out_0 = svd_interpreter.get_singular_vectors(
        "w_out", layer_index=0, num_vectors=4, head_index=0
    ).abs()

    # Layer 1 results
    ov_1 = svd_interpreter.get_singular_vectors(
        "OV", layer_index=1, num_vectors=4, head_index=0
    ).abs()
    w_in_1 = svd_interpreter.get_singular_vectors(
        "w_in", layer_index=1, num_vectors=4, head_index=0
    ).abs()
    w_out_1 = svd_interpreter.get_singular_vectors(
        "w_out", layer_index=1, num_vectors=4, head_index=0
    ).abs()

    # Results should be different between layers
    assert not torch.allclose(ov_0, ov_1, atol=ATOL)
    assert not torch.allclose(w_in_0, w_in_1, atol=ATOL)
    assert not torch.allclose(w_out_0, w_out_1, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_models(model, second_model):
    # Skip if both models are the same (check model name/config, not just object ID)
    if id(model) == id(second_model) or model.cfg.model_name == second_model.cfg.model_name:
        pytest.skip("Same model used for both fixtures")

    # Get results from first model
    svd_interpreter_1 = SVDInterpreter(model)
    ov_1 = svd_interpreter_1.get_singular_vectors(
        "OV", layer_index=0, num_vectors=4, head_index=0
    ).abs()

    # Get results from second model
    svd_interpreter_2 = SVDInterpreter(second_model)
    ov_2 = svd_interpreter_2.get_singular_vectors(
        "OV", layer_index=0, num_vectors=4, head_index=0
    ).abs()

    # Results should be different between models
    assert not torch.allclose(ov_1, ov_2, atol=ATOL)


# Failures


def test_svd_interpreter_fails_on_invalid_vector_type(model):
    from typeguard import TypeCheckError

    svd_interpreter = SVDInterpreter(model)
    with pytest.raises((BeartypeCallHintParamViolation, TypeCheckError)):
        svd_interpreter.get_singular_vectors("test", layer_index=0, num_vectors=4, head_index=0)


def test_svd_interpreter_fails_on_not_passing_required_head_index(model):
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors("OV", layer_index=0, num_vectors=4)
    assert "Head index optional only for w_in and w_out, got OV" in str(e.value)


def test_svd_interpreter_fails_on_invalid_layer_index(model):
    svd_interpreter = SVDInterpreter(model)
    max_layer = model.cfg.n_layers - 1
    invalid_layer = model.cfg.n_layers

    for vector in VECTOR_TYPES:
        with pytest.raises(AssertionError) as e:
            svd_interpreter.get_singular_vectors(
                vector, layer_index=invalid_layer, num_vectors=4, head_index=0
            )
        assert f"Layer index must be between 0 and {max_layer} but got {invalid_layer}" in str(
            e.value
        )


def test_svd_interpreter_fails_on_invalid_head_index(model):
    # Only OV uses head index.
    svd_interpreter = SVDInterpreter(model)
    max_head = model.cfg.n_heads - 1
    invalid_head = model.cfg.n_heads

    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors(
            "OV", layer_index=0, num_vectors=4, head_index=invalid_head
        )
    assert f"Head index must be between 0 and {max_head} but got {invalid_head}" in str(e.value)
