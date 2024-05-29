import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from transformer_lens import HookedTransformer, SVDInterpreter

MODEL = "solu-2l"
VECTOR_TYPES = ["OV", "w_in", "w_out"]
ATOL = 2e-4  # Absolute tolerance - how far does a float have to be before we consider it no longer equal?


@pytest.fixture(scope="module")
def model():
    return HookedTransformer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def unfolded_model():
    return HookedTransformer.from_pretrained(MODEL, fold_ln=False)


@pytest.fixture(scope="module")
def second_model():
    return HookedTransformer.from_pretrained("solu-3l")


expected_OV_match = torch.Tensor(
    [[[0.6597, 0.8689, 0.6344, 0.7345]], [[0.5244, 0.6705, 0.5940, 0.7240]]]
)

expected_w_in_match = torch.Tensor(
    [[[0.7714, 0.6608, 0.6452, 0.6933]], [[0.7647, 0.6466, 0.6406, 0.6458]]]
)

expected_w_in_unfolded_match = torch.Tensor(
    [[[0.3639, 0.3164, 0.3095, 0.3430]], [[0.3614, 0.3050, 0.3041, 0.3140]]]
)

expected_w_out_match = torch.Tensor(
    [[[0.5097, 0.5389, 0.7906, 0.7178]], [[0.5076, 0.5350, 0.7674, 0.7106]]]
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
    with pytest.raises(BeartypeCallHintParamViolation) as e:
        svd_interpreter.get_singular_vectors("test", layer_index=0, num_vectors=4, head_index=0)


def test_svd_interpreter_fails_on_not_passing_required_head_index(model):
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors("OV", layer_index=0, num_vectors=4)
        assert str(e.value) == "Head index optional only for w_in and w_out, got OV"


def test_svd_interpreter_fails_on_invalid_layer_index(model):
    svd_interpreter = SVDInterpreter(model)
    for vector in VECTOR_TYPES:
        with pytest.raises(AssertionError) as e:
            svd_interpreter.get_singular_vectors(vector, layer_index=2, num_vectors=4, head_index=0)
        assert str(e.value) == "Layer index must be between 0 and 1 but got 2"


def test_svd_interpreter_fails_on_invalid_head_index(model):
    # Only OV uses head index.
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors("OV", layer_index=0, num_vectors=4, head_index=8)
    assert str(e.value) == "Head index must be between 0 and 7 but got 8"
