import pytest
import torch
import typeguard

from transformer_lens import HookedTransformer, SVDInterpreter

MODEL = "solu-2l"
VECTOR_TYPES = ["OV", "w_in", "w_out"]
ATOL = 1e-4  # Absolute tolerance - how far does a float have to be before we consider it no longer equal?
# ATOL is set to 1e-4 because the tensors we check on are also to 4 decimal places.
model = HookedTransformer.from_pretrained(MODEL)
unfolded_model = HookedTransformer.from_pretrained(MODEL, fold_ln=False)
second_model = HookedTransformer.from_pretrained("solu-3l")


expected_OV_match = torch.Tensor(
    [[[0.6597, 0.8689, 0.5669, 0.7345]], [[0.5232, 0.6705, 0.5623, 0.7240]]]
)

expected_w_in_match = torch.Tensor(
    [[[0.5572, 0.6466, 0.6406, 0.6094]], [[0.5417, 0.6103, 0.5773, 0.5726]]]
)

expected_w_in_unfolded_match = torch.Tensor(
    [[[0.2766, 0.3050, 0.3041, 0.3119]], [[0.2651, 0.2988, 0.2810, 0.2896]]]
)

expected_w_out_match = torch.Tensor(
    [[[0.5097, 0.4950, 0.5451, 0.7178]], [[0.5076, 0.4922, 0.5140, 0.7106]]]
)

# Successes


def test_svd_interpreter():
    svd_interpreter = SVDInterpreter(model)
    ov = svd_interpreter.get_singular_vectors(
        "OV", num_vectors=4, layer_index=0, head_index=0
    )
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", num_vectors=4, layer_index=0, head_index=0
    )
    w_out = svd_interpreter.get_singular_vectors(
        "w_out", num_vectors=4, layer_index=0, head_index=0
    )
    ov, w_in, w_out = (
        ov.topk(2, dim=0).values,
        w_in.topk(2, dim=0).values,
        w_out.topk(2, dim=0).values,
    )
    assert ov.shape == w_in.shape == w_out.shape == expected_OV_match.shape
    assert torch.allclose(ov, expected_OV_match, atol=ATOL)
    assert torch.allclose(w_in, expected_w_in_match, atol=ATOL)
    assert torch.allclose(w_out, expected_w_out_match, atol=ATOL)


def test_w_in_when_fold_ln_is_false():
    svd_interpreter = SVDInterpreter(unfolded_model)
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", num_vectors=4, layer_index=0, head_index=0
    )
    w_in = w_in.topk(2, dim=0).values
    assert torch.allclose(w_in, expected_w_in_unfolded_match, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_layers():
    svd_interpreter = SVDInterpreter(model)
    ov = svd_interpreter.get_singular_vectors(
        "OV", layer_index=1, num_vectors=4, head_index=0
    )
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", layer_index=1, num_vectors=4, head_index=0
    )
    w_out = svd_interpreter.get_singular_vectors(
        "w_out", layer_index=1, num_vectors=4, head_index=0
    )

    ov, w_in, w_out = (
        ov.topk(2, dim=0).values,
        w_in.topk(2, dim=0).values,
        w_out.topk(2, dim=0).values,
    )
    assert ov.shape == w_in.shape == w_out.shape == expected_OV_match.shape
    assert not torch.allclose(ov, expected_OV_match, atol=ATOL)
    assert not torch.allclose(w_in, expected_w_in_match, atol=ATOL)
    assert not torch.allclose(w_out, expected_w_out_match, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_models():
    svd_interpreter = SVDInterpreter(second_model)
    ov = svd_interpreter.get_singular_vectors(
        "OV", layer_index=1, num_vectors=4, head_index=0
    )
    w_in = svd_interpreter.get_singular_vectors(
        "w_in", layer_index=1, num_vectors=4, head_index=0
    )
    w_out = svd_interpreter.get_singular_vectors(
        "w_out", layer_index=1, num_vectors=4, head_index=0
    )

    ov, w_in, w_out = (
        ov.topk(2, dim=0).values,
        w_in.topk(2, dim=0).values,
        w_out.topk(2, dim=0).values,
    )
    assert not torch.allclose(ov, expected_OV_match, atol=ATOL)
    assert not torch.allclose(w_in, expected_w_in_match, atol=ATOL)
    assert not torch.allclose(w_out, expected_w_out_match, atol=ATOL)


# Failures


def test_svd_interpreter_fails_on_invalid_vector_type():
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(typeguard.TypeCheckError) as e:
        svd_interpreter.get_singular_vectors(
            "test", layer_index=0, num_vectors=4, head_index=0
        )
    assert 'argument "vector_type" (str) did not match any element in the union' in str(
        e.value
    )


def test_svd_interpreter_fails_on_not_passing_required_head_index():
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors("OV", layer_index=0, num_vectors=4)
        assert str(e.value) == "Head index optional only for w_in and w_out, got OV"


def test_svd_interpreter_fails_on_invalid_layer_index():
    svd_interpreter = SVDInterpreter(model)
    for vector in VECTOR_TYPES:
        with pytest.raises(AssertionError) as e:
            svd_interpreter.get_singular_vectors(
                vector, layer_index=2, num_vectors=4, head_index=0
            )
        assert str(e.value) == "Layer index must be between 0 and 1 but got 2"


def test_svd_interpreter_fails_on_invalid_head_index():
    # Only OV uses head index.
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors(
            "OV", layer_index=0, num_vectors=4, head_index=8
        )
    assert str(e.value) == "Head index must be between 0 and 7 but got 8"
