from transformer_lens import HookedTransformer, SVDInterpreter
import torch
import pytest

MODEL = "solu-2l"
VECTOR_TYPES = ["OV", "w_in", "w_out"]
ATOL = 1e-4  # Absolute tolerance - how far does a float have to be before we consider it no longer equal?
# ATOL is set to 1e-4 because the tensors we check on are also to 4 decimal places.
model = HookedTransformer.from_pretrained(MODEL)
second_model = HookedTransformer.from_pretrained("solu-3l")


expected_OV_match = torch.Tensor([[[-0.0194,  0.0610,  0.0048,  0.0720]],
                                  [[0.0073, -0.0200,  0.0377,  0.0305]]])

expected_w_in_match = torch.Tensor([[[-0.0147, -0.0331, -0.0298,  0.1258]],
                                    [[0.0556,  0.0204, -0.0140,  0.1427]]])

expected_w_out_match = torch.Tensor([[[-0.0004, -0.1660, -0.0963,  0.1154]],
                                     [[-0.1311,  0.0775,  0.0413, -0.0657]]])

# Successes


def test_svd_interpreter():
    svd_interpreter = SVDInterpreter(model)
    ov_svd = svd_interpreter.get_top_singular_vectors(
        'OV', layer_index=0, num_vectors=4, head_index=0)
    w_in_svd = svd_interpreter.get_top_singular_vectors(
        'w_in', layer_index=0, num_vectors=4, head_index=0)
    w_out_svd = svd_interpreter.get_top_singular_vectors(
        'w_out', layer_index=0, num_vectors=4, head_index=0)

    assert torch.allclose(ov_svd, expected_OV_match, atol=ATOL)
    assert torch.allclose(w_in_svd, expected_w_in_match, atol=ATOL)
    assert torch.allclose(w_out_svd, expected_w_out_match, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_layers():
    svd_interpreter = SVDInterpreter(model)
    ov_svd = svd_interpreter.get_top_singular_vectors(
        'OV', layer_index=1, num_vectors=4, head_index=0)
    w_in_svd = svd_interpreter.get_top_singular_vectors(
        'w_in', layer_index=1, num_vectors=4, head_index=0)
    w_out_svd = svd_interpreter.get_top_singular_vectors(
        'w_out', layer_index=1, num_vectors=4, head_index=0)

    assert not torch.allclose(ov_svd, expected_OV_match, atol=ATOL)
    assert not torch.allclose(w_in_svd, expected_w_in_match, atol=ATOL)
    assert not torch.allclose(w_out_svd, expected_w_out_match, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_models():
    svd_interpreter = SVDInterpreter(second_model)
    ov_svd = svd_interpreter.get_top_singular_vectors(
        'OV', layer_index=1, num_vectors=4, head_index=0)
    w_in_svd = svd_interpreter.get_top_singular_vectors(
        'w_in', layer_index=1, num_vectors=4, head_index=0)
    w_out_svd = svd_interpreter.get_top_singular_vectors(
        'w_out', layer_index=1, num_vectors=4, head_index=0)

    assert not torch.allclose(ov_svd, expected_OV_match, atol=ATOL)
    assert not torch.allclose(w_in_svd, expected_w_in_match, atol=ATOL)
    assert not torch.allclose(w_out_svd, expected_w_out_match, atol=ATOL)


# Failures


def test_svd_interpreter_passes_invalid_vector_type():
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_top_singular_vectors(
            'test', layer_index=0, num_vectors=4, head_index=0)
    assert str(e.value) == "Head index optional only for w_in and w_out, got test"


def test_svd_interpreter_fails_on_invalid_layer_index():
    svd_interpreter = SVDInterpreter(model)
    for vector in VECTOR_TYPES:
        with pytest.raises(AssertionError) as e:
            svd_interpreter.get_top_singular_vectors(
                vector, layer_index=2, num_vectors=4, head_index=0)
        assert str(e.value) == "Layer index must be between 0 and 1 but got 2"


def test_svd_interpreter_fails_on_invalid_head_index():
    # Only OV uses head index.
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_top_singular_vectors(
            'OV', layer_index=0, num_vectors=4, head_index=8)
    assert str(e.value) == "Head index must be between 0 and 7 but got 8"
