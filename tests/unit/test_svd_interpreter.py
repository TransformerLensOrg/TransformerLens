from transformer_lens import HookedTransformer, SVDInterpreter
import torch
import pytest

MODEL = "solu-2l"
VECTOR_TYPES = ["OV", "w_in", "w_out"]
ATOL = 1e-4  # Absolute tolerance - how far does a float have to be before we consider it no longer equal?
# ATOL is set to 1e-4 because the tensors we check on are also to 4 decimal places.
model = HookedTransformer.from_pretrained(MODEL)
second_model = HookedTransformer.from_pretrained("solu-3l")


expected_OV_match = torch.Tensor([[[0.5244, 0.5373, 0.6345, 0.7345]],
                                  [[0.5042, 0.5367, 0.5940, 0.7241]]]).to(device=model.cfg.device)

expected_w_in_match = torch.Tensor([[[0.7714, 0.6467, 0.6452, 0.6934]],
                                    [[0.7647, 0.6104, 0.6264, 0.6458]]]).to(device=model.cfg.device)

expected_w_out_match = torch.Tensor([[[0.9165, 0.6831, 0.8395, 0.6889]],
                                     [[0.7071, 0.5630, 0.7174, 0.6414]]]).to(device=model.cfg.device)

# Successes


def test_svd_interpreter():
    svd_interpreter = SVDInterpreter(model)
    ov = svd_interpreter.get_singular_vectors(
        'OV', num_vectors=4, layer_index=0, head_index=0)
    w_in = svd_interpreter.get_singular_vectors(
        'w_in', num_vectors=4, layer_index=0, head_index=0)
    w_out = svd_interpreter.get_singular_vectors(
        'w_out', num_vectors=4, layer_index=0, head_index=0)
    ov, w_in, w_out = ov.topk(2, dim=0).values, w_in.topk(
        2, dim=0).values, w_out.topk(2, dim=0).values
    assert ov.shape == w_in.shape == w_out.shape == expected_OV_match.shape
    assert torch.allclose(ov, expected_OV_match, atol=ATOL)
    assert torch.allclose(w_in, expected_w_in_match, atol=ATOL)
    assert torch.allclose(w_out, expected_w_out_match, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_layers():
    svd_interpreter = SVDInterpreter(model)
    ov = svd_interpreter.get_singular_vectors(
        'OV', layer_index=1, num_vectors=4, head_index=0)
    w_in = svd_interpreter.get_singular_vectors(
        'w_in', layer_index=1, num_vectors=4, head_index=0)
    w_out = svd_interpreter.get_singular_vectors(
        'w_out', layer_index=1, num_vectors=4, head_index=0)

    ov, w_in, w_out = ov.topk(2, dim=0).values, w_in.topk(
        2, dim=0).values, w_out.topk(2, dim=0).values
    assert ov.shape == w_in.shape == w_out.shape == expected_OV_match.shape
    assert not torch.allclose(ov, expected_OV_match, atol=ATOL)
    assert not torch.allclose(w_in, expected_w_in_match, atol=ATOL)
    assert not torch.allclose(w_out, expected_w_out_match, atol=ATOL)


def test_svd_interpreter_returns_different_answers_for_different_models():
    svd_interpreter = SVDInterpreter(second_model)
    ov_svd = svd_interpreter.get_singular_vectors(
        'OV', layer_index=1, num_vectors=4, head_index=0)
    w_in_svd = svd_interpreter.get_singular_vectors(
        'w_in', layer_index=1, num_vectors=4, head_index=0)
    w_out_svd = svd_interpreter.get_singular_vectors(
        'w_out', layer_index=1, num_vectors=4, head_index=0)

    assert ov_svd[:2].shape == w_in_svd[:2].shape == w_out_svd[:2].shape == expected_OV_match.shape
    assert not torch.allclose(ov_svd[:2], expected_OV_match, atol=ATOL)
    assert not torch.allclose(w_in_svd[:2], expected_w_in_match, atol=ATOL)
    assert not torch.allclose(w_out_svd[:2], expected_w_out_match, atol=ATOL)


# Failures


def test_svd_interpreter_passes_invalid_vector_type():
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(TypeError) as e:
        svd_interpreter.get_singular_vectors(
            'test', layer_index=0, num_vectors=4, head_index=0)
    assert str(
        e.value) == 'type of argument "vector_type" must be one of (Literal[OV], Literal[w_in], Literal[w_out]); got str instead'


def test_svd_interpreter_fails_on_invalid_layer_index():
    svd_interpreter = SVDInterpreter(model)
    for vector in VECTOR_TYPES:
        with pytest.raises(AssertionError) as e:
            svd_interpreter.get_singular_vectors(
                vector, layer_index=2, num_vectors=4, head_index=0)
        assert str(e.value) == "Layer index must be between 0 and 1 but got 2"


def test_svd_interpreter_fails_on_invalid_head_index():
    # Only OV uses head index.
    svd_interpreter = SVDInterpreter(model)
    with pytest.raises(AssertionError) as e:
        svd_interpreter.get_singular_vectors(
            'OV', layer_index=0, num_vectors=4, head_index=8)
    assert str(e.value) == "Head index must be between 0 and 7 but got 8"
