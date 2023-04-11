from transformer_lens import HookedTransformer, head_detector
import torch
import pytest
import math

MODEL = "solu-2l"
ATOL = 1e-4 # Absolute tolerance - how far does a float have to be before we consider it no longer equal?
# ATOL is set to 1e-4 because the tensors we check on are also to 4 decimal places.

model = HookedTransformer.from_pretrained(MODEL)
test_regular_sequence = " four token sequence" # Four tokens including BOS
test_duplicated_sequence = " seven token sequence seven token sequence"

expected_regular_sequence_previous_match = torch.tensor([
    [0.3567, 0.2326, 0.2587, 0.2669, 0.1437, 0.5924, 0.2706, 0.2647],
    [0.3250, 0.2487, 0.2685, 0.2771, 0.2737, 0.2520, 0.2523, 0.3359]])

expected_duplicated_sequence_previous_match = torch.tensor([
    [0.2978, 0.1737, 0.1593, 0.1686, 0.0905, 0.6462, 0.1885, 0.1767],
    [0.2923, 0.2045, 0.1845, 0.2083, 0.1797, 0.1529, 0.1564, 0.2445]])

expected_duplicated_sequence_duplicate_match = torch.tensor([
    [0.0904, 0.0944, 0.0010, 0.0155, 0.2024, 0.0071, 0.0164, 0.0715],
    [0.0381, 0.0038, 0.0309, 0.0184, 0.0322, 0.0103, 0.0066, 0.0446]])

expected_duplicated_sequence_induction_match = torch.tensor([
    [0.1242, 0.0539, 0.0109, 0.0178, 0.0005, 0.0560, 0.0312, 0.0521],
    [0.0659, 0.1994, 0.0430, 0.0289, 0.0470, 0.0119, 0.1726, 0.0665]])

expected_previous_exclude_bos_match = torch.tensor([
    [0.4312, 0.1414, 0.4195, 0.3316, 0.0016, 0.7672, 0.4385, 0.2628],
    [0.4030, 0.1467, 0.3050, 0.3247, 0.3062, 0.2421, 0.2043, 0.3593]])

expected_previous_exclude_current_token_match = torch.tensor([
    [0.5441, 0.4149, 0.3545, 0.3771, 0.2738, 0.8821, 0.3797, 0.3835],
    [0.6770, 0.3445, 0.3959, 0.4228, 0.4032, 0.3449, 0.3434, 0.5770]])

expected_previous_exclude_bos_and_current_token_match = torch.tensor([
    [0.6092, 0.5601, 0.8043, 0.8732, 0.1130, 0.9122, 0.6857, 0.4405],
    [0.7011, 0.7523, 0.5545, 0.6449, 0.7958, 0.7565, 0.7082, 0.7833]])


# Successes
def test_detect_head():
    assert torch.allclose(head_detector.detect_head(model, test_regular_sequence, head_name="previous_token_head"), expected_regular_sequence_previous_match, atol=ATOL)
    assert torch.allclose(head_detector.detect_head(model, test_regular_sequence, head_name="duplicate_token_head"), torch.zeros(model.cfg.n_layers, model.cfg.n_heads), atol=ATOL)
    assert torch.allclose(head_detector.detect_head(model, test_regular_sequence, head_name="induction_head"), torch.zeros(model.cfg.n_layers, model.cfg.n_heads), atol=ATOL)

    assert torch.allclose(head_detector.detect_head(model, test_duplicated_sequence, head_name="previous_token_head"), expected_duplicated_sequence_previous_match, atol=ATOL)
    assert torch.allclose(head_detector.detect_head(model, test_duplicated_sequence, head_name="duplicate_token_head"), expected_duplicated_sequence_duplicate_match, atol=ATOL)
    assert torch.allclose(head_detector.detect_head(model, test_duplicated_sequence, head_name="induction_head"), expected_duplicated_sequence_induction_match, atol=ATOL)
    
def test_detect_head_exclude_bos():
    assert torch.allclose(head_detector.detect_head(model, test_regular_sequence, head_name="previous_token_head", exclude_bos=True),expected_previous_exclude_bos_match, atol=ATOL)
    
def test_detect_head_exclude_current_token():
    assert torch.allclose(head_detector.detect_head(model, test_regular_sequence, head_name="previous_token_head", exclude_current_token=True),expected_previous_exclude_current_token_match, atol=ATOL)
    
def test_detect_head_exclude_bos_and_current_token():
    assert torch.allclose(head_detector.detect_head(model, test_regular_sequence, head_name="previous_token_head", exclude_bos=True, exclude_current_token=True), expected_previous_exclude_bos_and_current_token_match, atol=ATOL)
    
def test_detect_head_with_cache():
    _, cache = model.run_with_cache(test_regular_sequence, remove_batch_dim=True)
    assert torch.allclose(head_detector.detect_head(model, test_regular_sequence, head_name="previous_token_head", cache=cache),expected_regular_sequence_previous_match, atol=ATOL)

# Errors
def test_detect_head_with_neither_head_name_nor_detection_pattern():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, test_regular_sequence)
    assert str(e.value) == "Exactly one of head_name or detection_pattern must be specified."

def test_detect_head_with_both_head_name_and_detection_pattern():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, test_regular_sequence, head_name="test", detection_pattern=torch.ones(4, 4))
    assert str(e.value) == "Exactly one of head_name or detection_pattern must be specified."

def test_detect_head_with_invalid_head_name():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, test_regular_sequence, head_name="test")
    assert str(e.value) == "Head name not valid."

def test_detect_head_with_zero_sequence_length():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, "", head_name="previous_token_head")
    assert str(e.value) == "The sequence must be non-empty and must fit within the model's context window."

def test_detect_head_with_sequence_length_outside_context_window():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, "a " * model.cfg.n_ctx, head_name="previous_token_head")
    assert str(e.value) == "The sequence must be non-empty and must fit within the model's context window."

def test_detect_head_with_invalid_detection_pattern():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, test_duplicated_sequence, detection_pattern=torch.ones(4, 4))
    assert str(e.value) == "The detection pattern must be a square matrix of shape (sequence_length, sequence_length)."

def test_detect_head_with_specific_heads():
    match = head_detector.detect_head(model, test_regular_sequence, head_name="previous_token_head", specific_heads=[(0, 0)])
    assert torch.allclose(match[0, 0], expected_regular_sequence_previous_match[0, 0], atol=ATOL)
    assert math.isclose(torch.sum(match), match[0, 0].item() - (model.cfg.n_layers * model.cfg.n_heads - 1), abs_tol=ATOL) 

    match = head_detector.detect_head(model, test_duplicated_sequence, head_name="previous_token_head", specific_heads=[(0, 0)])
    assert torch.allclose(match[0, 0], expected_duplicated_sequence_previous_match[0, 0], atol=ATOL)
    assert math.isclose(torch.sum(match), match[0, 0].item() - (model.cfg.n_layers * model.cfg.n_heads - 1), abs_tol=ATOL)

    match = head_detector.detect_head(model, test_duplicated_sequence, head_name="duplicate_token_head", specific_heads=[(0, 0)])
    assert torch.allclose(match[0, 0], expected_duplicated_sequence_duplicate_match[0, 0], atol=ATOL)
    assert math.isclose(torch.sum(match), match[0, 0].item() - (model.cfg.n_layers * model.cfg.n_heads - 1), abs_tol=ATOL)

    match = head_detector.detect_head(model, test_duplicated_sequence, head_name="induction_head", specific_heads=[(0, 0)])
    assert torch.allclose(match[0, 0], expected_duplicated_sequence_induction_match[0, 0], atol=ATOL)
    assert math.isclose(torch.sum(match), match[0, 0].item() - (model.cfg.n_layers * model.cfg.n_heads - 1), abs_tol=ATOL)


# Detection patterns
def test_previous_token_head_detection():
    regular_detection_pattern = head_detector.get_previous_token_head_detection_pattern(model, test_regular_sequence)
    assert regular_detection_pattern.shape == (4, 4)
    assert (regular_detection_pattern[1: , :-1] == torch.eye(3)).all()
    assert torch.sum(regular_detection_pattern) == 3

    duplicate_detection_pattern = head_detector.get_previous_token_head_detection_pattern(model, test_duplicated_sequence)
    assert duplicate_detection_pattern.shape == (7, 7)
    assert (duplicate_detection_pattern[1: , :-1] == torch.eye(6)).all()
    assert torch.sum(duplicate_detection_pattern) == 6

def test_duplicate_token_head_detection():
    assert (head_detector.get_duplicate_token_head_detection_pattern(model, test_regular_sequence) == torch.zeros(4, 4)).all()
    detection_pattern = head_detector.get_duplicate_token_head_detection_pattern(model, test_duplicated_sequence)
    assert detection_pattern.shape == (7, 7)
    assert (detection_pattern[4: , 1:4] == torch.eye(3)).all()
    assert torch.sum(detection_pattern) == 3

def test_induction_head_detection():
    assert (head_detector.get_duplicate_token_head_detection_pattern(model, test_regular_sequence) == torch.zeros(4, 4)).all()
    detection_pattern = head_detector.get_induction_head_detection_pattern(model, test_duplicated_sequence)
    assert detection_pattern.shape == (7, 7)
    assert (detection_pattern[4: , 2:5] == torch.eye(3)).all()
    assert torch.sum(detection_pattern) == 3