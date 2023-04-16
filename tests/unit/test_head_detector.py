import math

import pytest
import torch

from transformer_lens import HookedTransformer, head_detector

MODEL = "solu-2l"
ATOL = 1e-4  # Absolute tolerance - how far does a float have to be before we consider it no longer equal?
# ATOL is set to 1e-4 because the tensors we check on are also to 4 decimal places.

model = HookedTransformer.from_pretrained(MODEL)
test_regular_sequence = " four token sequence"  # Four tokens including BOS
test_duplicated_sequence = " seven token sequence seven token sequence"
test_duplicated_sequence2 = " one two three one two three"
test_regular_sequence_padded = " 2 2 2 seven token seq"
zeros_detection_pattern = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
test_duplicated_seq_len = model.to_tokens(test_duplicated_sequence).shape[-1]

expected_regular_sequence_previous_match = torch.tensor(
    [
        [0.3567, 0.2326, 0.2587, 0.2669, 0.1437, 0.5924, 0.2706, 0.2647],
        [0.3250, 0.2487, 0.2685, 0.2771, 0.2737, 0.2520, 0.2523, 0.3359],
    ]
)

expected_duplicated_sequence_previous_match = torch.tensor(
    [
        [0.2978, 0.1737, 0.1593, 0.1686, 0.0905, 0.6462, 0.1885, 0.1767],
        [0.2923, 0.2045, 0.1845, 0.2083, 0.1797, 0.1529, 0.1564, 0.2445],
    ]
)

expected_duplicated_sequence_duplicate_match = torch.tensor(
    [
        [0.0904, 0.0944, 0.0010, 0.0155, 0.2024, 0.0071, 0.0164, 0.0715],
        [0.0381, 0.0038, 0.0309, 0.0184, 0.0322, 0.0103, 0.0066, 0.0446],
    ]
)

expected_duplicated_sequence_induction_match = torch.tensor(
    [
        [0.1242, 0.0539, 0.0109, 0.0178, 0.0005, 0.0560, 0.0312, 0.0521],
        [0.0659, 0.1994, 0.0430, 0.0289, 0.0470, 0.0119, 0.1726, 0.0665],
    ]
)

expected_previous_exclude_bos_match = torch.tensor(
    [
        [0.4312, 0.1414, 0.4195, 0.3316, 0.0016, 0.7672, 0.4385, 0.2628],
        [0.4030, 0.1467, 0.3050, 0.3247, 0.3062, 0.2421, 0.2043, 0.3593],
    ]
)

expected_previous_exclude_current_token_match = torch.tensor(
    [
        [0.5441, 0.4149, 0.3545, 0.3771, 0.2738, 0.8821, 0.3797, 0.3835],
        [0.6770, 0.3445, 0.3959, 0.4228, 0.4032, 0.3449, 0.3434, 0.5770],
    ]
)

expected_previous_exclude_bos_and_current_token_match = torch.tensor(
    [
        [0.6092, 0.5601, 0.8043, 0.8732, 0.1130, 0.9122, 0.6857, 0.4405],
        [0.7011, 0.7523, 0.5545, 0.6449, 0.7958, 0.7565, 0.7082, 0.7833],
    ]
)


# Successes
class Test_detect_head_successful:
    
    @pytest.mark.parametrize(
        ("head_name", "expected"), 
        (("previous_token_head", expected_regular_sequence_previous_match),
         ("duplicate_token_head", zeros_detection_pattern),
         ("induction_head", zeros_detection_pattern)
         )
    )
    def test_regular_sequence(self, head_name, expected):
        result = head_detector.detect_head(model, test_regular_sequence, detection_pattern=head_name)
        assert torch.allclose(result, expected, atol=ATOL)
    
    @pytest.mark.parametrize(
        ("head_name", "expected"), 
        (("previous_token_head", expected_duplicated_sequence_previous_match),
         ("duplicate_token_head", expected_duplicated_sequence_duplicate_match),
         ("induction_head", expected_duplicated_sequence_induction_match)
         )
    )
    def test_duplicated_sequence(self, head_name, expected):
        result = head_detector.detect_head(model, test_duplicated_sequence, detection_pattern=head_name)
        assert torch.allclose(result, expected, atol=ATOL)

@pytest.mark.parametrize("head_name", head_detector.HEAD_NAMES)
def test_batched_equal_lengths(head_name):
    result_regular_padded = head_detector.detect_head(
        model, test_regular_sequence_padded, head_name
    )
    result_duplicated = head_detector.detect_head(
        model, test_duplicated_sequence, head_name
    )
    result_duplicated2 = head_detector.detect_head(
        model, test_duplicated_sequence2, head_name
    )
    result_batched = head_detector.detect_head(
        model,
        [
            test_regular_sequence_padded,
            test_duplicated_sequence,
            test_duplicated_sequence2,
        ],
        head_name,
    )
    expected = (result_regular_padded + result_duplicated + result_duplicated2) / 3
    assert torch.allclose(result_batched, expected, atol=ATOL)


# class Test_batched_unequal_lengths:
#     def test_previous(self):
#         s1 = test_regular_sequence
#         s2 = test_duplicated_sequence
#         s3 = [s1, s2]
#         r1 = head_detector.detect_head(model, s1, "previous_token_head")
#         r2 = head_detector.detect_head(model, s2, "previous_token_head")
#         r3 = head_detector.detect_head(model, s3, "previous_token_head")
#         expected = (r1 + r2) / 2
#         r1_weight = len(model.to_str_tokens(s1))
#         r2_weight = len(model.to_str_tokens(s2))
#         div = r1_weight + r2_weight
#         r1_normalized = r1 * r1_weight
#         r2_normalized = r2 * r2_weight
#         expected_normalized = (r1_normalized + r2_normalized) / div
#         assert torch.allclose(r3, expected_normalized, atol=ATOL)

def test_detect_head_exclude_bos():
    assert torch.allclose(
        head_detector.detect_head(
            model,
            test_regular_sequence,
            "previous_token_head",
            exclude_bos=True,
        ),
        expected_previous_exclude_bos_match,
        atol=ATOL,
    )


def test_detect_head_exclude_current_token():
    assert torch.allclose(
        head_detector.detect_head(
            model,
            test_regular_sequence,
            "previous_token_head",
            exclude_current_token=True,
        ),
        expected_previous_exclude_current_token_match,
        atol=ATOL,
    )


def test_detect_head_exclude_bos_and_current_token():
    assert torch.allclose(
        head_detector.detect_head(
            model,
            test_regular_sequence,
            "previous_token_head",
            exclude_bos=True,
            exclude_current_token=True,
        ),
        expected_previous_exclude_bos_and_current_token_match,
        atol=ATOL,
    )


def test_detect_head_with_cache():
    _, cache = model.run_with_cache(test_regular_sequence, remove_batch_dim=True)
    assert torch.allclose(
        head_detector.detect_head(
            model,
            test_regular_sequence,
            "previous_token_head",
            cache=cache,
        ),
        expected_regular_sequence_previous_match,
        atol=ATOL,
    )


##########
# Errors #
##########


def test_detect_head_with_invalid_head_name():
    with pytest.raises((AssertionError, TypeError)) as e:
        head_detector.detect_head(
            model, test_regular_sequence, "test"
        )


def test_detect_head_with_zero_sequence_length():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, "", "previous_token_head")
    assert (
        str(e.value)
        == "The sequence must be non-empty and must fit within the model's context window."
    )


def test_detect_head_with_sequence_length_outside_context_window():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(
            model, "a " * model.cfg.n_ctx, "previous_token_head"
        )
    assert (
        str(e.value)
        == "The sequence must be non-empty and must fit within the model's context window."
    )


def test_detect_head_with_invalid_detection_pattern():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, test_duplicated_sequence, torch.ones(4, 4))
    assert "The detection pattern must be a lower triangular" in str(e.value)




class Test_detect_head_non_lower_triangular_detection_pattern:
    detection_pattern = torch.tril(
        torch.ones(test_duplicated_seq_len, test_duplicated_seq_len)
    ).cuda()

    def test_no_error(self):
        head_detector.detect_head(
            model, test_duplicated_sequence, self.detection_pattern
        )
        assert True  # ugly, need to make a separate context manager for not raising an error

    def test_raises_error(self):
        detection_pattern = self.detection_pattern.clone()
        detection_pattern[0, 1] = 1
        with pytest.raises(AssertionError) as e:
            head_detector.detect_head(
                model, test_duplicated_sequence, detection_pattern
            )
        assert "The detection pattern must be a lower triangular" in str(e.value)


#################################
# Detecting with specific heads #
#################################


class Test_specific_heads:
    class Test_regular_sentence_previous_token_head:
        match = head_detector.detect_head(
            model,
            test_regular_sequence,
            "previous_token_head",
            heads=[(0, 0)],
        )

        def test_allclose(self):
            assert torch.allclose(
                self.match[0, 0],
                expected_regular_sequence_previous_match[0, 0],
                atol=ATOL,
            )

        def test_isclose(self):
            assert math.isclose(
                torch.sum(self.match),
                self.match[0, 0].item() - (model.cfg.n_layers * model.cfg.n_heads - 1),
                abs_tol=ATOL,
            )

    class Test_duplicated_sentence_previous_token_head:
        match = head_detector.detect_head(
            model,
            test_duplicated_sequence,
            "previous_token_head",
            heads=[(0, 0)],
        )

        def test_allclose(self):
            assert torch.allclose(
                self.match[0, 0],
                expected_duplicated_sequence_previous_match[0, 0],
                atol=ATOL,
            )

        def test_isclose(self):
            assert math.isclose(
                torch.sum(self.match),
                self.match[0, 0].item() - (model.cfg.n_layers * model.cfg.n_heads - 1),
                abs_tol=ATOL,
            )

    class Test_duplicated_sentence_duplicate_token_head:
        match = head_detector.detect_head(
            model,
            test_duplicated_sequence,
            "duplicate_token_head",
            heads=[(0, 0)],
        )

        def test_allclose(self):
            assert torch.allclose(
                self.match[0, 0],
                expected_duplicated_sequence_duplicate_match[0, 0],
                atol=ATOL,
            )

        def test_isclose(self):
            assert math.isclose(
                torch.sum(self.match),
                self.match[0, 0].item() - (model.cfg.n_layers * model.cfg.n_heads - 1),
                abs_tol=ATOL,
            )

    class Test_duplicated_sentence_induction_head:
        match = head_detector.detect_head(
            model,
            test_duplicated_sequence,
            "induction_head",
            heads=[(0, 0)],
        )

        def test_allclose(self):
            assert torch.allclose(
                self.match[0, 0],
                expected_duplicated_sequence_induction_match[0, 0],
                atol=ATOL,
            )

        def test_isclose(self):
            assert math.isclose(
                torch.sum(self.match),
                self.match[0, 0].item() - (model.cfg.n_layers * model.cfg.n_heads - 1),
                abs_tol=ATOL,
            )


######################
# Detection patterns #
######################


class Test_previous_token_head:
    regular_detection_pattern = head_detector.get_previous_token_head_detection_pattern(
        model.to_tokens(test_regular_sequence).cpu()
    )

    def test_regular_detection_pattern1(self):
        assert self.regular_detection_pattern.shape == (4, 4)

    def test_regular_detection_pattern2(self):
        assert (self.regular_detection_pattern[1:, :-1] == torch.eye(3)).all()

    def test_regular_detection_pattern3(self):
        assert torch.sum(self.regular_detection_pattern) == 3

    duplicate_detection_pattern = (
        head_detector.get_previous_token_head_detection_pattern(
            model.to_tokens(test_duplicated_sequence).cpu()
        )
    )

    def test_duplicate_detection_pattern1(self):
        assert self.duplicate_detection_pattern.shape == (7, 7)

    def test_duplicate_detection_pattern2(self):
        assert (self.duplicate_detection_pattern[1:, :-1] == torch.eye(6)).all()

    def test_duplicate_detection_pattern3(self):
        assert torch.sum(self.duplicate_detection_pattern) == 6


class Test_duplicate_token_head:
    detection_pattern = head_detector.get_duplicate_token_head_detection_pattern(
        model.to_tokens(test_duplicated_sequence).cpu()
    )

    def test1(self):
        assert (
            head_detector.get_duplicate_token_head_detection_pattern(
                model.to_tokens(test_regular_sequence).cpu()
            )
            == torch.zeros(4, 4)
        ).all()

    def test2(self):
        assert self.detection_pattern.shape == (7, 7)

    def test3(self):
        assert (self.detection_pattern[4:, 1:4] == torch.eye(3)).all()

    def test4(self):
        assert torch.sum(self.detection_pattern) == 3


class Test_induction_head_detection:
    detection_pattern = head_detector.get_induction_head_detection_pattern(
        model.to_tokens(test_duplicated_sequence).cpu()
    )

    def test1(self):
        assert (
            head_detector.get_duplicate_token_head_detection_pattern(
                model.to_tokens(test_regular_sequence).cpu()
            )
            == torch.zeros(4, 4)
        ).all()

    def test2(self):
        assert self.detection_pattern.shape == (7, 7)

    def test3(self):
        assert (self.detection_pattern[4:, 2:5] == torch.eye(3)).all()

    def test4(self):
        assert torch.sum(self.detection_pattern) == 3
