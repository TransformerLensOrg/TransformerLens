import math

import pytest
import torch

from transformer_lens import HookedTransformer, head_detector
# from transformer_lens.utils import check_structure as check

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
        [-0.0370, -0.2850, -0.2330, -0.2160, -0.4630, 0.4350, -0.2090, -0.2210],
        [-0.1000, -0.2530, -0.2130, -0.1960, -0.2030, -0.2460, -0.2450, -0.0780],
    ]
)

expected_duplicated_sequence_previous_match = torch.tensor(
    [
        [-0.2620, -0.5100, -0.5390, -0.5200, -0.6760, 0.4350, -0.4800, -0.5040],
        [-0.2720, -0.4480, -0.4880, -0.4410, -0.4980, -0.5510, -0.5440, -0.3680],
    ]
)

expected_duplicated_sequence_duplicate_match = torch.tensor(
    [
        [-0.2480, -0.2400, -0.4270, -0.3980, -0.0240, -0.4140, -0.3960, -0.2860],
        [-0.3520, -0.4210, -0.3670, -0.3920, -0.3640, -0.4080, -0.4150, -0.3390],
    ]
)

expected_duplicated_sequence_induction_match = torch.tensor(
    [
        [-0.1800, -0.3210, -0.4070, -0.3930, -0.4280, -0.3170, -0.3660, -0.3240],
        [-0.2970, -0.0300, -0.3430, -0.3710, -0.3350, -0.4050, -0.0830, -0.2950],
    ]
)

expected_previous_exclude_bos_match = torch.tensor(
    [
        [0.4560, 0.3180, 0.4930, 0.4770, 0.2720, 0.7640, 0.4870, 0.4300],
        [0.3770, 0.4750, 0.4380, 0.4330, 0.4550, 0.4850, 0.4870, 0.4130],
    ]
)

expected_previous_exclude_current_token_match = torch.tensor(
    [
        [0.3080, 0.1550, 0.0380, 0.0760, 0.0130, 0.7630, 0.0780, 0.0890],
        [0.4200, 0.0250, 0.1090, 0.1490, 0.1190, 0.0230, 0.0200, 0.3400],
    ]
)

expected_previous_exclude_bos_and_current_token_match = torch.tensor(
    [
        [0.5500, 0.5080, 0.5130, 0.5190, 0.4980, 0.8420, 0.5240, 0.4890],
        [0.6470, 0.5030, 0.5100, 0.5280, 0.5260, 0.5050, 0.5030, 0.5810],
    ]
)


# Successes
class Test_detect_head_successful:
    @pytest.mark.parametrize(
        ("head_name", "expected"),
        (
            ("previous_token_head", expected_regular_sequence_previous_match),
            ("duplicate_token_head", zeros_detection_pattern),
            ("induction_head", zeros_detection_pattern),
        ),
    )
    def test_regular_sequence(self, head_name, expected):
        result = head_detector.detect_head(
            model, test_regular_sequence, detection_pattern=head_name
        )
        assert torch.allclose(result, expected, atol=ATOL)

    @pytest.mark.parametrize(
        ("head_name", "expected"),
        (
            ("previous_token_head", expected_duplicated_sequence_previous_match),
            ("duplicate_token_head", expected_duplicated_sequence_duplicate_match),
            ("induction_head", expected_duplicated_sequence_induction_match),
        ),
    )
    def test_duplicated_sequence(self, head_name, expected):
        result = head_detector.detect_head(
            model, test_duplicated_sequence, detection_pattern=head_name
        )
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


class Test_batched_unequal_lengths:
    def test_previous(self):
        s1 = test_regular_sequence
        s2 = test_duplicated_sequence
        s3 = [s1, s2]
        r1 = head_detector.detect_head(model, s1, "previous_token_head")
        r2 = head_detector.detect_head(model, s2, "previous_token_head")
        r3 = head_detector.detect_head(model, s3, "previous_token_head")

        assert torch.allclose(r3, (r1 + r2) / 2, atol=ATOL)


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
        head_detector.detect_head(model, test_regular_sequence, "test")


def test_detect_head_with_zero_sequence_length():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, "", "previous_token_head")
    assert (
        str(e.value)
        == "The sequence must be non-empty and must fit within the model's context window."
    )


def test_detect_head_with_sequence_length_outside_context_window():
    with pytest.raises(AssertionError) as e:
        head_detector.detect_head(model, "a " * model.cfg.n_ctx, "previous_token_head")
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
    )

    def test_no_error(self):
        head_detector.detect_head(
            model, test_duplicated_sequence, self.detection_pattern.to(device=model.cfg.device)
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
