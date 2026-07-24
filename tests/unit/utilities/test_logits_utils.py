"""Unit tests for transformer_lens.utilities.logits_utils."""

import pytest
import torch

from transformer_lens.utilities.logits_utils import (
    _apply_repetition_penalty,
    logits_to_df,
    sample_logits,
)


class _StubTokenizer:
    """Minimal tokenizer surface used by logits_to_df (decode of single ids)."""

    def __init__(self, vocab: list[str]):
        self._vocab = vocab

    def decode(self, ids: list[int]) -> str:
        return "".join(self._vocab[i] for i in ids)


@pytest.fixture(scope="module")
def logits() -> torch.Tensor:
    return torch.tensor([1.0, 3.0, 2.0, 0.5])


class TestLogitsToDf:
    def test_columns_no_tokenizer(self, logits: torch.Tensor):
        df = logits_to_df(logits)
        assert list(df.columns) == ["token_index", "logit", "log_prob", "probability"]

    def test_columns_with_tokenizer(self, logits: torch.Tensor):
        tok = _StubTokenizer(["a", "b", "c", "d"])
        df = logits_to_df(logits, tokenizer=tok)
        assert list(df.columns) == [
            "token_index",
            "token_string",
            "logit",
            "log_prob",
            "probability",
        ]

    def test_sorted_by_descending_probability(self, logits: torch.Tensor):
        df = logits_to_df(logits)
        probs = df["probability"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_token_indices_match_logit_argsort(self, logits: torch.Tensor):
        df = logits_to_df(logits)
        # logits = [1.0, 3.0, 2.0, 0.5] -> argsort desc = [1, 2, 0, 3]
        assert df["token_index"].tolist() == [1, 2, 0, 3]

    def test_top_k_truncates(self, logits: torch.Tensor):
        df = logits_to_df(logits, top_k=2)
        assert len(df) == 2
        assert df["token_index"].tolist() == [1, 2]

    def test_token_string_decoded(self, logits: torch.Tensor):
        tok = _StubTokenizer(["a", "b", "c", "d"])
        df = logits_to_df(logits, tokenizer=tok, top_k=2)
        assert df["token_string"].tolist() == ["b", "c"]

    def test_log_prob_and_probability_consistent(self, logits: torch.Tensor):
        df = logits_to_df(logits)
        assert torch.allclose(
            torch.tensor(df["log_prob"].tolist()).exp(),
            torch.tensor(df["probability"].tolist()),
            atol=1e-6,
        )

    def test_probabilities_sum_to_one(self, logits: torch.Tensor):
        df = logits_to_df(logits)
        assert df["probability"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_logit_column_preserves_input_values(self, logits: torch.Tensor):
        df = logits_to_df(logits)
        # Order is argsort-desc; just check membership and float-equality.
        assert sorted(df["logit"].tolist()) == sorted(logits.tolist())

    def test_rejects_non_1d_input(self):
        # Shape constraint enforced by jaxtyping/beartype on Float[Tensor, "d_vocab"].
        from beartype.roar import BeartypeCallHintParamViolation

        with pytest.raises(BeartypeCallHintParamViolation):
            logits_to_df(torch.zeros(3, 4))


class TestSampleLogitsTopK:
    def test_top_k_larger_than_vocab_does_not_crash(self):
        # Regression test: before clamping top_k, final_logits.topk(top_k)
        # raised "selected index k out of range" when top_k > vocab size.
        out = sample_logits(torch.randn(1, 3), top_k=10)
        assert out.shape == (1,)
        assert 0 <= out.item() < 3

    def test_top_k_larger_than_vocab_batched(self):
        out = sample_logits(torch.randn(4, 5), top_k=8)
        assert out.shape == (4,)
        assert torch.all((out >= 0) & (out < 5))

    def test_top_k_equal_to_vocab(self):
        out = sample_logits(torch.randn(1, 4), top_k=4)
        assert out.shape == (1,)
        assert 0 <= out.item() < 4

    def test_top_k_restricts_to_dominant_token(self):
        # With top_k=1 only the argmax token is ever sampled.
        logits = torch.tensor([[0.0, 100.0, 0.0, 0.0]])
        outs = [sample_logits(logits, top_k=1).item() for _ in range(20)]
        assert set(outs) == {1}

    def test_top_k_rejects_non_positive(self):
        with pytest.raises(AssertionError):
            sample_logits(torch.randn(1, 4), top_k=0)


class TestSampleLogitsTemperature:
    def test_temperature_zero_is_greedy_argmax(self):
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
        out = sample_logits(logits, temperature=0.0)
        assert out.tolist() == [1]

    def test_temperature_zero_batched_argmax(self):
        logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]])
        out = sample_logits(logits, temperature=0.0)
        assert out.tolist() == [1, 0]

    def test_temperature_zero_applies_repetition_penalty(self):
        # Token 1 is the argmax but has appeared, so the penalty should push
        # the greedy choice onto the next-best unseen token (token 2).
        logits = torch.tensor([[0.0, 10.0, 9.0, 0.0]])
        tokens = torch.tensor([[1]])
        out = sample_logits(logits, temperature=0.0, repetition_penalty=100.0, tokens=tokens)
        assert out.tolist() == [2]


class TestSampleLogitsTopP:
    def test_top_p_keeps_dominant_token(self):
        # One token holds essentially all the probability mass, so even a small
        # top_p must keep it and it is the only token ever sampled.
        logits = torch.tensor([[0.0, 50.0, 0.0, 0.0]])
        outs = [sample_logits(logits, top_p=0.5).item() for _ in range(20)]
        assert set(outs) == {1}

    def test_top_p_rejects_out_of_range(self):
        with pytest.raises(AssertionError):
            sample_logits(torch.randn(1, 4), top_p=0.0)
        with pytest.raises(AssertionError):
            sample_logits(torch.randn(1, 4), top_p=1.5)


class TestSampleLogitsFreqPenalty:
    def test_freq_penalty_suppresses_repeated_token(self):
        # Token 0 starts as the clear favourite, but appears many times in the
        # context; a large frequency penalty should make it never get sampled.
        logits = torch.tensor([[5.0, 4.0, 4.0, 4.0]])
        tokens = torch.zeros((1, 50), dtype=torch.long)  # token 0 repeated 50x
        outs = [sample_logits(logits, freq_penalty=10.0, tokens=tokens).item() for _ in range(50)]
        assert 0 not in outs

    def test_freq_penalty_requires_tokens(self):
        with pytest.raises(AssertionError):
            sample_logits(torch.randn(1, 4), freq_penalty=1.0)


class TestApplyRepetitionPenalty:
    def test_positive_logits_divided_negative_multiplied(self):
        logits = torch.tensor([[2.0, -2.0, 0.0]])
        tokens = torch.tensor([[0, 1]])
        out = _apply_repetition_penalty(logits, tokens, penalty=2.0)
        # token 0 positive -> divided; token 1 negative -> multiplied; token 2 untouched
        assert out.tolist() == [[1.0, -4.0, 0.0]]

    def test_does_not_mutate_input(self):
        logits = torch.tensor([[2.0, -2.0, 0.0]])
        original = logits.clone()
        _apply_repetition_penalty(logits, torch.tensor([[0, 1]]), penalty=2.0)
        assert torch.equal(logits, original)
