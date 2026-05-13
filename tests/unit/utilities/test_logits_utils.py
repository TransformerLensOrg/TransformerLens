"""Unit tests for transformer_lens.utilities.logits_utils."""

import pandas as pd
import pytest
import torch

from transformer_lens.utilities.logits_utils import logits_to_df


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
    def test_returns_dataframe(self, logits: torch.Tensor):
        df = logits_to_df(logits)
        assert isinstance(df, pd.DataFrame)

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
