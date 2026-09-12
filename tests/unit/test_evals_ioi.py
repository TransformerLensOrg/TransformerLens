"""Tests for IOIDataset and ioi_eval in transformer_lens/evals.py."""

from unittest.mock import MagicMock

import torch

from transformer_lens.evals import IOIDataset, ioi_eval


def _make_tokenizer():
    """Minimal mock tokenizer sufficient for IOIDataset."""
    tok = MagicMock()
    tok.encode.side_effect = lambda text, add_special_tokens=True: [1, 2, 3]
    tok.bos_token_id = 0
    return tok


def _make_automatic_bos_tokenizer():
    """Tokenizer stub whose default encode path inserts a BOS token."""
    tok = MagicMock()
    tok.bos_token_id = 0

    def encode(text, add_special_tokens=True):
        if text.startswith(" "):
            payload = [30 if text.strip() == "Alice" else 31]
        else:
            target = 30 if text.split()[-1] == "Alice" else 31
            payload = [10, 20, target]
        return ([tok.bos_token_id] if add_special_tokens else []) + payload

    tok.encode.side_effect = encode
    return tok


def test_ioi_dataset_produces_diverse_samples():
    """IOIDataset must generate varied samples, not all-identical ones.

    Regression test for #515: random.seed(42) was called inside get_sample()
    on every invocation, so every sample was identical.
    """
    tokenizer = _make_tokenizer()
    dataset = IOIDataset(tokenizer, num_samples=20)
    texts = [s["text"] for s in dataset.samples]
    assert len(set(texts)) > 1, (
        "All IOIDataset samples are identical — "
        "random.seed() must not be called inside get_sample()."
    )


def test_ioi_dataset_reproducible_with_seed():
    """IOIDataset with the same seed must produce the same samples."""
    tokenizer = _make_tokenizer()
    ds1 = IOIDataset(tokenizer, num_samples=20, seed=42)
    ds2 = IOIDataset(tokenizer, num_samples=20, seed=42)
    assert [s["text"] for s in ds1.samples] == [
        s["text"] for s in ds2.samples
    ], "IOIDataset with the same seed should be reproducible."


def test_ioi_dataset_different_seeds_differ():
    """IOIDataset with different seeds should (very likely) produce different samples."""
    tokenizer = _make_tokenizer()
    ds1 = IOIDataset(tokenizer, num_samples=20, seed=0)
    ds2 = IOIDataset(tokenizer, num_samples=20, seed=99)
    texts1 = [s["text"] for s in ds1.samples]
    texts2 = [s["text"] for s in ds2.samples]
    assert texts1 != texts2, "Different seeds should produce different orderings."


def test_ioi_dataset_no_seed_is_valid():
    """IOIDataset without a seed should work fine (no error)."""
    tokenizer = _make_tokenizer()
    dataset = IOIDataset(tokenizer, num_samples=10)
    assert len(dataset.samples) == 10


def test_ioi_dataset_symmetric():
    """IOIDataset with symmetric=True should produce 2x samples (one pair per call)."""
    tokenizer = _make_tokenizer()
    dataset = IOIDataset(tokenizer, num_samples=10, symmetric=True)
    assert len(dataset.samples) == 10


def test_ioi_dataset_prepend_bos_is_the_only_source_of_special_tokens():
    tokenizer = _make_automatic_bos_tokenizer()
    with_bos = IOIDataset(
        tokenizer,
        templates=["[A] met [B] and [A]"],
        names=["Alice", "Bob"],
        nouns={},
        num_samples=1,
        prepend_bos=True,
        seed=0,
    )[0]
    without_bos = IOIDataset(
        tokenizer,
        templates=["[A] met [B] and [A]"],
        names=["Alice", "Bob"],
        nouns={},
        num_samples=1,
        prepend_bos=False,
        seed=0,
    )[0]

    assert with_bos["prompt"].tolist() == [0, 10, 20, 31]
    assert without_bos["prompt"].tolist() == [10, 20, 31]
    for item in (with_bos, without_bos):
        assert {tuple(item["IO"].tolist()), tuple(item["S"].tolist())} == {(30,), (31,)}


def test_ioi_eval_reads_logits_before_the_first_answer_token():
    tokenizer = _make_automatic_bos_tokenizer()
    dataset = IOIDataset(
        tokenizer,
        templates=["[A] met [B] and [A]"],
        names=["Alice", "Bob"],
        nouns={},
        num_samples=1,
        prepend_bos=True,
        seed=0,
    )

    class PositionSensitiveModel:
        def __call__(self, tokens, return_type):
            assert return_type == "logits"
            logits = torch.zeros(tokens.shape[0], tokens.shape[1], 32)
            for row, prompt in enumerate(tokens):
                answer = int(prompt[-1])
                distractor = 31 if answer == 30 else 30
                logits[row, 2, answer] = 1
                logits[row, 2, distractor] = -1
                logits[row, 3, answer] = -1
                logits[row, 3, distractor] = 1
            return logits

    result = ioi_eval(PositionSensitiveModel(), dataset=dataset, batch_size=1, tokenizer=tokenizer)

    assert result == {"Logit Difference": 2.0, "Accuracy": 1.0}
