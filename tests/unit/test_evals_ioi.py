"""Tests for IOIDataset in transformer_lens/evals.py.

Regression test for https://github.com/TransformerLensOrg/TransformerLens/issues/515
"""

from unittest.mock import MagicMock

from transformer_lens.evals import IOIDataset


def _make_tokenizer():
    """Minimal mock tokenizer sufficient for IOIDataset."""
    tok = MagicMock()
    tok.encode.side_effect = lambda text: [1, 2, 3]
    tok.bos_token_id = 0
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
    assert [s["text"] for s in ds1.samples] == [s["text"] for s in ds2.samples], (
        "IOIDataset with the same seed should be reproducible."
    )


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
