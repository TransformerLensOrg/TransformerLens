import pytest

from transformer_lens.evals import IOIDataset, ioi_eval
from transformer_lens.HookedTransformer import HookedTransformer


@pytest.fixture(scope="module")
def model():
    return HookedTransformer.from_pretrained("gpt2-small")


def test_basic_ioi_eval(model):
    """
    Test IOI evaluation with default dataset and settings.
    """
    results = ioi_eval(model, num_samples=100)
    assert results["Accuracy"] >= 0.99


def test_symmetric_samples(model):
    """
    Test IOI evaluation with symmetric=True so prompts are in symmetric pairs.
    """
    ds = IOIDataset(tokenizer=model.tokenizer, num_samples=100, symmetric=True)
    results = ioi_eval(model, dataset=ds)
    assert results["Logit Difference"] > 2.0
    assert results["Accuracy"] > 0.9


def test_custom_dataset_ioi_eval(model):
    """
    Test IOI eval with custom dataset using different templates, names, and objects.
    """
    ds = IOIDataset(
        tokenizer=model.tokenizer,
        num_samples=100,
        templates=["[A] met with [B]. [B] gave the [OBJECT] to [A]"],
        names=["Alice", "Bob", "Charlie"],
        nouns={"OBJECT": ["ball", "book"]},
    )
    results = ioi_eval(model, dataset=ds)
    assert results["Logit Difference"] > 2.0
    assert results["Accuracy"] >= 0.99


def test_multitoken_names_ioi_eval(model):
    """
    Test the IOI evaluation with multi-token names in the dataset.
    """
    ds = IOIDataset(
        tokenizer=model.tokenizer,
        num_samples=100,
        names=["John Smith", "John Doe"],
    )
    results = ioi_eval(model, dataset=ds)
    assert results["Logit Difference"] > 2.0
    assert results["Accuracy"] >= 0.99


def test_inverted_template(model):
    """
    Test IOI eval with an unnatural template (BAAA).
    This should result in a negative logit difference and very low accuracy.
    """
    ds = IOIDataset(
        tokenizer=model.tokenizer,
        num_samples=100,
        templates=["[B] met with [A]. [A] said hello to [A]"],
    )
    results = ioi_eval(model, dataset=ds)
    assert results["Logit Difference"] < -2.0
    assert results["Accuracy"] <= 0.01
