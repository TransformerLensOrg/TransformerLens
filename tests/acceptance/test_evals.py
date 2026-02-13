import pytest

from transformer_lens.evals import (
    IOIDataset,
    ioi_eval,
    make_mmlu_data_loader,
    mmlu_eval,
)
from transformer_lens.HookedTransformer import HookedTransformer


@pytest.fixture(scope="module")
def model():
    return HookedTransformer.from_pretrained("gpt2-small", device="cpu")


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


def test_mmlu_data_loader_single_subject():
    """
    Test loading MMLU data for a single subject.
    """
    data = make_mmlu_data_loader(subjects="abstract_algebra", num_samples=5)
    assert len(data) == 5
    assert all(isinstance(d, dict) for d in data)
    assert all("question" in d for d in data)
    assert all("choices" in d for d in data)
    assert all("answer" in d for d in data)
    assert all("subject" in d for d in data)
    assert all(len(d["choices"]) == 4 for d in data)
    assert all(d["subject"] == "abstract_algebra" for d in data)


def test_mmlu_data_loader_multiple_subjects():
    """
    Test loading MMLU data for multiple subjects.
    """
    subjects = ["abstract_algebra", "anatomy"]
    data = make_mmlu_data_loader(subjects=subjects, num_samples=3)
    assert len(data) == 6  # 3 samples per subject
    subjects_in_data = {d["subject"] for d in data}
    assert subjects_in_data == set(subjects)


def test_mmlu_data_loader_invalid_subject():
    """
    Test that invalid subject names raise an error.
    """
    with pytest.raises(ValueError, match="Invalid subject"):
        make_mmlu_data_loader(subjects="invalid_subject_name")


def test_mmlu_eval_single_subject(model):
    """
    Test MMLU evaluation on a single subject with a small number of samples.
    Uses a small model and few samples for fast CI execution.
    """
    results = mmlu_eval(model, subjects="abstract_algebra", num_samples=5, device="cpu")
    assert "accuracy" in results
    assert "num_correct" in results
    assert "num_total" in results
    assert "subject_scores" in results
    assert 0 <= results["accuracy"] <= 1
    assert results["num_total"] == 5
    assert results["num_correct"] <= results["num_total"]
    assert "abstract_algebra" in results["subject_scores"]


def test_mmlu_eval_multiple_subjects(model):
    """
    Test MMLU evaluation on multiple subjects.
    """
    subjects = ["abstract_algebra", "anatomy"]
    results = mmlu_eval(model, subjects=subjects, num_samples=3, device="cpu")
    assert results["num_total"] == 6  # 3 samples per subject
    assert len(results["subject_scores"]) == 2
    assert all(subject in results["subject_scores"] for subject in subjects)
    assert all(0 <= acc <= 1 for acc in results["subject_scores"].values())
