"""The eval helpers default to the model's own device (`cfg.device`), not "cuda".

`evaluate_on_dataset` and `induction_loss` used to hardcode "cuda", and `evaluate` dropped
the device entirely, which broke CPU-only and MPS machines.
"""

from typing import Any, Dict, List

import torch

from transformer_lens import evals
from transformer_lens.evals import evaluate, evaluate_on_dataset, induction_loss


class StubConfig:
    """Stand-in for the two cfg fields the eval helpers read."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.default_prepend_bos = False


class StubModel:
    """Records the device of every tensor handed to it. Loads nothing."""

    def __init__(self, device: str = "cpu", d_vocab: int = 20000) -> None:
        self.cfg = StubConfig(device)
        self.d_vocab = d_vocab
        self.seen_devices: List[torch.device] = []

    def __call__(self, tokens: torch.Tensor, return_type: str = "loss") -> Any:
        self.seen_devices.append(tokens.device)
        if return_type == "logits":
            batch, seq = tokens.shape
            return torch.zeros(batch, seq, self.d_vocab, device=tokens.device)
        return torch.zeros(1, device=tokens.device)


def make_loader(n_batches: int = 2) -> List[Dict[str, torch.Tensor]]:
    """A data loader is only iterated over, so a list of batches is enough."""
    return [{"tokens": torch.zeros(1, 4, dtype=torch.long)} for _ in range(n_batches)]


def test_evaluate_on_dataset_uses_the_models_device() -> None:
    """Batches go to cfg.device, not a hardcoded "cuda"."""
    model = StubModel(device="cpu")

    evaluate_on_dataset(model, make_loader(), truncate=1)

    assert model.seen_devices
    assert all(device.type == "cpu" for device in model.seen_devices)


def test_evaluate_on_dataset_honours_an_explicit_device() -> None:
    """An explicit argument wins over cfg.device; the "meta" stub catches it being ignored."""
    model = StubModel(device="meta")

    evaluate_on_dataset(model, make_loader(), truncate=1, device="cpu")

    assert model.seen_devices
    assert all(device.type == "cpu" for device in model.seen_devices)


def test_induction_loss_builds_tokens_on_the_models_device() -> None:
    """The random tokens land on cfg.device; "meta" can't match any hardcoded default."""
    model = StubModel(device="meta")

    induction_loss(model, batch_size=2, subseq_len=3)

    assert model.seen_devices
    assert all(device.type == "meta" for device in model.seen_devices)


def test_induction_loss_honours_an_explicit_device() -> None:
    """An explicit argument wins over cfg.device."""
    model = StubModel(device="meta")

    induction_loss(model, batch_size=2, subseq_len=3, device="cpu")

    assert model.seen_devices
    assert all(device.type == "cpu" for device in model.seen_devices)


def test_evaluate_forwards_its_device_to_evaluate_on_dataset(monkeypatch) -> None:
    """`evaluate` forwards an explicit device through to the batches."""
    monkeypatch.setattr(evals, "DATASET_NAMES", ["stub"])
    monkeypatch.setattr(evals, "DATASET_LOADERS", [lambda tokenizer, batch_size: make_loader()])
    model = StubModel(device="meta")

    losses = evaluate(model, truncate=1, tokenizer=object(), device="cpu")

    assert set(losses) == {"stub_loss"}
    assert model.seen_devices
    assert all(device.type == "cpu" for device in model.seen_devices)
