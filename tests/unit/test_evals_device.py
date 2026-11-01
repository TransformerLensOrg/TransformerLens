"""The eval helpers must run on the model's own device.

`evaluate_on_dataset` and `induction_loss` defaulted `device` to the literal
string "cuda", and `evaluate` never forwarded a device at all and exposed no
parameter for one. On a CPU-only or MPS machine that moved batches to a device
the model was not on. `cfg.device` already holds the right answer, which is what
`mmlu_eval` in the same module uses.
"""

import inspect
from typing import Any, Dict, List

import torch

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
    """Before the fix this moved batches to "cuda" whatever cfg.device said."""
    model = StubModel(device="cpu")

    evaluate_on_dataset(model, make_loader(), truncate=1)

    assert model.seen_devices
    assert all(device.type == "cpu" for device in model.seen_devices)


def test_evaluate_on_dataset_honours_an_explicit_device() -> None:
    """An explicit argument still wins over cfg.device."""
    model = StubModel(device="cpu")

    evaluate_on_dataset(model, make_loader(), truncate=1, device="cpu")

    assert all(device.type == "cpu" for device in model.seen_devices)


def test_induction_loss_builds_tokens_on_the_models_device() -> None:
    """Before the fix the random sequence was always allocated on "cuda"."""
    model = StubModel(device="cpu")

    induction_loss(model, batch_size=2, subseq_len=3)

    assert model.seen_devices
    assert all(device.type == "cpu" for device in model.seen_devices)


def test_eval_helpers_no_longer_hardcode_cuda() -> None:
    """None means "ask the model"; a literal "cuda" default cannot be overridden
    by `evaluate`, which is how the bug reached callers."""
    for function in (evaluate_on_dataset, induction_loss, evaluate):
        parameter = inspect.signature(function).parameters.get("device")
        assert parameter is not None, f"{function.__name__} takes no device argument"
        assert parameter.default is None, f"{function.__name__} defaults device to a fixed device"
