"""Unit tests for the benchmarks' local RNG seeding."""

import pytest
import torch

from transformer_lens.benchmarks.utils import deterministic_rng


@pytest.fixture
def device_generators(monkeypatch):
    """A CUDA device and MPS, faked so their generators show on a CPU-only machine."""
    state = {"cuda": "caller", "mps": "caller"}

    def setter(name):
        return lambda new, *args, **kwargs: state.__setitem__(name, new)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *args, **kwargs: state["cuda"])
    monkeypatch.setattr(torch.cuda, "set_rng_state", setter("cuda"))
    monkeypatch.setattr(torch.cuda, "manual_seed", setter("cuda"))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", setter("cuda"))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda *args, **kwargs: state["mps"])
    monkeypatch.setattr(torch.mps, "set_rng_state", setter("mps"))
    monkeypatch.setattr(torch.mps, "manual_seed", setter("mps"))
    return state


def test_device_generators_are_seeded_inside_and_restored_after(device_generators):
    # Sampling on a GPU draws from the device's generator, so it must be seeded too
    with deterministic_rng(seed=7):
        assert device_generators == {"cuda": 7, "mps": 7}
    assert device_generators == {"cuda": "caller", "mps": "caller"}

    with pytest.raises(RuntimeError, match="sampling failed"):
        with deterministic_rng(seed=7):
            raise RuntimeError("sampling failed")
    assert device_generators == {"cuda": "caller", "mps": "caller"}


def test_cpu_draws_are_seeded_and_the_callers_stream_continues():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    with deterministic_rng(seed=7):
        inside = torch.rand(3)
    assert torch.equal(torch.rand(3), expected)
    assert torch.equal(inside, torch.rand(3, generator=torch.Generator().manual_seed(7)))
