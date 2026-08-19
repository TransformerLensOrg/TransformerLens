"""Loss benchmarks must follow the explicit-label Bridge contract."""

from typing import Any

import pytest
import torch

from transformer_lens.benchmarks.forward_pass import benchmark_loss_equivalence
from transformer_lens.model_bridge import TransformerBridge


def test_benchmark_loss_equivalence_supplies_tokenized_self_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = object.__new__(TransformerBridge)
    torch.nn.Module.__init__(bridge)
    labels = torch.tensor([[1, 2, 3]])
    forward_kwargs: dict[str, Any] = {}

    def to_tokens(text: str, **kwargs: Any) -> torch.Tensor:
        assert text == "benchmark text"
        return labels

    def forward(input: str, **kwargs: Any) -> torch.Tensor:
        assert input == "benchmark text"
        forward_kwargs.update(kwargs)
        return torch.tensor(1.25)

    monkeypatch.setattr(bridge, "to_tokens", to_tokens)
    monkeypatch.setattr(bridge, "forward", forward)

    result = benchmark_loss_equivalence(bridge, "benchmark text", reference_loss=1.25)

    assert result.passed
    assert forward_kwargs["labels"] is labels
    assert forward_kwargs["return_type"] == "loss"
