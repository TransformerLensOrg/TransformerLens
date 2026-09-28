"""_masked_softmax on fully masked query rows.

Compatibility mode masks with -inf, so a query with no visible key (the leading
pads of a left-padded row) softmaxes to NaN. Scrubbing the NaN after softmax
fixed the forward, but softmax's backward still leaked NaN into the scores.
"""

from __future__ import annotations

import copy

import torch

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)


def _attn_bridge(compatibility_mode: bool):
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(
        make_bridge_cfg("LlamaForCausalLM", d_model=8, n_heads=2, d_head=4)
    )
    bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules["attn"])
    bridge.compatibility_mode = compatibility_mode
    return bridge


def _left_padded_scores() -> torch.Tensor:
    """[1, 1, 3, 3] causal scores whose first key is a pad: query 0 sees nothing."""
    scores = torch.randn(1, 1, 3, 3)
    blocked = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    blocked[:, 0] = True
    return scores.masked_fill(blocked, -torch.inf).requires_grad_(True)


def test_compatibility_fully_masked_row_has_zero_pattern_and_finite_grad() -> None:
    bridge = _attn_bridge(compatibility_mode=True)
    scores = _left_padded_scores()

    pattern = bridge._masked_softmax(scores, dtype=torch.float32)
    (grad,) = torch.autograd.grad((pattern * torch.randn_like(pattern)).sum(), scores)

    assert (pattern[..., 0, :] == 0).all()
    torch.testing.assert_close(
        pattern[..., 1:, :], torch.softmax(scores.detach()[..., 1:, :], dim=-1)
    )
    assert torch.isfinite(grad).all()
    assert (grad[..., 0, :] == 0).all()


def test_non_compatibility_mode_is_plain_softmax() -> None:
    bridge = _attn_bridge(compatibility_mode=False)
    scores = torch.randn(1, 2, 4, 4)

    torch.testing.assert_close(
        bridge._masked_softmax(scores), torch.softmax(scores, dim=-1), rtol=0, atol=0
    )
