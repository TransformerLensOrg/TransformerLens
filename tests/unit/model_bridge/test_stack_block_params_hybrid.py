"""_stack_block_params on interleaved-MoE (hybrid) models.

The filter matched on the FIRST path segment only, and every block has `mlp` —
so nothing was filtered and the sparse layer's AttributeError killed the
accessor for the whole model instead of returning the dense layers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


class _DenseMLP(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.W_in = torch.full((4, 8), value)


class _SparseMLP(torch.nn.Module):
    """MoE layer: per-expert weights, no single W_in."""

    @property
    def W_in(self):
        raise AttributeError("W_in exists only on dense layers")


class _Block(torch.nn.Module):
    def __init__(self, mlp) -> None:
        super().__init__()
        self.mlp = mlp  # nn.Module assignment registers into _modules


_BORROWED_HELPERS = (
    "_enumerate_blocks",
    "_resolve_submodule_name",
    "_rewrite_submodule_path",
)


def _stub(mlps):
    blocks = torch.nn.ModuleList([_Block(m) for m in mlps])
    # _stack_block_params walks the registered block lists, so the stub exposes
    # the same _modules shape a real bridge does and borrows the real helpers.
    stub = SimpleNamespace(
        blocks=blocks,
        _modules={"blocks": blocks},
        cfg=SimpleNamespace(n_devices=1, device=None),
    )
    for helper in _BORROWED_HELPERS:
        setattr(stub, helper, getattr(TransformerBridge, helper).__get__(stub))
    return stub


def test_interleaved_model_stacks_dense_layers_only(caplog) -> None:
    stub = _stub([_DenseMLP(0.0), _SparseMLP(), _DenseMLP(2.0)])
    with caplog.at_level("WARNING"):
        stacked = TransformerBridge._stack_block_params(stub, "mlp.W_in")
    assert stacked.shape == (2, 4, 8)
    assert torch.equal(stacked[1], torch.full((4, 8), 2.0))
    # The hybrid warning must carry the index mapping, as designed.
    assert any("only 2/3 blocks" in record.getMessage() for record in caplog.records)


def test_all_sparse_raises_with_the_full_path() -> None:
    stub = _stub([_SparseMLP(), _SparseMLP()])
    with pytest.raises(AttributeError, match="mlp.W_in"):
        TransformerBridge._stack_block_params(stub, "mlp.W_in")
