"""MLPBridge W_in/W_out/W_gate must resolve to the TL orientation for every wrapper.

nn.Linear stores [out_features, in_features] (the transpose of the TL
convention) while Conv1D stores [in_features, out_features] (TL as-is) — a raw
``in.weight`` alias silently returned transposed weights for every
nn.Linear-backed model (all of boot_native), in both aspect ratios.
"""

import pytest
import torch
from transformers.pytorch_utils import Conv1D

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge


def _native_bridge(d_model, d_mlp):
    cfg = TransformerBridgeConfig(
        n_layers=1,
        d_model=d_model,
        d_head=d_model // 4,
        n_heads=4,
        d_mlp=d_mlp,
        d_vocab=50,
        n_ctx=8,
        act_fn="gelu",
        seed=0,
    )
    return TransformerBridge.boot_native(cfg)


@pytest.mark.parametrize(
    "d_model,d_mlp",
    [
        (32, 128),  # expansion (the usual shape)
        (128, 32),  # bottleneck — the aspect ratio a shape heuristic cannot see
    ],
)
def test_native_linear_weights_are_tl_oriented(d_model, d_mlp):
    bridge = _native_bridge(d_model, d_mlp)
    mlp = bridge.blocks[0].mlp

    assert mlp.W_in.shape == (d_model, d_mlp)
    assert mlp.W_out.shape == (d_mlp, d_model)
    # The stacked bridge-level accessors follow: [n_layers, d_model, d_mlp] etc.
    assert bridge.W_in.shape == (1, d_model, d_mlp)
    assert bridge.W_out.shape == (1, d_mlp, d_model)

    # Orientation must reflect the actual parameters, not just shapes: the
    # nn.Linear weight is [d_mlp, d_model]; the property must be its transpose.
    raw = getattr(mlp, "in").weight
    assert torch.equal(mlp.W_in, raw.T)


def test_conv1d_weight_passes_through():
    """Conv1D (GPT-2 style) already stores [in, out]; no transpose expected."""

    class _Proj:
        def __init__(self, weight, component):
            self.weight = weight
            self.original_component = component

    mlp = MLPBridge.__new__(MLPBridge)
    conv = Conv1D(8, 4)  # nf=8 (out), nx=4 (in) -> weight [4, 8]
    object.__setattr__(mlp, "_test_proj", _Proj(conv.weight, conv))
    # exercise the resolver directly with a synthetic projection
    weight = mlp._tl_oriented_weight("_test_proj")
    assert weight.shape == (4, 8)
    assert torch.equal(weight, conv.weight)


def test_missing_gate_raises_attribute_error():
    """Non-gated MLPs must not appear to have W_gate (hasattr contract)."""
    bridge = _native_bridge(32, 128)
    mlp = bridge.blocks[0].mlp
    assert not hasattr(mlp, "W_gate")
    with pytest.raises(AttributeError, match="gate"):
        _ = mlp.W_gate
