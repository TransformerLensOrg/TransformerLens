"""Tests for the SigLIP/CLIP vision-encoder ``set_original_component`` shim.

transformers >= 5.6.0 flattened ``SiglipVisionModel``/``CLIPVisionModel`` so they no
longer expose an inner ``.vision_model``, but the bridges resolve their submodule paths
through ``vision_model.*``. The shim self-aliases ``vision_model`` to the model itself
when it's absent, using ``object.__setattr__`` specifically so the self-reference is NOT
registered as a submodule — otherwise ``named_modules``/``state_dict`` would recurse
forever on the cycle.
"""

import pytest
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.clip_vision_encoder import (
    CLIPVisionEncoderBridge,
)
from transformer_lens.model_bridge.generalized_components.siglip_vision_encoder import (
    SiglipVisionEncoderBridge,
)


class _FlattenedVisionModel(nn.Module):
    """transformers >= 5.6.0 shape: transformer inlined, no inner ``.vision_model``."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.post_layernorm = nn.LayerNorm(2)


class _WrappedVisionModel(nn.Module):
    """transformers < 5.6.0 shape: a real inner ``.vision_model`` submodule."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_model = nn.Linear(2, 2)


@pytest.fixture(
    params=[CLIPVisionEncoderBridge, SiglipVisionEncoderBridge], ids=lambda c: c.__name__
)
def bridge(request):
    """A fresh CLIP/SigLIP vision-encoder bridge (the shim is identical in both)."""
    return request.param(name="vision_tower")


def test_aliases_self_when_vision_model_absent(bridge):
    """>= 5.6.0: vision_model is synthesized so vision_model.* paths still resolve."""
    comp = _FlattenedVisionModel()
    assert not hasattr(comp, "vision_model")  # precondition

    bridge.set_original_component(comp)

    assert comp.vision_model is comp
    assert bridge.original_component is comp


def test_self_alias_is_not_a_registered_submodule(bridge):
    """The reason for object.__setattr__: a registered self-cycle would recurse forever."""
    comp = _FlattenedVisionModel()
    bridge.set_original_component(comp)

    assert comp.vision_model is comp  # alias was added...
    assert "vision_model" not in comp._modules  # ...but as a plain attr, not a submodule
    # Enumerating modules must terminate and not surface the alias as a child.
    assert "vision_model" not in [name for name, _ in comp.named_modules()]
    bridge.state_dict()  # would hang/blow the stack if vision_model were registered


def test_preserves_existing_vision_model(bridge):
    """< 5.6.0: a real inner vision_model must be left untouched (no self-aliasing)."""
    comp = _WrappedVisionModel()
    inner = comp.vision_model

    bridge.set_original_component(comp)

    assert comp.vision_model is inner
    assert bridge.original_component is comp


def test_idempotent(bridge):
    """Calling it twice keeps a single self-alias and never registers a submodule."""
    comp = _FlattenedVisionModel()
    bridge.set_original_component(comp)
    bridge.set_original_component(comp)

    assert comp.vision_model is comp
    assert "vision_model" not in comp._modules
