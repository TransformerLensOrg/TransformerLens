"""VisionProjectionBridge forward passthrough.

Regression coverage for Mistral3-style projectors that take an extra
positional (image_sizes): the bridge must pass *args through untouched
while still firing hook_in/hook_out.
"""

import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.vision_projection import (
    VisionProjectionBridge,
)


class _TwoArgProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_image_sizes = None

    def forward(self, features, image_sizes):
        self.seen_image_sizes = image_sizes
        return features * 2


def test_extra_positional_passes_through_and_hooks_fire():
    proj = _TwoArgProjector()
    bridge = VisionProjectionBridge(name="multi_modal_projector")
    bridge.set_original_component(proj)

    fired = []
    bridge.hook_out.add_hook(lambda t, hook: fired.append(t.shape))

    x = torch.ones(2, 4)
    sizes = torch.tensor([[8, 8], [16, 16]])
    out = bridge(x, sizes)

    assert torch.equal(out, x * 2)
    assert proj.seen_image_sizes is sizes
    assert fired == [x.shape]
