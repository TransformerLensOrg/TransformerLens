"""CLS-token classifier head bridge component.

`ViTForImageClassification.forward` / `DeiTForImageClassification.forward` both do:

    sequence_output = outputs.last_hidden_state
    pooled_output = sequence_output[:, 0, :]      # CLS token only
    logits = self.classifier(pooled_output)

This is *not* expressible with the base `GeneralizedComponent.forward()` (nor a
plain `LinearBridge`): that generic forward calls `original_component(*args,
**kwargs)` with no slicing step, so applying it directly to `classifier` would
run the Linear over every position in the sequence, not just the CLS token.
Hence a real override, verified against the actual base.py this time (not the
extrapolated version from before you shared it):

- `self.hook_in` / `self.hook_out` / `self.original_component` are exactly the
  names and semantics base.py defines.
- `original_component` being unset returns `None` (not an exception) per
  base.py's property, so a direct `self.original_component(pooled)` call would
  fail with a confusing `'NoneType' object is not callable` instead of a clear
  message — added an explicit guard raising the same RuntimeError text
  `GeneralizedComponent.forward()` itself uses, so the failure mode is
  consistent across both generic and custom components.

Deliberately NOT covering `DeiTForImageClassificationWithTeacher` (the dual
cls_classifier + distillation_classifier head, averaged) — see vit.py's
module docstring for why.
"""

from typing import Any

from torch import Tensor

from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent


class VisionClassifierHeadBridge(GeneralizedComponent):
    """Wraps a plain classifier nn.Linear, but slices a single token position first.

    token_index=0 (default) for the CLS token — use this for
    ViTForImageClassification and (non-teacher) DeiTForImageClassification, both
    of which classify off index 0 regardless of whether the underlying embeddings
    module also carries a distillation token at index 1.
    """

    def __init__(self, name: str, config: Any = None, submodules=None, token_index: int = 0) -> None:
        super().__init__(name, config, submodules)
        self.token_index = token_index

    def forward(self, sequence_output: Tensor, **kwargs: Any) -> Tensor:
        original_component = self.original_component
        if original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        sequence_output = self.hook_in(sequence_output)
        pooled = sequence_output[:, self.token_index, :]
        logits = original_component(pooled)
        return self.hook_out(logits)
