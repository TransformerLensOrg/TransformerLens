"""CLS-token classifier head bridge component.

HF's ViTForImageClassification.forward() / DeiTForImageClassification.forward()
already pool the CLS token before calling self.classifier:

    sequence_output = outputs.last_hidden_state
    pooled_output = sequence_output[:, 0, :]
    logits = self.classifier(pooled_output)

So this component only ever receives an already-pooled (batch, hidden) tensor.
No slicing needed here — this is a thin, hook-named pass-through.

Deliberately NOT covering DeiTForImageClassificationWithTeacher (dual
cls_classifier + distillation_classifier head) — see vit.py's docstring.
"""

from typing import Any

from torch import Tensor

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class VisionClassifierHeadBridge(GeneralizedComponent):
    """Wraps the classifier nn.Linear that HF calls with an already-pooled CLS token."""

    def forward(self, pooled_output: Tensor, **kwargs: Any) -> Tensor:
        original_component = self.original_component
        if original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        pooled_output = self.hook_in(pooled_output)
        logits = original_component(pooled_output)
        return self.hook_out(logits)
