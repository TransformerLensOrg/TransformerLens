"""Vision (ViT-style) embeddings bridge component.

Wraps a HF `ViTEmbeddings` / `DeiTEmbeddings` module directly and forwards through
it unmodified. The patch-conv projection, CLS-token (and, for DeiT, distillation-
token) concatenation, and position-embedding addition are HF's math — we don't
reimplement any of it, we just point at the real module and hook its input/output.

Why one class covers both ViT and DeiT: `ViTEmbeddings.forward` and
`DeiTEmbeddings.forward` have an *identical* signature
    (pixel_values, bool_masked_pos=None, interpolate_pos_encoding=False)
The only structural difference between them (DeiT prepends a distillation token
in addition to CLS, and sizes `position_embeddings` for +2 slots instead of +1)
lives entirely inside whichever module `set_original_component` resolves to — it's
invisible to this wrapper, so no DeiT-specific branching is needed here.

Verified against the real generalized_components/base.py (not extrapolated):
`GeneralizedComponent.forward(*args, **kwargs)` already does exactly what we need
— it hooks the input via `self.hook_in`, casts it to match the wrapped module's
own parameter dtype (equivalent to HF's own `pixel_values.to(expected_dtype)`
"kept for BC" cast in ViTModel.forward — both land on the same dtype in the
overwhelming common case of a non-mixed-precision checkpoint), calls
`self.original_component(*args, **kwargs)`, and hooks the output via
`self.hook_out`. So this subclass only needs to guarantee `pixel_values` reaches
`super().forward()` *positionally* — the base class's own kwarg-name sniffing
list (`input`, `hidden_states`, `input_ids`, `query_input`, `x`, `inputs_embeds`)
doesn't include `pixel_values`, so if this component were ever called
all-keyword (`self.embeddings(pixel_values=x, ...)`, unlike the two HF call
sites we've confirmed) the base class's hook_in would silently never fire.
Re-emitting positionally here closes that gap regardless of how *we* were
called.
"""

from typing import Any, Optional

import torch
from torch import Tensor

from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent


class VisionEmbeddingsBridge(GeneralizedComponent):
    """Bridge for ViTEmbeddings / DeiTEmbeddings.

    Point `name=` at the embeddings module directly, e.g.:
        VisionEmbeddingsBridge(name="embeddings")        # bare ViTModel/DeiTModel
        VisionEmbeddingsBridge(name="vit.embeddings")     # ViTForImageClassification
        VisionEmbeddingsBridge(name="deit.embeddings")    # DeiTForImageClassification
    """

    def forward(
        self,
        pixel_values: Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tensor:
        return super().forward(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )
