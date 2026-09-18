"""Pooler bridge component.

This module contains the bridge component for [CLS] pooling heads.
"""

from __future__ import annotations

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class BertPoolerBridge(GeneralizedComponent):
    """Bridge component for BERT's [CLS] pooler.

    Wraps the whole pooler, so ``hook_out`` carries the post-tanh pooled
    ``[CLS]`` vector rather than the pre-activation projection — the tensor
    ``HookedEncoder``'s ``BertPooler`` exposes as ``hook_pooler_out``, which is
    aliased here so code migrated from the legacy stack keeps working.
    """

    hook_aliases = {"hook_pooler_out": "hook_out"}
