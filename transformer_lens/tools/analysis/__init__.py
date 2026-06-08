"""Analysis tools for TransformerLens.

This subpackage collects high-level, single-call interpretability analyses that
sit on top of the hook/cache system. They work with both ``HookedTransformer``
and the newer ``TransformerBridge`` (the two share the ``ActivationCache`` API).

Tools:
    - direct_logit_attribution: Direct Logit Attribution (DLA) over components,
      layers, or attention heads.
"""

from transformer_lens.tools.analysis.direct_logit_attribution import (
    DirectLogitAttribution,
    direct_logit_attribution,
)

__all__ = ["DirectLogitAttribution", "direct_logit_attribution"]
