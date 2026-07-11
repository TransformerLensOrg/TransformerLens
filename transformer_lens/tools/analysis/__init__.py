"""Analysis tools for TransformerLens.

This subpackage collects high-level, single-call interpretability analyses that
sit on top of the hook/cache system. They work with both ``HookedTransformer``
and the newer ``TransformerBridge`` (the two share the ``ActivationCache`` API).

Tools:
    - direct_logit_attribution: Direct Logit Attribution (DLA) over components,
      layers, or attention heads.
    - direct_path_patching: Direct path patching for head-to-head circuit
      analysis.
    - jacobian_lens: The Jacobian lens (J-lens) — per-layer causal transport to
      the output vocabulary basis, with loading of published lens artifacts,
      native fitting, readouts, and interventions.
"""

from transformer_lens.tools.analysis.direct_logit_attribution import (
    DirectLogitAttribution,
    direct_logit_attribution,
)
from transformer_lens.tools.analysis.direct_path_patching import (
    get_act_patch_direct_path,
    get_act_patch_direct_path_all_sources,
)
from transformer_lens.tools.analysis.jacobian_lens import (
    JacobianLens,
    JacobianLensReadout,
)

__all__ = [
    "DirectLogitAttribution",
    "JacobianLens",
    "JacobianLensReadout",
    "direct_logit_attribution",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
]
