"""Analysis tools for TransformerLens.

This subpackage collects high-level, single-call interpretability analyses that
sit on top of the hook/cache system. Model support is documented per tool;
new analyses may target the ``TransformerBridge`` API exclusively.

Tools:
    - direct_logit_attribution: Direct Logit Attribution (DLA) over components,
      layers, or attention heads.
    - direct_path_patching: Direct path patching for head-to-head circuit
      analysis.
    - jacobian_lens: The Jacobian lens (J-lens) — per-layer causal transport to
      the output vocabulary basis, with loading of published lens artifacts,
      native fitting, readouts, interventions, and J-space sparse decomposition.
    - sparse_probing: Leakage-safe k-sparse linear probing over activation
      tensors (model-free, dependency-free).
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
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
    JSpaceOccupancy,
    JSpaceVarianceProfile,
    estimate_occupancy,
    get_sparse_decomposition,
)
from transformer_lens.tools.analysis.sparse_probing import (
    SparseProbeMetrics,
    SparseProbeResult,
    SparseSweepResult,
    fit_sparse_probe,
    sweep_sparse_probe,
)

__all__ = [
    "DirectLogitAttribution",
    "JSpaceDecomposition",
    "JSpaceOccupancy",
    "JSpaceVarianceProfile",
    "JacobianLens",
    "JacobianLensReadout",
    "SparseProbeMetrics",
    "SparseProbeResult",
    "SparseSweepResult",
    "direct_logit_attribution",
    "estimate_occupancy",
    "fit_sparse_probe",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "get_sparse_decomposition",
    "sweep_sparse_probe",
]
