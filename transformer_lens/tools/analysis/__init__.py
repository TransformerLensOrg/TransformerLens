"""Analysis tools for TransformerLens.

This subpackage collects high-level, single-call interpretability analyses that
sit on top of the hook/cache system. Model support is documented per tool;
new analyses may target the ``TransformerBridge`` API exclusively.

Tools:
    - backward_lens: GPT-2 MLP weight-gradient factors projected into vocabulary
      space with explicit raw-gradient sign semantics.
    - direct_logit_attribution: Direct Logit Attribution (DLA) over components,
      layers, or attention heads.
    - direct_path_patching: Direct path patching for head-to-head circuit
      analysis.
    - jacobian_lens: The Jacobian lens (J-lens) — per-layer causal transport to
      the output vocabulary basis, with loading of published lens artifacts,
      native fitting, readouts, interventions, J-space sparse decomposition, and
      anchored coordinate patching (offline and dynamic/hooked).
    - projection_kernel: Basis-invariant subspace overlap and TransformerBridge
      attention-head OQ/OK/OV affinity.
"""

from transformer_lens.tools.analysis.backward_lens import (
    BackwardLens,
    BackwardLensLayerResult,
    BackwardLensMatrixResult,
    BackwardLensResult,
    LinearGradientFactors,
    ProjectedFactor,
    VocabularyRanking,
    WeightLayout,
)
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
from transformer_lens.tools.analysis.jacobian_lens_coordinate_patch import (
    CoordinatePatch,
    solve_coordinate_patch,
    solve_coordinate_patch_positions,
)
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
    JSpaceOccupancy,
    JSpaceVarianceProfile,
    estimate_occupancy,
    get_sparse_decomposition,
)
from transformer_lens.tools.analysis.projection_kernel import (
    AttentionHeadRef,
    HeadAffinityPair,
    HeadAffinityResult,
    ProjectionKernelResult,
    RandomSubspaceReference,
    SubspaceBasis,
    attention_head_subspace_affinity,
    orthonormal_subspace,
    projection_kernel,
    random_projection_kernel_moments,
)

__all__ = [
    "AttentionHeadRef",
    "BackwardLens",
    "BackwardLensLayerResult",
    "BackwardLensMatrixResult",
    "BackwardLensResult",
    "CoordinatePatch",
    "DirectLogitAttribution",
    "HeadAffinityPair",
    "HeadAffinityResult",
    "JSpaceDecomposition",
    "JSpaceOccupancy",
    "JSpaceVarianceProfile",
    "JacobianLens",
    "JacobianLensReadout",
    "LinearGradientFactors",
    "ProjectedFactor",
    "ProjectionKernelResult",
    "RandomSubspaceReference",
    "SubspaceBasis",
    "VocabularyRanking",
    "WeightLayout",
    "attention_head_subspace_affinity",
    "direct_logit_attribution",
    "estimate_occupancy",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "get_sparse_decomposition",
    "orthonormal_subspace",
    "projection_kernel",
    "random_projection_kernel_moments",
    "solve_coordinate_patch",
    "solve_coordinate_patch_positions",
]
