"""Analysis tools for TransformerLens.

This subpackage collects high-level, single-call interpretability analyses that
sit on top of the hook/cache system. Model support is documented per tool;
new analyses may target the ``TransformerBridge`` API exclusively.

Tools:
    - attribution_patching: Attribution patching (gradient-linearized activation
      patching) over residual-stream nodes — typed computational graph, a
      names-filtered manual-backward gradient cache, and signed node scores. Edge
      scoring (EAP), integrated gradients (EAP-IG), and faithfulness land in
      follow-on PRs.
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
    - svd_circuits: Per-head QK/OV singular-vector decomposition with a
      degeneracy guard, OV vocab and logit readout, per-position activation
      projection onto singular directions, and a mandatory causal patch gate
      that reconstructs a head's output onto a chosen singular subspace.
"""

from transformer_lens.tools.analysis.attribution_patching import (
    AttributionResult,
    EdgeAttributionConfig,
    Node,
    attribution_patch,
)
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
from transformer_lens.tools.analysis.svd_circuits import (
    ActivationProjection,
    DegenerateDirectionError,
    HeadDecomposition,
    HeadSVD,
    LogitSignature,
    PatchResult,
    RankReportRow,
    decompose_head,
    logit_signature,
    patch_along_directions,
    project_activations,
    vocab_readout,
)

__all__ = [
    "ActivationProjection",
    "AttentionHeadRef",
    "AttributionResult",
    "BackwardLens",
    "BackwardLensLayerResult",
    "BackwardLensMatrixResult",
    "BackwardLensResult",
    "CoordinatePatch",
    "DegenerateDirectionError",
    "DirectLogitAttribution",
    "EdgeAttributionConfig",
    "HeadAffinityPair",
    "HeadAffinityResult",
    "HeadDecomposition",
    "HeadSVD",
    "JSpaceDecomposition",
    "JSpaceOccupancy",
    "JSpaceVarianceProfile",
    "JacobianLens",
    "JacobianLensReadout",
    "LinearGradientFactors",
    "LogitSignature",
    "Node",
    "PatchResult",
    "ProjectedFactor",
    "ProjectionKernelResult",
    "RandomSubspaceReference",
    "RankReportRow",
    "SubspaceBasis",
    "VocabularyRanking",
    "WeightLayout",
    "attention_head_subspace_affinity",
    "attribution_patch",
    "decompose_head",
    "direct_logit_attribution",
    "estimate_occupancy",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "get_sparse_decomposition",
    "logit_signature",
    "orthonormal_subspace",
    "patch_along_directions",
    "project_activations",
    "projection_kernel",
    "random_projection_kernel_moments",
    "solve_coordinate_patch",
    "solve_coordinate_patch_positions",
    "vocab_readout",
]
