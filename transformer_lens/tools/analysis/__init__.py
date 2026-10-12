"""Analysis tools for TransformerLens.

This subpackage collects high-level, single-call interpretability analyses that
sit on top of the hook/cache system. Model support is documented per tool;
new analyses may target the ``TransformerBridge`` API exclusively.

Tools:
    - attribution_patching: Node and edge attribution patching (EAP) and faithfulness.
    - backward_lens: GPT-2 MLP weight-gradient factors projected into vocabulary space.
    - direct_logit_attribution: Direct Logit Attribution over components, layers, or heads.
    - direct_path_patching: Direct path patching for head-to-head circuit analysis.
    - jacobian_lens: Jacobian lens loading, fitting, readout, decomposition, and interventions.
    - jacobian_lens_causal_swap_benchmark: Causal coordinate-swap benchmark for J-lens patches.
    - logit_lens: Logit lens and the ``logit_readout`` kernel for any ``d_model`` vector.
    - projection_kernel: Subspace overlap and attention-head OQ/OK/OV affinity.
    - relevance_lens: A Jacobian lens whose transport is fitted with scoped LRP rules.
    - sparse_probing: Leakage-safe k-sparse binary probes with null controls.
    - svd_circuits: Per-head QK/OV SVD, vocabulary readout, and singular-subspace patching.
"""

from transformer_lens.tools.analysis.attribution_patching import (
    AttributionResult,
    EdgeAttributionConfig,
    FaithfulnessConfig,
    FaithfulnessResult,
    Node,
    attribution_patch,
    faithfulness,
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

# The benchmark's _cli module stays unimported here so ``python -m`` runs it once.
from transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark import (
    BenchmarkCorpus,
    FunctionSpec,
    corpus_definition,
    load_artifact,
    run_causal_swap_benchmark,
    select_displacement_matched_control_token,
    success_rate_ci,
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
from transformer_lens.tools.analysis.logit_lens import (
    LogitLensResult,
    LogitReadout,
    logit_lens,
    logit_readout,
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
from transformer_lens.tools.analysis.relevance_lens import RelevanceLens
from transformer_lens.tools.analysis.sparse_probing import (
    SparseProbeControl,
    SparseProbeConvergenceError,
    SparseProbeMetrics,
    SparseProbeRejection,
    SparseProbeResult,
    SparseProbeSweep,
    fit_sparse_probe,
    sweep_sparse_probe,
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
    "BenchmarkCorpus",
    "CoordinatePatch",
    "DegenerateDirectionError",
    "DirectLogitAttribution",
    "EdgeAttributionConfig",
    "FaithfulnessConfig",
    "FaithfulnessResult",
    "FunctionSpec",
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
    "LogitLensResult",
    "LogitReadout",
    "LogitSignature",
    "Node",
    "PatchResult",
    "ProjectedFactor",
    "ProjectionKernelResult",
    "RandomSubspaceReference",
    "RankReportRow",
    "RelevanceLens",
    "SparseProbeControl",
    "SparseProbeConvergenceError",
    "SparseProbeMetrics",
    "SparseProbeRejection",
    "SparseProbeResult",
    "SparseProbeSweep",
    "SubspaceBasis",
    "VocabularyRanking",
    "WeightLayout",
    "attention_head_subspace_affinity",
    "attribution_patch",
    "corpus_definition",
    "decompose_head",
    "direct_logit_attribution",
    "estimate_occupancy",
    "faithfulness",
    "fit_sparse_probe",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "get_sparse_decomposition",
    "load_artifact",
    "logit_lens",
    "logit_readout",
    "logit_signature",
    "orthonormal_subspace",
    "patch_along_directions",
    "project_activations",
    "projection_kernel",
    "random_projection_kernel_moments",
    "run_causal_swap_benchmark",
    "select_displacement_matched_control_token",
    "solve_coordinate_patch",
    "solve_coordinate_patch_positions",
    "success_rate_ci",
    "sweep_sparse_probe",
    "vocab_readout",
]
