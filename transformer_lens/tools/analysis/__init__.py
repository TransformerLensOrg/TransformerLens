"""Analysis tools for TransformerLens.

This subpackage collects high-level, single-call interpretability analyses that
sit on top of the hook/cache system. Model support is documented per tool;
new analyses may target the ``TransformerBridge`` API exclusively.

Entry points and the classes they accept are exported here; result dataclasses stay in the
module that returns them.

Tools:
    - attribution_patching: Attribution patching (gradient-linearized activation
      patching) over residual-stream nodes and edges: typed computational graph,
      a names-filtered manual-backward gradient cache, and signed node and edge
      scores (EAP), plus forward-only ablate-outside faithfulness with corrupt
      or held-out mean replacements and circuit-relative class diagnostics.
      Integrated gradients (EAP-IG) are not implemented.
    - direct_logit_attribution: Direct Logit Attribution (DLA) over components,
      layers, or attention heads.
    - get_act_patch_direct_path, get_act_patch_direct_path_all_sources:
      re-exported from ``transformer_lens.patching``; exact direct path
      patching for head-to-head circuit analysis.
    - interchange: Interchange interventions on linear subspaces: set a base's
      activation inside a subspace to its value on a source input, with
      difference-of-means, probe, random and neuron bases, a norm-matched
      placebo, batched gradient-free evaluation, distributed alignment search
      (DAS) to learn the subspace, and DAS's trained controls (arbitrary labels
      and a randomly initialised model).
    - jacobian_lens: The Jacobian lens (J-lens) — per-layer causal transport to
      the output vocabulary basis, with loading of published lens artifacts,
      native fitting, readouts, interventions, J-space sparse decomposition, and
      anchored coordinate patching (offline and dynamic/hooked).
    - logit_lens: One-call logit lens over the accumulated residual stream and
      the ``logit_readout`` kernel that reads any ``d_model`` vector through the
      real final norm, unembedding and adapter output transform, chunked over
      the vocabulary with target ranks, top-k and vocabulary subsets.
    - projection_kernel: Basis-invariant subspace overlap and TransformerBridge
      attention-head OQ/OK/OV affinity.
    - sparse_probing: Leakage-safe k-sparse binary probes over supplied
      activation tensors, with train-only selection and raw null controls.
"""

import warnings
from typing import TYPE_CHECKING, Any

from transformer_lens.patching import (
    get_act_patch_direct_path,
    get_act_patch_direct_path_all_sources,
)
from transformer_lens.tools.analysis.attribution_patching import (
    EdgeAttributionConfig,
    FaithfulnessConfig,
    Node,
    attribution_patch,
    faithfulness,
)
from transformer_lens.tools.analysis.direct_logit_attribution import (
    direct_logit_attribution,
)
from transformer_lens.tools.analysis.interchange import (
    LearnedSubspace,
    NormMatchedPlacebo,
    SubspaceInterchange,
    arbitrary_targets,
    collect_activations,
    fit_das,
    intervened_logits,
    mean_difference_basis,
    neuron_basis,
    probe_basis,
    random_basis,
    random_init_bridge,
    random_neuron_basis,
    reinitialized,
    run_with_intervention,
)
from transformer_lens.tools.analysis.jacobian_lens import JacobianLens
from transformer_lens.tools.analysis.jacobian_lens_coordinate_patch import (
    solve_coordinate_patch,
)
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
    get_sparse_decomposition,
)
from transformer_lens.tools.analysis.logit_lens import logit_lens, logit_readout
from transformer_lens.tools.analysis.projection_kernel import (
    SubspaceBasis,
    attention_head_subspace_affinity,
    orthonormal_subspace,
    projection_kernel,
    random_projection_kernel_moments,
)
from transformer_lens.tools.analysis.sparse_probing import (
    SparseProbeConvergenceError,
    fit_sparse_probe,
    sweep_sparse_probe,
)

__all__ = [
    "EdgeAttributionConfig",
    "FaithfulnessConfig",
    "JSpaceDecomposition",
    "JacobianLens",
    "LearnedSubspace",
    "Node",
    "NormMatchedPlacebo",
    "SparseProbeConvergenceError",
    "SubspaceBasis",
    "SubspaceInterchange",
    "arbitrary_targets",
    "attention_head_subspace_affinity",
    "attribution_patch",
    "collect_activations",
    "direct_logit_attribution",
    "faithfulness",
    "fit_das",
    "fit_sparse_probe",
    "get_act_patch_direct_path",
    "get_act_patch_direct_path_all_sources",
    "get_sparse_decomposition",
    "intervened_logits",
    "logit_lens",
    "logit_readout",
    "mean_difference_basis",
    "neuron_basis",
    "orthonormal_subspace",
    "probe_basis",
    "projection_kernel",
    "random_basis",
    "random_init_bridge",
    "random_neuron_basis",
    "random_projection_kernel_moments",
    "reinitialized",
    "run_with_intervention",
    "solve_coordinate_patch",
    "sweep_sparse_probe",
]

# Moved to transformer_lens.utilities with unchanged results.
_MOVED_TO_UTILITIES = ("success_rate_ci",)

if TYPE_CHECKING:
    from transformer_lens.utilities import success_rate_ci
else:
    # Hidden from type checkers, which would otherwise type every unknown name here as Any.
    def __getattr__(name: str) -> Any:
        if name not in _MOVED_TO_UTILITIES:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        warnings.warn(
            f"{__name__}.{name} is deprecated; import it from transformer_lens.utilities.",
            DeprecationWarning,
            stacklevel=2,
        )
        from transformer_lens import utilities

        return getattr(utilities, name)
