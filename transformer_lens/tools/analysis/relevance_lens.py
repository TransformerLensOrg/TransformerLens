"""Relevance Lens (R-lens): a RelP/LRP-rule estimator for the Jacobian lens.

The relevance lens keeps the Jacobian lens's fitting loop, transport matrices,
readout, vocabulary vectors, sparse decomposition, and interventions unchanged.
Only the backward pass differs: each per-layer transport matrix is estimated
with a scoped set of Layer-wise Relevance Propagation rules installed on the
model's residual-stream norms and gated MLPs, instead of the ordinary
vector-Jacobian product. The forward pass is bit-identical to the model's native
forward, so the two estimators differ solely in the backward semantics they
select.

Three rules are applied, matching the RelP reference:

1. **LN-rule** (residual-stream norms): preserve the normalization forward value
   while treating its denominator as a constant in the backward pass.
2. **Identity-rule** (GELU/SiLU activations): express ``f(x) = x * phi(x)`` and
   detach ``phi(x)``, so the local VJP is ``grad_out * phi(x)``.
3. **Half-rule** (multiplicative MLP gates): preserve the forward product
   ``u * v`` while halving each branch's ordinary product-rule gradient.

Linear layers and attention retain ordinary autograd. The attention-specific
AH-rule is out of scope.

Warning:
    Like the Jacobian lens, the relevance lens requires a freshly booted raw
    ``TransformerBridge.boot_transformers`` model. Compatibility mode and direct
    ``process_weights`` calls change the residual basis and are refused rather
    than returning silently wrong readouts.

Example::

    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import RelevanceLens

    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    lens = RelevanceLens.fit(model, prompts, corpus="pile-10k:fixed-manifest")
    result = lens.readout(model, "The Eiffel Tower is in the city of")
"""

from __future__ import annotations

import warnings
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRuleCoverage,
    RelevanceRules,
    use_relevance_rules,
)
from transformer_lens.tools.analysis._model_state import require_eval_mode
from transformer_lens.tools.analysis.jacobian_lens import (
    DEFAULT_SKIP_FIRST_POSITIONS,
    JacobianLens,
    _fit_transport_matrices,
    _get_model_revision,
    _normalize_layer,
    _ordinary_vjp,
    _require_raw_bridge,
    _validate_metadata,
)

# Estimator identity recorded on every relevance lens. Distinct from the
# ordinary Jacobian estimator so merge() refuses to combine the two.
ESTIMATOR_RELEVANCE = "relevance_lens"

# Version of the rule semantics above. Bump when a rule's backward VJP changes,
# so artifacts fitted under different semantics cannot be merged silently.
RELEVANCE_RULE_VERSION = 1

# The rule set the relevance estimator installs: residual-stream norms, gated
# MLP activations, and multiplicative MLP gates. Attention is deliberately
# excluded (the AH-rule is out of scope).
RELEVANCE_RULES = RelevanceRules(
    normalization=True,
    activation=True,
    multiplicative_gate=True,
)

# Names of the enabled rule kinds, derived from the rule set so the recorded
# provenance cannot drift from what the fit actually installs.
_ENABLED_RULE_NAMES: Tuple[str, ...] = tuple(
    name
    for name in ("normalization", "activation", "multiplicative_gate")
    if getattr(RELEVANCE_RULES, name)
)


class RelevanceLens(JacobianLens):
    """A Jacobian lens whose transport matrices use relevance-rule backward.

    Inherits the full Jacobian lens surface -- transport, readout, vocabulary
    vectors and dictionary, sparse decomposition, and the steering, ablation,
    and swap interventions -- unchanged. Only :meth:`fit` differs, and it
    records the estimator identity and the rule configuration that produced the
    matrices.

    Attributes:
        rule_coverage: Which canonical mounts the rule scope installed versus
            skipped during the fit.
        relevance_rule_version: Version of the rule semantics used.
        enabled_rules: Names of the ``RelevanceRules`` fields that were enabled.
    """

    def __init__(
        self,
        jacobians: Dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
        metadata: Optional[Dict[str, Any]] = None,
        rule_coverage: Optional[RelevanceRuleCoverage] = None,
        relevance_rule_version: int = RELEVANCE_RULE_VERSION,
        enabled_rules: Sequence[str] = (),
    ) -> None:
        super().__init__(jacobians, n_prompts=n_prompts, d_model=d_model, metadata=metadata)
        self.estimator = ESTIMATOR_RELEVANCE
        self.rule_coverage = rule_coverage
        self.relevance_rule_version = int(relevance_rule_version)
        self.enabled_rules: List[str] = list(enabled_rules)

    @classmethod
    def fit(
        cls,
        model: Any,
        prompts: Sequence[str],
        *,
        corpus: str,
        source_layers: Optional[Sequence[int]] = None,
        dim_batch: int = 8,
        max_seq_len: int = 128,
        skip_first_positions: int = DEFAULT_SKIP_FIRST_POSITIONS,
        show_progress: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RelevanceLens":
        """Fit a relevance lens on a raw ``TransformerBridge``.

        Shares the Jacobian lens's estimator-independent drive loop and differs
        only in the backward step: the three relevance rules are installed on
        the model for the duration of the fit, so each transport matrix is
        estimated with rule-modified backward semantics. The forward pass is
        unchanged, and the rule scope is released before this method returns.

        Args:
            model: A raw ``TransformerBridge``. Model parameters are temporarily
                frozen during fitting and restored after. The model and all of
                its submodules must be in evaluation mode.
            prompts: Prompt strings. Prompts too short to contain a valid
                position (``seq_len <= skip_first_positions + 1``) are skipped
                with a warning and do not count toward ``n_prompts``.
            corpus: Stable identifier for the prompt corpus or slice, recorded
                in artifact provenance.
            source_layers: Layers to fit. Defaults to every layer below the
                final layer. Negative indices count from ``n_layers``.
            dim_batch: Output dimensions per backward pass.
            max_seq_len: Prompts are truncated to this many tokens.
            skip_first_positions: Leading positions excluded from the source
                average.
            show_progress: Show a tqdm progress bar over prompts.
            metadata: Extra provenance merged into :attr:`metadata`.

        Returns:
            The fitted :class:`RelevanceLens`.

        Raises:
            TypeError: If model is not a ``TransformerBridge``.
            ValueError: On compatibility mode, training mode, invalid provenance
                or layer indices, or if no prompt was long enough to fit on.
        """
        _require_raw_bridge(model, estimator=cls.__name__)
        require_eval_mode(model, operation=f"{cls.__name__}.fit()")
        if not isinstance(corpus, str) or not corpus.strip():
            raise ValueError("corpus must be a non-empty provenance identifier")
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        resolved_target = n_layers - 1
        if source_layers is None:
            resolved_sources = list(range(resolved_target))
        else:
            resolved_sources = sorted(
                {_normalize_layer(layer, n_layers) for layer in source_layers}
            )
        if not resolved_sources:
            raise ValueError("source_layers is empty")
        if resolved_sources[-1] >= resolved_target:
            raise ValueError(
                f"every source layer must be below target_layer={resolved_target}; "
                f"got {resolved_sources}"
            )
        if dim_batch < 1:
            raise ValueError(f"dim_batch must be >= 1, got {dim_batch}")
        if skip_first_positions < 0:
            raise ValueError(f"skip_first_positions must be >= 0, got {skip_first_positions}")
        fit_dtype = model.W_U.dtype
        if fit_dtype in (torch.float16, torch.bfloat16):
            warnings.warn(
                f"fitting in {fit_dtype} accumulates transport gradients at reduced "
                "precision; use a float32 TransformerBridge for the highest-fidelity "
                "fit",
                UserWarning,
                stacklevel=2,
            )

        with use_relevance_rules(model, RELEVANCE_RULES) as coverage:
            transport_matrices, n_done = _fit_transport_matrices(
                model,
                prompts,
                source_layers=resolved_sources,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first_positions=skip_first_positions,
                show_progress=show_progress,
                backward_provider=_ordinary_vjp,
            )

        fit_metadata: Dict[str, Any] = {
            "model_name": getattr(model.cfg, "model_name", None),
            "model_revision": _get_model_revision(model),
            "transformer_lens_version": version("transformer-lens"),
            "model_system": "TransformerBridge",
            "estimator": ESTIMATOR_RELEVANCE,
            "processing": {
                "compatibility_mode": False,
                "weight_basis": "raw_huggingface",
            },
            "hook_convention": "blocks.{layer}.hook_out",
            "corpus": corpus,
            "n_prompts": n_done,
            "fit_dtype": str(fit_dtype).removeprefix("torch."),
            "target_layer": resolved_target,
            "dim_batch": dim_batch,
            "max_seq_len": max_seq_len,
            "skip_first_positions": skip_first_positions,
            "transformer_lens_fit": True,
            "relevance_rule_version": RELEVANCE_RULE_VERSION,
            "enabled_rules": list(_ENABLED_RULE_NAMES),
            "rule_coverage": {
                "installed": list(coverage.installed),
                "skipped": list(coverage.skipped),
            },
        }
        reserved = sorted(set(fit_metadata).intersection(metadata or {}))
        if reserved:
            raise ValueError(f"metadata cannot override fit provenance keys: {reserved}")
        full_metadata = dict(metadata or {})
        full_metadata.update(fit_metadata)
        _validate_metadata(full_metadata)
        return cls(
            transport_matrices,
            n_prompts=n_done,
            d_model=d_model,
            metadata=full_metadata,
            rule_coverage=coverage,
            relevance_rule_version=RELEVANCE_RULE_VERSION,
            enabled_rules=_ENABLED_RULE_NAMES,
        )
