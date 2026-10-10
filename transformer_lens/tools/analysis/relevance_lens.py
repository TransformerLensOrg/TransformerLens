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
    _RelevanceRuleCoverageEntry,
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

# Coverage schema changes do not change backward-rule semantics.
_RULE_COVERAGE_SCHEMA_VERSION = 1

# The rule set the relevance estimator installs: residual-stream norms, gated
# MLP activations, and multiplicative MLP gates. Attention is deliberately
# excluded (the AH-rule is out of scope).
RELEVANCE_RULES = RelevanceRules(
    normalization=True,
    activation=True,
    multiplicative_gate=True,
)

# Names of the rule kinds the estimator requests, derived from the rule set so
# the request cannot drift from what the fit attempts to install.
_REQUESTED_RULE_NAMES: Tuple[str, ...] = tuple(
    name
    for name in ("normalization", "activation", "multiplicative_gate")
    if getattr(RELEVANCE_RULES, name)
)


def _installed_rule_names(coverage: RelevanceRuleCoverage) -> List[str]:
    """Rule kinds that actually shaped the matrices, in request order."""
    installed_kinds = {entry.kind for entry in coverage.installed}
    return [kind for kind in _REQUESTED_RULE_NAMES if kind in installed_kinds]


def _decode_rule_coverage(recorded: Any) -> RelevanceRuleCoverage:
    """Validate kind-aware coverage without inferring kinds from legacy paths."""
    if not isinstance(recorded, dict):
        raise ValueError("rule_coverage must be a kind-aware coverage dictionary")
    if "schema_version" not in recorded:
        raise ValueError(
            "rule_coverage requires schema_version and kind/path entries; "
            "legacy path-only coverage is ambiguous, so refit the relevance lens"
        )
    schema_version = recorded["schema_version"]
    if type(schema_version) is not int or schema_version != _RULE_COVERAGE_SCHEMA_VERSION:
        raise ValueError(f"unsupported rule_coverage schema_version: {schema_version!r}")
    if set(recorded) != {"schema_version", "installed", "skipped"}:
        raise ValueError("rule_coverage requires schema_version, installed, and skipped fields")

    def decode_entries(field: str) -> Tuple[_RelevanceRuleCoverageEntry, ...]:
        raw_entries = recorded[field]
        if not isinstance(raw_entries, (list, tuple)):
            raise ValueError(f"rule_coverage.{field} must be a sequence of kind/path entries")
        entries = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict) or set(raw_entry) != {"kind", "path"}:
                raise ValueError(f"rule_coverage.{field} entries must contain kind and path")
            kind = raw_entry["kind"]
            path = raw_entry["path"]
            if not isinstance(kind, str) or kind not in _REQUESTED_RULE_NAMES:
                raise ValueError(f"rule_coverage.{field} records unknown rule kind {kind!r}")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"rule_coverage.{field} paths must be non-empty strings")
            entries.append(_RelevanceRuleCoverageEntry(kind=kind, path=path))
        if len(set(entries)) != len(entries):
            raise ValueError(f"rule_coverage.{field} contains duplicate kind/path entries")
        return tuple(entries)

    coverage = RelevanceRuleCoverage(
        installed=decode_entries("installed"), skipped=decode_entries("skipped")
    )
    if set(coverage.installed) & set(coverage.skipped):
        raise ValueError("rule_coverage cannot install and skip the same kind/path entry")
    return coverage


def _encode_rule_coverage(coverage: RelevanceRuleCoverage) -> Dict[str, Any]:
    """Encode coverage as safe, versioned artifact metadata."""
    recorded = {
        "schema_version": _RULE_COVERAGE_SCHEMA_VERSION,
        "installed": [{"kind": entry.kind, "path": entry.path} for entry in coverage.installed],
        "skipped": [{"kind": entry.kind, "path": entry.path} for entry in coverage.skipped],
    }
    _decode_rule_coverage(recorded)
    return recorded


def _coverage_from_metadata(metadata: Dict[str, Any]) -> Optional[RelevanceRuleCoverage]:
    """Restore optional coverage, rejecting malformed or ambiguous records."""
    if "rule_coverage" not in metadata:
        return None
    return _decode_rule_coverage(metadata["rule_coverage"])


def _coverage_labels(entries: Sequence[_RelevanceRuleCoverageEntry]) -> List[str]:
    """Distinguish rule kinds sharing a component path in diagnostics."""
    return [f"{entry.kind}: {entry.path}" for entry in entries]


class RelevanceLens(JacobianLens):
    """A Jacobian lens whose transport matrices use relevance-rule backward.

    Inherits the full Jacobian lens surface -- transport, readout, vocabulary
    vectors and dictionary, sparse decomposition, and the steering, ablation,
    and swap interventions -- unchanged. Only :meth:`fit` differs, and it
    records the estimator identity and the rule configuration that produced the
    matrices. Loading requires explicit ``estimator="relevance_lens"`` metadata;
    unlabelled artifacts and Jacobian running-sum checkpoints are refused.

    Attributes:
        rule_coverage: Rule-kind and canonical-path pairs installed versus
            skipped during the fit.
        relevance_rule_version: Version of the rule semantics used.
        enabled_rules: Names of the ``RelevanceRules`` fields that were enabled.
    """

    # Relevance artifacts are not published in the Jacobian lens registry, so
    # short model names are refused rather than resolving to a Jacobian lens.
    _uses_artifact_registry = False

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
        if metadata is not None and "estimator" in metadata:
            self._validate_artifact_metadata("RelevanceLens constructor", metadata)
        super().__init__(jacobians, n_prompts=n_prompts, d_model=d_model, metadata=metadata)
        self.estimator = ESTIMATOR_RELEVANCE
        # Recorded semantics take precedence over defaults when restoring an artifact.
        if "relevance_rule_version" in self.metadata:
            recorded_version = self.metadata["relevance_rule_version"]
            if not isinstance(recorded_version, int) or isinstance(recorded_version, bool):
                raise ValueError("relevance_rule_version provenance must be an integer")
            relevance_rule_version = recorded_version
        if "enabled_rules" in self.metadata:
            recorded_rules = self.metadata["enabled_rules"]
            if not isinstance(recorded_rules, (list, tuple)) or not all(
                isinstance(name, str) for name in recorded_rules
            ):
                raise ValueError("enabled_rules provenance must be a list or tuple of strings")
            enabled_rules = recorded_rules
        self.relevance_rule_version = int(relevance_rule_version)
        self.enabled_rules: List[str] = list(enabled_rules)
        recorded_coverage = _coverage_from_metadata(self.metadata)
        if rule_coverage is None:
            rule_coverage = recorded_coverage
        elif recorded_coverage is not None and recorded_coverage != rule_coverage:
            raise ValueError("rule_coverage does not match recorded coverage provenance")
        self.rule_coverage = rule_coverage
        # Normalize recorded configuration so attributes and saved provenance agree.
        self.metadata["estimator"] = ESTIMATOR_RELEVANCE
        self.metadata["relevance_rule_version"] = self.relevance_rule_version
        self.metadata["enabled_rules"] = list(self.enabled_rules)
        if self.rule_coverage is not None:
            self.metadata["rule_coverage"] = _encode_rule_coverage(self.rule_coverage)

    @classmethod
    def _validate_artifact_metadata(cls, path: str, metadata: Any) -> None:
        """Require original relevance provenance before construction can stamp it."""
        recorded_estimator = metadata.get("estimator") if isinstance(metadata, dict) else None
        if not isinstance(recorded_estimator, str) or recorded_estimator != ESTIMATOR_RELEVANCE:
            raise ValueError(
                f"{path} requires estimator provenance {ESTIMATOR_RELEVANCE!r}; "
                f"recorded estimator is {recorded_estimator!r}. "
                "Unlabelled or foreign artifacts cannot be converted to relevance lenses."
            )

    @classmethod
    def _validate_checkpoint_payload(cls, path: str, payload: Dict[str, Any]) -> None:
        """Refuse the Jacobian-specific running-sum conversion path."""
        raise ValueError(
            f"{path} is a running-sum checkpoint; RelevanceLens requires an artifact "
            "with explicit relevance estimator provenance, not a Jacobian checkpoint."
        )

    @classmethod
    def _validate_merge_inputs(cls, lenses: Sequence[JacobianLens]) -> None:
        """Require relevance shards even when all inputs share ordinary provenance."""
        for index, lens in enumerate(lenses):
            if not isinstance(lens, RelevanceLens):
                raise ValueError(
                    "RelevanceLens.merge() requires relevance lens shards with relevance "
                    f"estimator provenance; shard {index} is {type(lens).__name__}."
                )
            cls._validate_artifact_metadata(f"merge shard {index}", lens.metadata)

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

        Coverage is checked before any fitting work. A model that can honor no
        rule is refused, since the fit would otherwise return ordinary Jacobian
        matrices labelled as a relevance lens. A model that honors some rules
        but skips others is fitted with a warning, and the artifact records only
        the rules that actually shaped the matrices.

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
                or layer indices, if no prompt was long enough to fit on, or if
                no relevance rule could be installed on the model.
            RelevanceRuleUnsupportedError: If a mount that is expected to honor
                a requested rule cannot, such as a gated MLP whose activation
                the Identity-rule does not hold for.

        Warns:
            UserWarning: If some rule mounts were skipped, so the matrices mix
                rule-modified and ordinary gradients.
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
            if not coverage.installed:
                raise ValueError(
                    "no relevance rule could be installed on this model, so the "
                    "fit would return ordinary Jacobian matrices labelled as a "
                    "relevance lens. Skipped mounts: "
                    f"{_coverage_labels(coverage.skipped)}. The relevance estimator needs "
                    "residual-stream norms on the native-autograd path and gated "
                    "MLPs with GELU or SiLU activations; use JacobianLens for "
                    "models without them."
                )
            if coverage.skipped:
                warnings.warn(
                    "some relevance-rule mounts were skipped, so the transport "
                    "matrices mix rule-modified and ordinary gradients. Installed "
                    f"rules: {_installed_rule_names(coverage)}. Skipped mounts: "
                    f"{_coverage_labels(coverage.skipped)}.",
                    UserWarning,
                    stacklevel=2,
                )
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
            "enabled_rules": _installed_rule_names(coverage),
            "rule_coverage": _encode_rule_coverage(coverage),
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
            enabled_rules=_installed_rule_names(coverage),
        )
