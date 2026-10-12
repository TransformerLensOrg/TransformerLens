"""Relevance Lens: a Jacobian lens fitted under RelP LN-, Identity- and Half-rule backward; the
forward pass is unchanged and attention keeps ordinary autograd."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRuleCoverage,
    RelevanceRules,
    _RelevanceRuleCoverageEntry,
    use_relevance_rules,
)
from transformer_lens.tools.analysis.jacobian_lens import (
    DEFAULT_SKIP_FIRST_POSITIONS,
    JacobianLens,
    _build_fit_metadata,
    _fit_transport_matrices,
    _ordinary_vjp,
    _prepare_fit,
)

# Distinct from the Jacobian estimator so merge() refuses to combine the two.
ESTIMATOR_RELEVANCE = "relevance_lens"

# Bump when a rule's backward VJP changes so artifacts of different semantics never merge.
RELEVANCE_RULE_VERSION = 1

# Coverage schema changes do not change backward-rule semantics.
_RULE_COVERAGE_SCHEMA_VERSION = 1

# Attention is excluded: the AH-rule is out of scope.
RELEVANCE_RULES = RelevanceRules(
    normalization=True,
    activation=True,
    multiplicative_gate=True,
)

# Derived from RELEVANCE_RULES so the request cannot drift from what the fit installs.
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
    """Jacobian lens fitted with relevance-rule backward; loading requires relevance provenance."""

    # Not in the Jacobian lens registry, so short model names must not resolve to a Jacobian lens.
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
        """Fit on a raw ``TransformerBridge`` with the relevance rules installed for the fit only;
        refuses a model on which no rule installs and warns on (and records) partial coverage."""
        options = _prepare_fit(
            model,
            operation=cls.__name__,
            estimator=ESTIMATOR_RELEVANCE,
            corpus=corpus,
            source_layers=source_layers,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first_positions=skip_first_positions,
            metadata=metadata,
            estimator_metadata_keys=("relevance_rule_version", "enabled_rules", "rule_coverage"),
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
                source_layers=list(options.source_layers),
                dim_batch=options.dim_batch,
                max_seq_len=options.max_seq_len,
                skip_first_positions=options.skip_first_positions,
                show_progress=show_progress,
                backward_provider=_ordinary_vjp,
            )

        return cls(
            transport_matrices,
            n_prompts=n_done,
            d_model=options.d_model,
            metadata=_build_fit_metadata(
                model,
                options,
                estimator=ESTIMATOR_RELEVANCE,
                n_prompts=n_done,
                metadata=metadata,
                estimator_metadata={
                    "relevance_rule_version": RELEVANCE_RULE_VERSION,
                    "enabled_rules": _installed_rule_names(coverage),
                    "rule_coverage": _encode_rule_coverage(coverage),
                },
            ),
            rule_coverage=coverage,
            relevance_rule_version=RELEVANCE_RULE_VERSION,
            enabled_rules=_installed_rule_names(coverage),
        )
