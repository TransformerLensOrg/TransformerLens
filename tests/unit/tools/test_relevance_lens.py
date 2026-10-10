"""Unit tests for ``RelevanceLens.fit``: shape/finite parity and rule coverage.

These tests build a tiny random Qwen2 fully offline (programmatic HF config,
random weights, no network access) so the fit runs on a real assembled
``TransformerBridge`` with the opaque gated-MLP native-forward path. Both
estimators are fitted on the same model and prompts, so the only difference
between the two sets of transport matrices is the backward semantics each
estimator selects.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple, Sequence

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

import transformer_lens.tools.analysis.jacobian_lens as jacobian_lens_module
import transformer_lens.tools.analysis.relevance_lens as relevance_lens_module
from tests.unit.tools.conftest import _ToyBridge
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRuleCoverage,
    RelevanceRuleUnsupportedError,
    _RelevanceRuleCoverageEntry,
    use_relevance_rules,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_config_from_hf
from transformer_lens.model_bridge.supported_architectures.gpt2 import (
    GPT2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)
from transformer_lens.tools.analysis import JacobianLens
from transformer_lens.tools.analysis.relevance_lens import (
    RELEVANCE_RULE_VERSION,
    RELEVANCE_RULES,
    RelevanceLens,
    _installed_rule_names,
)

N_LAYERS = 2
D_MODEL = 32
SOURCE_LAYERS = [0]
CORPUS = "unit-test-corpus"
PROMPTS = [
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega alpha beta gamma delta",
    "one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty one "
    "twenty two twenty three twenty four twenty five twenty six twenty seven",
    "red orange yellow green blue indigo violet cyan magenta amber teal olive "
    "maroon navy silver gold bronze copper crimson scarlet azure beige ivory "
    "khaki lilac peach plum russet saffron tan umber",
]
TINY_DIMS = dict(
    vocab_size=97,
    hidden_size=D_MODEL,
    intermediate_size=64,
    num_hidden_layers=N_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)


def _offline_tokenizer(prompts: Sequence[str]) -> PreTrainedTokenizerFast:
    """Word-level tokenizer over the prompts' own vocabulary, built offline.

    A real ``PreTrainedTokenizerFast`` (rather than a duck-typed stub) satisfies
    the jaxtyping contract on ``TransformerBridge.to_tokens``, and deriving the
    vocabulary from the prompts keeps every word in-vocab so no prompt collapses
    to a run of ``<unk>``.
    """
    words = sorted({word for prompt in prompts for word in prompt.split()})
    vocabulary = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    vocabulary.update({word: index + 4 for index, word in enumerate(words)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def _build_tiny_qwen2() -> TransformerBridge:
    hf_config = AutoConfig.for_model("qwen2", **TINY_DIMS)
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "Qwen2ForCausalLM", "qwen2-tiny", torch.float32
    )
    adapter = Qwen2ArchitectureAdapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_offline_tokenizer(PROMPTS))


def _build_tiny_qwen2_relu() -> TransformerBridge:
    """A gated MLP whose activation the Identity-rule cannot honor."""
    hf_config = AutoConfig.for_model("qwen2", **TINY_DIMS, hidden_act="relu")
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "Qwen2ForCausalLM", "qwen2-relu-tiny", torch.float32
    )
    adapter = Qwen2ArchitectureAdapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_offline_tokenizer(PROMPTS))


def _residual_gradient(model: TransformerBridge, tokens: torch.Tensor) -> torch.Tensor:
    """Probe downstream backward semantics without accumulating parameter gradients."""
    captured: dict[str, torch.Tensor] = {}

    def capture(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        captured["residual"] = activation.detach().requires_grad_(True)
        return captured["residual"]

    with torch.enable_grad(), model.hooks(fwd_hooks=[("blocks.0.hook_out", capture)]):
        logits = model(tokens)
        (gradient,) = torch.autograd.grad(logits[0, -1, 0], captured["residual"])
    assert torch.isfinite(gradient).all()
    assert gradient.count_nonzero() > 0
    return gradient.detach().clone()


class _FittedPair(NamedTuple):
    model: TransformerBridge
    logits_before_fit: torch.Tensor
    gradient_before_fit: torch.Tensor
    jacobian: JacobianLens
    relevance: RelevanceLens


@pytest.fixture(scope="module")
def fitted() -> _FittedPair:
    """Both estimators on one model, with separate forward and backward baselines."""
    model = _build_tiny_qwen2()
    tokens = model.to_tokens(PROMPTS[0])
    with torch.no_grad():
        logits_before_fit = model(tokens)
    gradient_before_fit = _residual_gradient(model, tokens)
    torch.testing.assert_close(
        _residual_gradient(model, tokens), gradient_before_fit, atol=0, rtol=0
    )
    jacobian = JacobianLens.fit(
        model,
        PROMPTS,
        corpus=CORPUS,
        source_layers=SOURCE_LAYERS,
        show_progress=False,
    )
    relevance = RelevanceLens.fit(
        model,
        PROMPTS,
        corpus=CORPUS,
        source_layers=SOURCE_LAYERS,
        show_progress=False,
    )
    return _FittedPair(
        model=model,
        logits_before_fit=logits_before_fit,
        gradient_before_fit=gradient_before_fit,
        jacobian=jacobian,
        relevance=relevance,
    )


def _expected_rule_mounts() -> tuple[_RelevanceRuleCoverageEntry, ...]:
    """Canonical rule-kind/path pairs, excluding attention-internal norms."""
    return tuple(
        _RelevanceRuleCoverageEntry(kind=kind, path=f"blocks.{layer}.{mount}")
        for kind, mounts in (
            ("normalization", ("ln1", "ln2")),
            ("activation", ("mlp",)),
            ("multiplicative_gate", ("mlp",)),
        )
        for layer in range(N_LAYERS)
        for mount in mounts
    )


@pytest.fixture(params=[JacobianLens, RelevanceLens], ids=["jacobian", "relevance"])
def estimator(request: pytest.FixtureRequest) -> Any:
    return request.param


def _forbid_fit_work(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("invalid fit options reached tokenization or rule installation")


class TestSharedFitValidation:
    @pytest.mark.parametrize(
        "options, message",
        [
            ({"corpus": ""}, "corpus"),
            ({"corpus": " "}, "corpus"),
            ({"source_layers": []}, "source_layers"),
            ({"source_layers": [N_LAYERS - 1]}, "target_layer"),
            ({"source_layers": [N_LAYERS]}, "out of range"),
            ({"dim_batch": 0}, "dim_batch"),
            ({"skip_first_positions": -1}, "skip_first_positions"),
        ],
    )
    def test_invalid_options_fail_before_fit_work(
        self,
        estimator: Any,
        options: dict[str, Any],
        message: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_tiny_qwen2()
        monkeypatch.setattr(jacobian_lens_module, "_fit_transport_matrices", _forbid_fit_work)
        monkeypatch.setattr(relevance_lens_module, "_fit_transport_matrices", _forbid_fit_work)
        monkeypatch.setattr(relevance_lens_module, "use_relevance_rules", _forbid_fit_work)
        with pytest.raises(ValueError, match=message):
            estimator.fit(model, PROMPTS, **{"corpus": CORPUS, **options})

    @pytest.mark.parametrize(
        "state, message",
        [
            ("compatibility", "compatibility mode"),
            ("processed", "process_weights"),
            ("training", r"model\.eval\(\)"),
            ("submodule", r"model\.eval\(\)"),
        ],
    )
    def test_model_state_rejections_are_shared(
        self, estimator: Any, state: str, message: str
    ) -> None:
        model = _build_tiny_qwen2()
        if state == "compatibility":
            model.compatibility_mode = True
        elif state == "processed":
            model._weights_processed = True
        elif state == "training":
            model.train()
        else:
            model.blocks[0].train()
        with pytest.raises(ValueError, match=message):
            estimator.fit(model, PROMPTS, corpus=CORPUS, show_progress=False)

    def test_hidden_original_submodule_in_training_mode_is_refused(self, estimator: Any) -> None:
        model = _ToyBridge()
        original_model = torch.nn.Sequential(torch.nn.Linear(4, 4)).eval()
        original_model[0].train()
        model.original_model = original_model
        with pytest.raises(ValueError, match="original_model"):
            estimator.fit(model, PROMPTS, corpus=CORPUS)
        assert original_model.training is False
        assert original_model[0].training is True

    def test_non_bridge_is_refused(self, estimator: Any) -> None:
        with pytest.raises(TypeError, match="TransformerBridge"):
            estimator.fit(object(), PROMPTS, corpus=CORPUS)

    def test_both_estimators_reach_the_same_raw_bridge_guard(
        self,
        estimator: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def refuse(*args: Any, **kwargs: Any) -> None:
            raise ValueError("shared raw-bridge guard")

        monkeypatch.setattr(jacobian_lens_module, "_require_raw_bridge", refuse)
        with pytest.raises(ValueError, match="shared raw-bridge guard"):
            estimator.fit(_ToyBridge(), PROMPTS, corpus=CORPUS)

    @pytest.mark.parametrize(
        "key",
        [
            "model_name",
            "model_revision",
            "transformer_lens_version",
            "model_system",
            "estimator",
            "processing",
            "hook_convention",
            "corpus",
            "n_prompts",
            "fit_dtype",
            "target_layer",
            "dim_batch",
            "max_seq_len",
            "skip_first_positions",
            "transformer_lens_fit",
        ],
    )
    def test_common_provenance_is_reserved_before_fit_work(
        self,
        estimator: Any,
        key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(jacobian_lens_module, "_fit_transport_matrices", _forbid_fit_work)
        monkeypatch.setattr(relevance_lens_module, "_fit_transport_matrices", _forbid_fit_work)
        monkeypatch.setattr(relevance_lens_module, "use_relevance_rules", _forbid_fit_work)
        with pytest.raises(ValueError, match="cannot override fit provenance"):
            estimator.fit(_ToyBridge(), PROMPTS, corpus=CORPUS, metadata={key: None})

    @pytest.mark.parametrize("key", ["relevance_rule_version", "enabled_rules", "rule_coverage"])
    def test_relevance_provenance_is_reserved_before_rule_installation(
        self,
        key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(relevance_lens_module, "use_relevance_rules", _forbid_fit_work)
        with pytest.raises(ValueError, match="cannot override fit provenance"):
            RelevanceLens.fit(_ToyBridge(), PROMPTS, corpus=CORPUS, metadata={key: None})

    @pytest.mark.parametrize("value", [torch.tensor(1), {"unsafe"}, object()])
    def test_unsafe_user_metadata_is_refused_before_fit_work(
        self,
        estimator: Any,
        value: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(jacobian_lens_module, "_fit_transport_matrices", _forbid_fit_work)
        monkeypatch.setattr(relevance_lens_module, "use_relevance_rules", _forbid_fit_work)
        with pytest.raises(ValueError, match="metadata"):
            estimator.fit(_ToyBridge(), PROMPTS, corpus=CORPUS, metadata={"run": value})


class TestSharedFitProvenance:
    def test_common_provenance_matches_between_estimators(self, fitted: _FittedPair) -> None:
        relevance_only = {"relevance_rule_version", "enabled_rules", "rule_coverage"}
        assert fitted.relevance.metadata["estimator"] == "relevance_lens"
        assert fitted.jacobian.metadata["estimator"] == "jacobian_lens"
        assert {
            key: value
            for key, value in fitted.relevance.metadata.items()
            if key not in relevance_only | {"estimator"}
        } == {key: value for key, value in fitted.jacobian.metadata.items() if key != "estimator"}

    def test_custom_metadata_is_preserved(self, estimator: Any) -> None:
        metadata = {"run": {"seed": 0, "tags": ["offline", "fit"]}}
        lens = estimator.fit(
            _build_tiny_qwen2(),
            PROMPTS[:1],
            corpus=CORPUS,
            source_layers=SOURCE_LAYERS,
            show_progress=False,
            metadata=metadata,
        )
        assert lens.metadata["run"] == metadata["run"]
        assert set(metadata) == {"run"}
        assert lens.metadata["n_prompts"] == lens.n_prompts == 1


class TestFitShapeAndFiniteness:
    def test_source_layers_and_d_model_match_jacobian_lens(self, fitted: _FittedPair) -> None:
        assert fitted.relevance.source_layers == fitted.jacobian.source_layers
        assert fitted.relevance.d_model == fitted.jacobian.d_model
        assert fitted.relevance.n_prompts == fitted.jacobian.n_prompts

    def test_every_transport_matrix_is_finite_and_square(self, fitted: _FittedPair) -> None:
        for layer in fitted.relevance.source_layers:
            matrix = fitted.relevance.jacobians[layer]
            assert matrix.shape == (D_MODEL, D_MODEL)
            assert torch.isfinite(matrix).all()


class TestRuleCoverageSelection:
    def test_installs_residual_norm_activation_and_gate_rules_only(
        self, fitted: _FittedPair
    ) -> None:
        coverage = fitted.relevance.rule_coverage
        assert coverage.installed == _expected_rule_mounts()
        assert coverage.skipped == ()

    def test_fit_preserves_forward_values(self, fitted: _FittedPair) -> None:
        with torch.no_grad():
            logits_after_fit = fitted.model(fitted.model.to_tokens(PROMPTS[0]))
        assert torch.equal(logits_after_fit, fitted.logits_before_fit)

    def test_rule_scope_does_not_leak_past_the_fit(self, fitted: _FittedPair) -> None:
        tokens = fitted.model.to_tokens(PROMPTS[0])
        after_fit = _residual_gradient(fitted.model, tokens)
        with use_relevance_rules(fitted.model, RELEVANCE_RULES):
            active_gradient = _residual_gradient(fitted.model, tokens)
        assert not torch.allclose(active_gradient, fitted.gradient_before_fit, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(after_fit, fitted.gradient_before_fit, atol=0, rtol=0)
        assert all(parameter.grad is None for parameter in fitted.model.parameters())


class _FitFailure(RuntimeError):
    """Injected failure during a rule-active backward pass."""


class TestFitRestoresBackwardSemantics:
    def test_restores_gradients_after_backward_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _build_tiny_qwen2()
        tokens = model.to_tokens(PROMPTS[0])
        baseline = _residual_gradient(model, tokens)
        parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
        active_gradients = []

        def fail_backward(*args: Any, **kwargs: Any) -> Any:
            active_gradients.append(_residual_gradient(model, tokens))
            raise _FitFailure("injected backward failure")

        monkeypatch.setattr(relevance_lens_module, "_ordinary_vjp", fail_backward)
        with pytest.raises(_FitFailure, match="injected backward failure"):
            RelevanceLens.fit(
                model,
                PROMPTS[:1],
                corpus=CORPUS,
                source_layers=SOURCE_LAYERS,
                show_progress=False,
            )
        assert len(active_gradients) == 1
        assert not torch.allclose(active_gradients[0], baseline, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(_residual_gradient(model, tokens), baseline, atol=0, rtol=0)
        assert [parameter.requires_grad for parameter in model.parameters()] == parameter_flags
        assert all(parameter.grad is None for parameter in model.parameters())

    def test_nested_fit_preserves_outer_backward_rules(self) -> None:
        model = _build_tiny_qwen2()
        tokens = model.to_tokens(PROMPTS[0])
        baseline = _residual_gradient(model, tokens)
        with use_relevance_rules(model, RELEVANCE_RULES):
            outer_gradient = _residual_gradient(model, tokens)
            assert not torch.allclose(outer_gradient, baseline, atol=1e-6, rtol=1e-4)
            RelevanceLens.fit(
                model,
                PROMPTS[:1],
                corpus=CORPUS,
                source_layers=SOURCE_LAYERS,
                show_progress=False,
            )
            torch.testing.assert_close(
                _residual_gradient(model, tokens), outer_gradient, atol=0, rtol=0
            )
        torch.testing.assert_close(_residual_gradient(model, tokens), baseline, atol=0, rtol=0)
        assert all(parameter.grad is None for parameter in model.parameters())


class TestEstimatorDiffersFromOrdinaryJacobian:
    def test_early_layer_transport_differs_from_jacobian_lens(self, fitted: _FittedPair) -> None:
        differing = [
            layer
            for layer in fitted.jacobian.source_layers
            if not torch.allclose(
                fitted.relevance.jacobians[layer],
                fitted.jacobian.jacobians[layer],
            )
        ]
        assert differing, (
            "every relevance transport matrix equals the ordinary Jacobian; the "
            "requested rules changed no backward semantics"
        )


def _shard(
    *,
    n_prompts: int,
    metadata: dict[str, Any] | None = None,
    rule_version: int = RELEVANCE_RULE_VERSION,
    enabled_rules: Sequence[str] = (
        "normalization",
        "activation",
        "multiplicative_gate",
    ),
) -> RelevanceLens:
    """A hand-built relevance shard, so merge/registry tests need no model."""
    return RelevanceLens(
        {0: torch.eye(D_MODEL)},
        n_prompts=n_prompts,
        d_model=D_MODEL,
        metadata=metadata,
        relevance_rule_version=rule_version,
        enabled_rules=enabled_rules,
    )


class TestMergeRejectsMixedEstimators:
    @pytest.mark.parametrize("n_shards", [1, 2])
    @pytest.mark.parametrize("metadata", [{}, {"estimator": "jacobian_lens"}])
    def test_jacobian_shards_cannot_be_relabelled(
        self, n_shards: int, metadata: dict[str, Any]
    ) -> None:
        shards = [
            JacobianLens({0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL, metadata=metadata)
            for _ in range(n_shards)
        ]
        before = [dict(shard.metadata) for shard in shards]

        with pytest.raises(ValueError, match="provenance"):
            RelevanceLens.merge(shards)

        assert [shard.metadata for shard in shards] == before

    def test_generic_lens_with_relevance_metadata_is_not_a_relevance_shard(self) -> None:
        shard = JacobianLens(
            {0: torch.eye(D_MODEL)},
            n_prompts=1,
            d_model=D_MODEL,
            metadata={"estimator": "relevance_lens"},
        )
        with pytest.raises(ValueError, match="provenance"):
            RelevanceLens.merge([shard])

    def test_relevance_shard_requires_positive_metadata(self) -> None:
        shard = _shard(n_prompts=1)
        del shard.metadata["estimator"]
        before = dict(shard.metadata)
        with pytest.raises(ValueError, match="provenance"):
            RelevanceLens.merge([shard])
        assert shard.metadata == before

    @pytest.mark.parametrize("reverse", [False, True])
    def test_jacobian_shard_cannot_merge_with_relevance_shard(self, reverse: bool) -> None:
        jacobian = JacobianLens({0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL)
        shards = [jacobian, _shard(n_prompts=1)]
        with pytest.raises(ValueError, match="provenance"):
            RelevanceLens.merge(shards[::-1] if reverse else shards)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_relevance_shard_cannot_merge_with_jacobian_shard(self, reverse: bool) -> None:
        jacobian = JacobianLens({0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL)
        shards = [jacobian, _shard(n_prompts=1)]
        with pytest.raises(ValueError, match="provenance"):
            JacobianLens.merge(shards[::-1] if reverse else shards)


class TestMergeRejectsMismatchedRuleConfiguration:
    def test_different_rule_version_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="provenance"):
            RelevanceLens.merge(
                [
                    _shard(n_prompts=1, rule_version=RELEVANCE_RULE_VERSION),
                    _shard(n_prompts=1, rule_version=RELEVANCE_RULE_VERSION + 1),
                ]
            )

    def test_different_enabled_rules_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="provenance"):
            RelevanceLens.merge(
                [
                    _shard(n_prompts=1),
                    _shard(n_prompts=1, enabled_rules=("normalization",)),
                ]
            )


class TestMergeWeightsByPromptCount:
    def test_identical_provenance_shards_merge_by_prompt_count(self) -> None:
        low = _shard(n_prompts=2)
        low.jacobians[0] = torch.zeros(D_MODEL, D_MODEL)
        high = _shard(n_prompts=6)
        high.jacobians[0] = 4 * torch.ones(D_MODEL, D_MODEL)

        merged = RelevanceLens.merge([low, high])

        assert merged.n_prompts == 8
        assert torch.allclose(merged.jacobians[0], 3 * torch.ones(D_MODEL, D_MODEL))
        assert merged.estimator == "relevance_lens"


class TestRegistryResolutionIsDisabled:
    def test_short_name_does_not_resolve_the_jacobian_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_if_called(name_or_path: str) -> Any:
            raise AssertionError(
                f"the Jacobian registry must not resolve {name_or_path!r} for a " "relevance lens"
            )

        monkeypatch.setattr(jacobian_lens_module, "_resolve_registry_entry", fail_if_called)
        with pytest.raises(ValueError, match="registry"):
            RelevanceLens.from_pretrained("gemma-2-2b")

    def test_explicit_hub_repo_is_still_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        path = tmp_path / "lens.pt"
        _shard(n_prompts=1).save(str(path))
        calls: dict[str, Any] = {}

        def fake_retry(function: Any, **kwargs: Any) -> str:
            calls["kwargs"] = kwargs
            return str(path)

        monkeypatch.setattr(jacobian_lens_module, "call_hf_with_retry", fake_retry)

        loaded = RelevanceLens.from_pretrained("example/lenses", filename="model/lens.pt")

        assert calls["kwargs"]["repo_id"] == "example/lenses"
        assert calls["kwargs"]["filename"] == "model/lens.pt"
        assert loaded.estimator == "relevance_lens"


@pytest.mark.parametrize(
    "kinds",
    [
        (),
        ("normalization",),
        ("activation",),
        ("multiplicative_gate",),
        ("normalization", "activation", "multiplicative_gate"),
        ("multiplicative_gate", "activation", "normalization"),
    ],
)
def test_installed_rule_names_use_kinds_not_shared_mount_names(kinds: tuple[str, ...]) -> None:
    coverage = RelevanceRuleCoverage(
        installed=tuple(
            _RelevanceRuleCoverageEntry(
                kind, "blocks.0.ln1" if kind == "normalization" else "blocks.0.mlp"
            )
            for kind in kinds
        ),
        skipped=(),
    )
    assert _installed_rule_names(coverage) == [
        kind for kind in ("normalization", "activation", "multiplicative_gate") if kind in kinds
    ]


class TestRuleProvenanceSurvivesPersistence:
    def test_fitted_rule_configuration_round_trips(
        self, fitted: _FittedPair, tmp_path: Any
    ) -> None:
        path = tmp_path / "lens.pt"
        fitted.relevance.save(str(path))

        loaded = RelevanceLens.load(str(path))

        assert loaded.estimator == "relevance_lens"
        assert loaded.relevance_rule_version == fitted.relevance.relevance_rule_version
        assert loaded.enabled_rules == fitted.relevance.enabled_rules
        assert loaded.rule_coverage == fitted.relevance.rule_coverage
        assert loaded.metadata["rule_coverage"] == {
            "schema_version": 1,
            "installed": [
                {"kind": entry.kind, "path": entry.path} for entry in _expected_rule_mounts()
            ],
            "skipped": [],
        }

    @pytest.mark.parametrize("kind", ["activation", "multiplicative_gate"])
    def test_partial_coverage_at_one_mount_round_trips(self, tmp_path: Any, kind: str) -> None:
        other_kind = "multiplicative_gate" if kind == "activation" else "activation"
        coverage = RelevanceRuleCoverage(
            installed=(_RelevanceRuleCoverageEntry(kind, "blocks.0.mlp"),),
            skipped=(_RelevanceRuleCoverageEntry(other_kind, "blocks.0.mlp"),),
        )
        lens = RelevanceLens(
            {0: torch.eye(D_MODEL)},
            n_prompts=1,
            d_model=D_MODEL,
            rule_coverage=coverage,
            enabled_rules=_installed_rule_names(coverage),
        )
        path = tmp_path / "partial.pt"
        lens.save(str(path))
        loaded = RelevanceLens.load(str(path))
        assert loaded.rule_coverage == coverage
        assert loaded.enabled_rules == [kind]
        assert loaded.metadata["rule_coverage"] == {
            "schema_version": 1,
            "installed": [{"kind": kind, "path": "blocks.0.mlp"}],
            "skipped": [{"kind": other_kind, "path": "blocks.0.mlp"}],
        }

    def test_explicit_coverage_cannot_hide_contradictory_metadata(self) -> None:
        metadata = {"rule_coverage": {"schema_version": 1, "installed": [], "skipped": []}}
        coverage = RelevanceRuleCoverage(
            installed=(_RelevanceRuleCoverageEntry("activation", "blocks.0.mlp"),),
            skipped=(),
        )
        with pytest.raises(ValueError, match="coverage provenance"):
            RelevanceLens(
                {0: torch.eye(D_MODEL)},
                n_prompts=1,
                d_model=D_MODEL,
                metadata=metadata,
                rule_coverage=coverage,
            )

    def test_non_default_rule_configuration_round_trips(self, tmp_path: Any) -> None:
        path = tmp_path / "lens.pt"
        _shard(
            n_prompts=1,
            rule_version=RELEVANCE_RULE_VERSION + 3,
            enabled_rules=("normalization",),
        ).save(str(path))

        loaded = RelevanceLens.load(str(path))

        assert loaded.relevance_rule_version == RELEVANCE_RULE_VERSION + 3
        assert loaded.enabled_rules == ["normalization"]
        assert loaded.relevance_rule_version == loaded.metadata["relevance_rule_version"]
        assert loaded.enabled_rules == loaded.metadata["enabled_rules"]

    def test_loaded_lens_still_refuses_a_mismatched_merge(self, tmp_path: Any) -> None:
        path = tmp_path / "lens.pt"
        _shard(n_prompts=1, rule_version=RELEVANCE_RULE_VERSION + 3).save(str(path))
        loaded = RelevanceLens.load(str(path))

        with pytest.raises(ValueError, match="provenance"):
            RelevanceLens.merge([loaded, _shard(n_prompts=1)])


class TestMergePropagatesRuleConfiguration:
    def test_merged_lens_keeps_the_rule_configuration(self) -> None:
        rule_version = RELEVANCE_RULE_VERSION + 2
        low = _shard(
            n_prompts=2,
            rule_version=rule_version,
            enabled_rules=("normalization",),
        )
        high = _shard(
            n_prompts=6,
            rule_version=rule_version,
            enabled_rules=("normalization",),
        )

        merged = RelevanceLens.merge([low, high])

        assert merged.relevance_rule_version == rule_version
        assert merged.enabled_rules == ["normalization"]


class TestPublicExport:
    def test_relevance_lens_is_importable_from_the_analysis_package(self) -> None:
        from transformer_lens.tools import analysis

        assert analysis.RelevanceLens is RelevanceLens
        assert "RelevanceLens" in analysis.__all__

    def test_analysis_exports_stay_alphabetized(self) -> None:
        from transformer_lens.tools import analysis

        assert analysis.__all__ == sorted(analysis.__all__)


class TestLoadRejectsForeignEstimator:
    def test_jacobian_artifact_cannot_load_as_a_relevance_lens(self, tmp_path: Any) -> None:
        path = tmp_path / "lens.pt"
        JacobianLens(
            {0: torch.eye(D_MODEL)},
            n_prompts=1,
            d_model=D_MODEL,
            metadata={"estimator": "jacobian_lens"},
        ).save(str(path))

        with pytest.raises(ValueError, match="estimator"):
            RelevanceLens.load(str(path))

    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            {},
            {"relevance_rule_version": RELEVANCE_RULE_VERSION},
            {"estimator": None},
            {"estimator": 1},
            {"estimator": ["relevance_lens"]},
            {"estimator": "unknown"},
        ],
    )
    def test_artifact_requires_positive_estimator_provenance(
        self, tmp_path: Any, metadata: Any
    ) -> None:
        path = tmp_path / "lens.pt"
        payload = {
            "J": {0: torch.eye(D_MODEL)},
            "n_prompts": 1,
            "source_layers": [0],
            "d_model": D_MODEL,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        torch.save(payload, path)
        before = path.read_bytes()

        with pytest.raises(ValueError, match="estimator"):
            RelevanceLens.load(str(path))

        assert path.read_bytes() == before

    def test_legacy_jacobian_artifact_remains_loadable_as_jacobian(self, tmp_path: Any) -> None:
        path = tmp_path / "legacy.pt"
        JacobianLens({0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL).save(str(path))

        loaded = JacobianLens.load(str(path))
        assert loaded.estimator == "jacobian_lens"
        assert loaded.metadata == {}
        with pytest.raises(ValueError, match="estimator"):
            RelevanceLens.load(str(path))

    @pytest.mark.parametrize("metadata", [{}, {"estimator": "relevance_lens"}])
    def test_running_sum_checkpoint_is_refused(self, tmp_path: Any, metadata: Any) -> None:
        path = tmp_path / "checkpoint.pt"
        torch.save(
            {"jacobian_sum": {0: torch.eye(D_MODEL)}, "n_done": 1, "metadata": metadata},
            path,
        )
        before = path.read_bytes()
        with pytest.raises(ValueError, match="checkpoint"):
            RelevanceLens.load(str(path))
        assert path.read_bytes() == before


def _coverage_record(
    *, installed: Any = (), skipped: Any = (), schema_version: Any = 1
) -> dict[str, Any]:
    return {"schema_version": schema_version, "installed": installed, "skipped": skipped}


class TestCoverageMetadataValidation:
    @pytest.mark.parametrize(
        "recorded",
        [
            None,
            [],
            {},
            {"installed": ["blocks.0.mlp"], "skipped": []},
            _coverage_record(schema_version=2),
            _coverage_record(schema_version=True),
            _coverage_record(schema_version="1"),
            {"schema_version": 1, "installed": []},
            _coverage_record(installed="blocks.0.mlp"),
            _coverage_record(skipped=None),
            _coverage_record(installed=["blocks.0.mlp"]),
            _coverage_record(skipped=["blocks.0.mlp"]),
            _coverage_record(installed=[{"path": "blocks.0.mlp"}]),
            _coverage_record(installed=[{"kind": "unknown", "path": "blocks.0.mlp"}]),
            _coverage_record(installed=[{"kind": None, "path": "blocks.0.mlp"}]),
            _coverage_record(installed=[{"kind": "activation", "path": 1}]),
            _coverage_record(installed=[{"kind": "activation", "path": ""}]),
            _coverage_record(skipped=[{"kind": "activation", "path": " "}]),
            _coverage_record(installed=[{"kind": "activation", "path": "blocks.0.mlp"}] * 2),
            _coverage_record(
                installed=[{"kind": "activation", "path": "blocks.0.mlp"}],
                skipped=[{"kind": "activation", "path": "blocks.0.mlp"}],
            ),
        ],
    )
    def test_invalid_coverage_is_refused_on_load(self, tmp_path: Any, recorded: Any) -> None:
        path = tmp_path / "coverage.pt"
        _shard(n_prompts=1).save(str(path))
        payload = torch.load(path, weights_only=True)
        payload["metadata"]["rule_coverage"] = recorded
        torch.save(payload, path)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="rule_coverage"):
            RelevanceLens.load(str(path))
        assert path.read_bytes() == before

    def test_optional_coverage_can_be_absent(self, tmp_path: Any) -> None:
        path = tmp_path / "without-coverage.pt"
        _shard(n_prompts=1).save(str(path))
        loaded = RelevanceLens.load(str(path))
        assert loaded.rule_coverage is None
        assert "rule_coverage" not in loaded.metadata

    def test_legacy_coverage_error_requires_refitting(self) -> None:
        with pytest.raises(ValueError, match="path-only coverage is ambiguous.*refit"):
            _shard(
                n_prompts=1,
                metadata={"rule_coverage": {"installed": ["blocks.0.mlp"], "skipped": []}},
            )


class TestConstructorProvenance:
    @pytest.mark.parametrize("estimator", [None, 1, ["relevance_lens"], "jacobian_lens", "other"])
    def test_contradictory_estimator_is_refused(self, estimator: Any) -> None:
        metadata = {"estimator": estimator}
        with pytest.raises(ValueError, match="estimator"):
            _shard(n_prompts=1, metadata=metadata)
        assert metadata == {"estimator": estimator}

    @pytest.mark.parametrize(
        "metadata",
        [
            {"relevance_rule_version": None},
            {"relevance_rule_version": True},
            {"relevance_rule_version": "1"},
            {"enabled_rules": None},
            {"enabled_rules": "normalization"},
            {"enabled_rules": [1]},
        ],
    )
    def test_invalid_recorded_configuration_is_not_replaced_by_defaults(
        self, metadata: dict[str, Any]
    ) -> None:
        before = dict(metadata)
        with pytest.raises(ValueError, match="provenance"):
            _shard(n_prompts=1, metadata=metadata)
        assert metadata == before

    def test_recorded_configuration_takes_precedence_without_mutating_metadata(self) -> None:
        metadata = {
            "estimator": "relevance_lens",
            "relevance_rule_version": RELEVANCE_RULE_VERSION + 2,
            "enabled_rules": ("normalization",),
        }
        lens = _shard(n_prompts=1, metadata=metadata)
        assert lens.estimator == lens.metadata["estimator"]
        assert lens.relevance_rule_version == lens.metadata["relevance_rule_version"]
        assert lens.enabled_rules == lens.metadata["enabled_rules"] == ["normalization"]
        assert metadata["enabled_rules"] == ("normalization",)

    def test_generic_merge_preserves_recorded_relevance_identity(self) -> None:
        merged = JacobianLens.merge([_shard(n_prompts=1), _shard(n_prompts=2)])
        assert merged.n_prompts == 3
        assert merged.estimator == merged.metadata["estimator"] == "relevance_lens"


class TestAnalysisSurfaceIsReusedNotDuplicated:
    """The estimator changes only how matrices are estimated.

    Readout, vocabulary vectors, sparse decomposition, and the interventions are
    inherited from the Jacobian lens rather than reimplemented, so a fix to any
    of them reaches both estimators. This pins that contract: a future
    reimplementation on the subclass would silently fork the two surfaces.
    """

    INHERITED_METHODS = (
        "transport",
        "readout",
        "lens_vectors",
        "lens_vector_dictionary",
        "decompose",
        "occupancy",
        "fraction_of_variance",
        "steering_hooks",
        "ablation_hooks",
        "swap_hooks",
        "swap_clamp_hooks",
        "coordinate_patch",
        "coordinate_patch_hooks",
        "validate_model",
        "save",
        "load",
        "merge",
        "from_pretrained",
    )

    def test_downstream_methods_are_inherited_unchanged(self) -> None:
        for name in self.INHERITED_METHODS:
            assert name not in vars(RelevanceLens), (
                f"{name} is redefined on RelevanceLens; the estimator is meant to "
                "change only how transport matrices are estimated"
            )
            # Compare the underlying functions: a classmethod accessor returns a
            # fresh bound method on every getattr, so identity on the accessor
            # itself would never hold.
            ours = getattr(RelevanceLens, name)
            theirs = getattr(JacobianLens, name)
            assert getattr(ours, "__func__", ours) is getattr(theirs, "__func__", theirs)

    def test_only_the_estimator_specific_surface_is_overridden(self) -> None:
        overridden = {
            name
            for name in vars(RelevanceLens)
            if not name.startswith("__") and callable(getattr(RelevanceLens, name))
        }
        assert overridden == {
            "fit",
            "_validate_artifact_metadata",
            "_validate_checkpoint_payload",
            "_validate_merge_inputs",
        }


def _build_tiny_gpt2() -> TransformerBridge:
    """A dense-MLP, plain-python-norm model that can honor none of the rules.

    GPT-2 has no gated MLP and its norms do not dispatch through the
    native-autograd branch the LN-rule wraps, so every canonical mount is
    reported skipped.
    """
    hf_config = AutoConfig.for_model(
        "gpt2",
        vocab_size=97,
        n_embd=D_MODEL,
        n_layer=N_LAYERS,
        n_head=4,
        n_positions=64,
        n_ctx=64,
    )
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "GPT2LMHeadModel", "gpt2-tiny", torch.float32
    )
    adapter = GPT2ArchitectureAdapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_offline_tokenizer(PROMPTS))


class TestFitRefusesSilentFallback:
    """A fit that installs no rule must not masquerade as a relevance lens.

    Without this guard the estimator would return ordinary Jacobian matrices
    labelled ``estimator="relevance_lens"``, which is indistinguishable from a
    real relevance fit in the artifact and would silently mislead a downstream
    comparison.
    """

    def test_fit_raises_when_no_rule_can_be_installed(self) -> None:
        model = _build_tiny_gpt2()

        with pytest.raises(ValueError, match="no relevance rule"):
            RelevanceLens.fit(
                model,
                PROMPTS,
                corpus=CORPUS,
                source_layers=SOURCE_LAYERS,
                show_progress=False,
            )

    def test_error_names_the_skipped_mounts(self) -> None:
        model = _build_tiny_gpt2()

        with pytest.raises(ValueError) as excinfo:
            RelevanceLens.fit(
                model,
                PROMPTS,
                corpus=CORPUS,
                source_layers=SOURCE_LAYERS,
                show_progress=False,
            )

        message = str(excinfo.value)
        assert "blocks.0.ln1" in message
        assert "blocks.0.mlp" in message


def _build_tiny_opt() -> TransformerBridge:
    """A model where the LN-rule installs but the MLP rules cannot.

    OPT has residual-stream norms on the native-autograd path but a dense ReLU
    MLP, so the normalization mounts install while every MLP mount is skipped.
    """
    hf_config = AutoConfig.for_model(
        "opt",
        vocab_size=97,
        hidden_size=D_MODEL,
        ffn_dim=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "OPTForCausalLM", "opt-tiny", torch.float32
    )
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_offline_tokenizer(PROMPTS))


class TestPartialCoverageIsRecordedHonestly:
    """``enabled_rules`` must name the rules that actually shaped the matrices.

    A model can honor some rules and not others. Recording the requested set
    rather than the installed set would claim the MLP rules contributed when
    every MLP mount was skipped, which misdescribes the artifact.
    """

    def test_partial_coverage_installs_only_the_normalization_rule(self) -> None:
        model = _build_tiny_opt()

        lens = RelevanceLens.fit(
            model,
            PROMPTS,
            corpus=CORPUS,
            source_layers=SOURCE_LAYERS,
            show_progress=False,
        )

        assert lens.rule_coverage.installed
        assert lens.rule_coverage.skipped
        assert lens.enabled_rules == ["normalization"]

    def test_recorded_enabled_rules_match_the_installed_mounts(self) -> None:
        model = _build_tiny_opt()

        lens = RelevanceLens.fit(
            model,
            PROMPTS,
            corpus=CORPUS,
            source_layers=SOURCE_LAYERS,
            show_progress=False,
        )

        assert lens.metadata["enabled_rules"] == lens.enabled_rules
        assert "activation" not in lens.enabled_rules
        assert "multiplicative_gate" not in lens.enabled_rules

    def test_partial_coverage_warns_with_the_skipped_mounts(self) -> None:
        model = _build_tiny_opt()

        with pytest.warns(UserWarning, match="skipped") as record:
            RelevanceLens.fit(
                model,
                PROMPTS,
                corpus=CORPUS,
                source_layers=SOURCE_LAYERS,
                show_progress=False,
            )

        message = str(record[0].message)
        assert "activation: blocks.0.mlp" in message
        assert "multiplicative_gate: blocks.0.mlp" in message
        assert "normalization" in message

    def test_full_coverage_does_not_warn_about_skipped_mounts(self) -> None:
        model = _build_tiny_qwen2()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            RelevanceLens.fit(
                model,
                PROMPTS,
                corpus=CORPUS,
                source_layers=SOURCE_LAYERS,
                show_progress=False,
            )

        assert not [
            item for item in record if "skipped" in str(item.message)
        ], "a fully covered fit must not warn about skipped mounts"


class TestUnsupportedActivationIsRefusedBeforeFitting:
    """A relu-family gated MLP must be refused, not scored with a wrong rule.

    The Identity-rule's backward multiplier ``f(x) / x`` reduces to ``relu(x)``
    for relu-squared rather than the true derivative ``2 * relu(x)``, so applying
    it there would produce plausible but wrong transport matrices. The refusal
    must happen before any fitting work, and must leave the model untouched.
    """

    def test_fit_raises_for_a_relu_gated_mlp(self) -> None:
        model = _build_tiny_qwen2_relu()

        with pytest.raises(RelevanceRuleUnsupportedError, match="activation"):
            RelevanceLens.fit(
                model,
                PROMPTS,
                corpus=CORPUS,
                source_layers=SOURCE_LAYERS,
                show_progress=False,
            )

    def test_refusal_leaves_the_model_untouched(self) -> None:
        model = _build_tiny_qwen2_relu()
        tokens = model.to_tokens(PROMPTS[0])
        with torch.no_grad():
            before = model(tokens)

        with pytest.raises(RelevanceRuleUnsupportedError):
            RelevanceLens.fit(
                model,
                PROMPTS,
                corpus=CORPUS,
                source_layers=SOURCE_LAYERS,
                show_progress=False,
            )

        with torch.no_grad():
            after = model(tokens)
        assert torch.equal(after, before)
        assert all(parameter.grad is None for parameter in model.parameters())
