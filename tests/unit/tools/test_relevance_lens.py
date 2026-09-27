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
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
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
    RelevanceLens,
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


class _FittedPair(NamedTuple):
    model: TransformerBridge
    logits_before_fit: torch.Tensor
    jacobian: JacobianLens
    relevance: RelevanceLens


@pytest.fixture(scope="module")
def fitted() -> _FittedPair:
    """Both estimators fitted on the same tiny model and prompts.

    Sharing one model keeps the comparison apples-to-apples: the only thing that
    differs between the two fits is the backward semantics the estimator
    selects. The pre-fit logits are captured so a test can prove the scoped rule
    context does not leak past the fit.
    """
    model = _build_tiny_qwen2()
    with torch.no_grad():
        logits_before_fit = model(model.to_tokens(PROMPTS[0]))
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
        jacobian=jacobian,
        relevance=relevance,
    )


def _expected_rule_mounts() -> set[str]:
    """Canonical mounts the three requested rules install on a pre-norm stack.

    Qwen2 is pre-norm only, so the LN-rule reaches the residual-stream norms at
    ``ln1``/``ln2`` and the Identity- and Half-rules reach the gated MLP at
    ``mlp``. Attention-internal q/k norms are deliberately absent: the LN-rule
    targets residual-stream norms only.
    """
    norms = {f"blocks.{layer}.{mount}" for layer in range(N_LAYERS) for mount in ("ln1", "ln2")}
    mlps = {f"blocks.{layer}.mlp" for layer in range(N_LAYERS)}
    return norms | mlps


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
        assert set(coverage.installed) == _expected_rule_mounts()
        assert coverage.skipped == ()

    def test_rule_scope_does_not_leak_past_the_fit(self, fitted: _FittedPair) -> None:
        with torch.no_grad():
            logits_after_fit = fitted.model(fitted.model.to_tokens(PROMPTS[0]))
        assert torch.equal(logits_after_fit, fitted.logits_before_fit)


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
    def test_jacobian_shard_cannot_merge_with_relevance_shard(self) -> None:
        jacobian = JacobianLens({0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL)
        with pytest.raises(ValueError, match="provenance"):
            RelevanceLens.merge([_shard(n_prompts=1), jacobian])

    def test_relevance_shard_cannot_merge_with_jacobian_shard(self) -> None:
        jacobian = JacobianLens({0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL)
        with pytest.raises(ValueError, match="provenance"):
            JacobianLens.merge([jacobian, _shard(n_prompts=1)])


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

    def test_artifact_without_estimator_provenance_still_loads(self, tmp_path: Any) -> None:
        path = tmp_path / "lens.pt"
        torch.save(
            {
                "J": {0: torch.eye(D_MODEL)},
                "n_prompts": 1,
                "source_layers": [0],
                "d_model": D_MODEL,
            },
            path,
        )

        loaded = RelevanceLens.load(str(path))

        assert loaded.estimator == "relevance_lens"


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
        assert overridden == {"fit", "load", "_merge_identity"}


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
        assert "blocks.0.mlp" in message
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
