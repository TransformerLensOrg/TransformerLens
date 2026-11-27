"""Each adapter coverage check reacts to the fault it exists to catch and passes without it.

A misleading adapter (wrong hooks or logits) must fail; an incomplete one must pass with
its gaps listed. Fixtures are seeded tiny models built from local HF configs, so no Hub
access is needed. A fault is applied to the adapter's component mapping before the bridge
is set up, the way a wrong adapter would have declared it.
"""

import copy
from typing import Any, Callable, Optional

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    BambaConfig,
    GraniteConfig,
    OpenAIGPTConfig,
    SmolLM3Config,
)

from transformer_lens.benchmarks import adapter_coverage as coverage
from transformer_lens.benchmarks import adapter_coverage_runner as runner
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    PositionEmbeddingsAttentionBridge,
    ScaledResidualBlockBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.sources import build_bridge_from_module

_DIMS: dict[str, Any] = dict(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
TOKENS = torch.tensor([[1, 17, 42, 99, 7, 63, 5, 80]])
Fault = Optional[Callable[[Any], None]]


def _boot(architecture: str, hf_model: Any, fault: Fault = None) -> Any:
    return build_bridge_from_module(
        hf_model,
        architecture,
        hf_config=copy.deepcopy(hf_model.config),
        device="cpu",
        post_adapter_hook=fault,
    ).eval()


def _build(architecture: str, config: Any, fault: Fault = None) -> tuple[Any, Any]:
    """A seeded tiny bridge and the pristine HF twin it is checked against."""
    config = copy.deepcopy(config)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model = AutoModelForCausalLM.from_config(config).eval()
    reference = copy.deepcopy(hf_model)
    return _boot(architecture, hf_model, fault), reference


def _status(result: BenchmarkResult) -> str:
    if result.severity == BenchmarkSeverity.SKIPPED:
        return "skip"
    if not result.passed:
        return "fail"
    return "gaps" if result.severity == BenchmarkSeverity.WARNING else "pass"


def _gaps(result: BenchmarkResult, kind: str) -> list[str]:
    assert result.details is not None
    return [gap["where"] for gap in result.details["gaps"] if gap["kind"] == kind]


def _report(bridge: Any, reference: Any) -> dict[str, Any]:
    """The runner's report for the checks that need no variants or reference adapter."""
    results = [
        coverage.benchmark_structural_coverage(bridge),
        coverage.benchmark_expected_hooks(bridge, TOKENS),
        coverage.benchmark_residual_floor(bridge, TOKENS),
        coverage.benchmark_hook_consistency(bridge, reference, TOKENS),
    ]
    return runner.summarize(results, coverage.declared_gaps(bridge.adapter))


def _shell_blocks(adapter: Any) -> None:
    """Map the blocks and nothing inside them."""
    blocks = adapter.component_mapping["blocks"]
    adapter.component_mapping["blocks"] = BlockBridge(name=blocks.name, config=adapter.cfg)


def _generic_mlp(adapter: Any) -> None:
    """Hide the MLP's projections behind one generic wrapper."""
    blocks = adapter.component_mapping["blocks"]
    blocks.submodules["mlp"] = GeneralizedComponent(name=blocks.submodules["mlp"].name)


def _unscaled_residual(adapter: Any) -> None:
    """Describe Granite's blocks as plain blocks, ignoring residual_multiplier."""
    blocks = adapter.component_mapping["blocks"]
    adapter.component_mapping["blocks"] = BlockBridge(
        name=blocks.name, config=adapter.cfg, submodules=blocks.submodules
    )


def _unscaled_logits(adapter: Any) -> None:
    """Forget that Granite divides its logits by logits_scaling."""
    adapter.output_logits_scale = lambda: 1.0


def _always_rotate(adapter: Any) -> None:
    """Use the attention bridge that applies RoPE on every layer."""
    blocks = adapter.component_mapping["blocks"]
    attention = blocks.submodules["attn"]
    blocks.submodules["attn"] = PositionEmbeddingsAttentionBridge(
        name=attention.name,
        config=adapter.cfg,
        submodules=attention.submodules,
        requires_attention_mask=True,
        requires_position_embeddings=True,
    )


def _no_mlp_out_hook(adapter: Any) -> None:
    """Declare that no single submodule feeds the MLP-side add, leaving no hook_mlp_out."""
    blocks = adapter.component_mapping["blocks"]
    adapter.component_mapping["blocks"] = ScaledResidualBlockBridge(
        name=blocks.name,
        config=adapter.cfg,
        submodules=blocks.submodules,
        scaled_mlp_submodule=None,
    )


def _unmapped_mixer(adapter: Any) -> None:
    """Leave a hybrid's state-space layers out of the mapping."""
    del adapter.component_mapping["blocks"].submodules["mixer"]


def _generic_mixer(adapter: Any) -> None:
    """Give each state-space layer an input and output hook and nothing inside."""
    blocks = adapter.component_mapping["blocks"]
    blocks.submodules["mixer"] = GeneralizedComponent(name="mamba", optional=True)


GRANITE = "GraniteForCausalLM"
SMOLLM3 = "SmolLM3ForCausalLM"
BAMBA = "BambaForCausalLM"


def _bamba() -> BambaConfig:
    """Mamba-2, attention, Mamba-2: a hybrid stack three layers deep."""
    return BambaConfig(
        **_DIMS,
        num_hidden_layers=3,
        attn_layer_indices=[1],
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=8,
    )


def _granite(**multipliers: Any) -> GraniteConfig:
    return GraniteConfig(**_DIMS, num_hidden_layers=2, **multipliers)


class TestStructuralCoverage:
    def test_passes_on_a_fully_mapped_adapter(self) -> None:
        bridge, _ = _build(GRANITE, _granite())
        result = coverage.benchmark_structural_coverage(bridge)
        assert _status(result) == "pass", result.message
        assert result.details is not None
        assert result.details["parameter_share"]["specific"] == 1.0
        assert result.details["pattern_coverage"]["share"] == 1.0

    def test_unmapped_block_internals_are_gaps_that_lower_coverage(self) -> None:
        bridge, _ = _build(GRANITE, _granite(), _shell_blocks)
        result = coverage.benchmark_structural_coverage(bridge)
        assert _status(result) == "gaps"
        assert "model.layers.{i}.self_attn.q_proj" in _gaps(result, "unmapped")
        assert result.details is not None
        # embed, the final norm and the head stay owned; nothing inside a block does.
        assert result.details["pattern_coverage"]["specific"] == 3
        assert result.details["pattern_coverage"]["share"] < 0.5

    def test_a_block_module_behind_a_generic_wrapper_is_a_gap(self) -> None:
        bridge, reference = _build(GRANITE, _granite(), _generic_mlp)
        result = coverage.benchmark_structural_coverage(bridge)
        assert _status(result) == "gaps"
        assert not _gaps(result, "unmapped")
        assert "model.layers.{i}.mlp.down_proj" in _gaps(result, "generic")
        report = _report(bridge, reference)
        assert report["passed"] is True
        assert 0 < report["coverage"] < 1 and report["coverage_basis"] == "ownership"
        assert {gap["kind"] for gap in report["gaps"]} == {"generic"}

    def test_coverage_counts_patterns_not_parameters(self) -> None:
        bridge, _ = _build(GRANITE, _granite(), _generic_mlp)
        details = coverage.benchmark_structural_coverage(bridge).details
        assert details is not None
        # Three of twelve patterns (the MLP's projections) are generic, whatever they weigh.
        assert details["pattern_coverage"] == {"specific": 9, "total": 12, "share": 0.75}
        assert details["parameter_share"]["generic"] != 0.25

    def test_a_tied_tensor_is_owned_once_by_its_best_owner(self) -> None:
        bridge, _ = _build(GRANITE, _granite(tie_word_embeddings=True))
        model = bridge.original_model
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
        assert _status(coverage.benchmark_structural_coverage(bridge)) == "pass"


class TestExpectedHooks:
    def test_passes_when_every_promised_hook_resolves_and_fires(self) -> None:
        bridge, _ = _build(GRANITE, _granite())
        result = coverage.benchmark_expected_hooks(bridge, TOKENS)
        assert _status(result) == "pass", result.message

    def test_a_promised_hook_nothing_backs_is_an_absent_hook_gap(self) -> None:
        bridge, _ = _build(GRANITE, _granite(), _shell_blocks)
        result = coverage.benchmark_expected_hooks(bridge, TOKENS)
        assert _status(result) == "gaps"
        assert "blocks.{i}.hook_resid_mid" in _gaps(result, "absent_hook")

    def test_a_promised_hook_that_never_fires_is_a_silent_hook_gap(self) -> None:
        bridge, _ = _build(GRANITE, _granite())
        # An alias re-pointed at a hook point that sits outside the forward path.
        for block in bridge.blocks:
            block.hook_off_path = HookPoint()
            block.hook_aliases = {**block.hook_aliases, "hook_resid_mid": "hook_off_path"}
        result = coverage.benchmark_expected_hooks(bridge, TOKENS)
        assert _status(result) == "gaps"
        assert _gaps(result, "silent_hook") == ["blocks.{i}.hook_resid_mid"]


class TestHookConsistency:
    CONFIG = dict(residual_multiplier=0.5, embedding_multiplier=3.0, logits_scaling=4.0)

    def test_passes_when_hooks_describe_the_scaled_computation(self) -> None:
        bridge, reference = _build(GRANITE, _granite(**self.CONFIG))
        result = coverage.benchmark_hook_consistency(bridge, reference, TOKENS)
        assert _status(result) == "pass", result.message
        assert result.details is not None
        assert [entry["part"] for entry in result.details["blocking"]] == [
            "logit_parity",
            "residual_sum",
            "logit_readout",
            "interventions",
        ]

    def test_fails_on_a_residual_stream_that_does_not_add_up(self) -> None:
        bridge, reference = _build(GRANITE, _granite(**self.CONFIG), _unscaled_residual)
        result = coverage.benchmark_hook_consistency(bridge, reference, TOKENS)
        assert _status(result) == "fail"
        assert result.details is not None
        # The forward is HuggingFace's own, so the logits still agree: only the hooks are wrong.
        assert result.details["logit_parity"]["status"] == "pass"
        assert result.details["residual_sum"]["status"] == "fail"

    def test_fails_when_the_final_residual_does_not_give_the_logits(self) -> None:
        bridge, reference = _build(GRANITE, _granite(**self.CONFIG), _unscaled_logits)
        result = coverage.benchmark_hook_consistency(bridge, reference, TOKENS)
        assert _status(result) == "fail"
        assert result.details is not None
        assert result.details["logit_readout"]["status"] == "fail"
        assert result.details["residual_sum"]["status"] == "pass"

    def test_fails_when_an_edit_at_a_hook_does_not_reach_the_logits(self) -> None:
        bridge, reference = _build(GRANITE, _granite(**self.CONFIG))
        projection = bridge.blocks[0].mlp.out

        def observe_only(*args: Any, **kwargs: Any) -> Any:
            output = projection.original_component(*args, **kwargs)
            projection.hook_out(output)
            return output

        # Fires the hook but discards what it returns, so interventions are silently dropped.
        object.__setattr__(projection, "forward", observe_only)
        result = coverage.benchmark_hook_consistency(
            bridge, reference, TOKENS, max_interventions=64
        )
        assert _status(result) == "fail"
        assert result.details is not None
        assert result.details["logit_parity"]["status"] == "pass"
        mismatched = [entry["hook"] for entry in result.details["interventions"]["mismatched"]]
        assert mismatched == ["blocks.0.mlp.out.hook_out"]

    def test_fails_when_the_bridge_and_reference_disagree(self) -> None:
        bridge, reference = _build(GRANITE, _granite(**self.CONFIG))
        with torch.no_grad():
            reference.model.layers[0].mlp.down_proj.weight.mul_(1.5)
        result = coverage.benchmark_hook_consistency(bridge, reference, TOKENS)
        assert _status(result) == "fail"
        assert result.details is not None
        assert result.details["logit_parity"]["status"] == "fail"


class TestScalingFactorCoverage:
    def _run(self, fault: Fault) -> BenchmarkResult:
        bridge, reference = _build(GRANITE, _granite(), fault)
        return coverage.benchmark_scaling_factor_coverage(
            bridge, reference, lambda hf_model: _boot(GRANITE, hf_model, fault), TOKENS
        )

    def test_identity_multipliers_hide_the_fault_from_the_checkpoint(self) -> None:
        bridge, reference = _build(GRANITE, _granite(), _unscaled_residual)
        result = coverage.benchmark_hook_consistency(bridge, reference, TOKENS)
        assert _status(result) == "pass", result.message

    def test_passes_when_the_adapter_models_every_multiplier(self) -> None:
        result = self._run(None)
        assert _status(result) == "pass", result.message
        assert result.details is not None
        assert set(result.details["exercised"]) == {
            "embedding_multiplier",
            "logits_scaling",
            "residual_multiplier",
            "attention_multiplier",
        }

    def test_fails_once_an_unmodelled_multiplier_leaves_identity(self) -> None:
        result = self._run(_unscaled_residual)
        assert _status(result) == "fail"
        assert "residual_multiplier" in result.message
        assert result.details is not None
        assert result.details["exercised"]["residual_multiplier"]["failed"] == ["residual_sum"]

    def test_skips_when_no_float_field_sits_at_identity(self) -> None:
        config = _granite(
            residual_multiplier=0.5,
            embedding_multiplier=3.0,
            logits_scaling=4.0,
            attention_multiplier=0.25,
        )
        bridge, reference = _build(GRANITE, config)
        result = coverage.benchmark_scaling_factor_coverage(
            bridge, reference, lambda hf_model: _boot(GRANITE, hf_model), TOKENS
        )
        assert _status(result) == "skip"


class TestLayerTypeCoverage:
    def _run(self, fault: Fault) -> BenchmarkResult:
        config = SmolLM3Config(**_DIMS, num_hidden_layers=2)
        assert set(config.no_rope_layers or []) == {1}, "fixture must lack a NoPE layer"
        bridge, reference = _build(SMOLLM3, config, fault)
        return coverage.benchmark_layer_type_coverage(
            bridge, reference, lambda hf_model: _boot(SMOLLM3, hf_model, fault), TOKENS
        )

    def test_one_layer_kind_hides_the_fault_from_the_checkpoint(self) -> None:
        bridge, reference = _build(
            SMOLLM3, SmolLM3Config(**_DIMS, num_hidden_layers=2), _always_rotate
        )
        result = coverage.benchmark_hook_consistency(bridge, reference, TOKENS)
        assert _status(result) == "pass", result.message

    def test_passes_when_the_adapter_handles_the_missing_kind(self) -> None:
        result = self._run(None)
        assert _status(result) == "pass", result.message
        assert result.details is not None
        assert "0" in result.details["fields"]["no_rope_layers"]["exercised"]

    def test_fails_on_a_layer_kind_the_checkpoint_lacks(self) -> None:
        result = self._run(_always_rotate)
        assert _status(result) == "fail"
        assert "no_rope_layers=0" in result.message
        assert result.details is not None
        failed = result.details["fields"]["no_rope_layers"]["exercised"]["0"]["failed"]
        assert "logit_parity" in failed

    def test_skips_when_the_config_has_no_per_layer_field(self) -> None:
        bridge, reference = _build(GRANITE, _granite())
        result = coverage.benchmark_layer_type_coverage(
            bridge, reference, lambda hf_model: _boot(GRANITE, hf_model), TOKENS
        )
        assert _status(result) == "skip"


class TestShrunkenVariants:
    """A checkpoint too large to rebuild is rebuilt small, with random weights, for C and E."""

    WIDE = dict(_DIMS, hidden_size=256, intermediate_size=512, vocab_size=5000)

    def test_shrinks_depth_width_and_vocabulary_and_still_runs(self) -> None:
        _, reference = _build(GRANITE, GraniteConfig(**self.WIDE, num_hidden_layers=6))
        small, note = coverage.shrunken_reference(reference)
        assert note["edits"] == {
            "num_hidden_layers": 2,
            "vocab_size": 1024,
            "hidden_size": 16,
            "intermediate_size": 32,
        }
        assert note["parameters"] < sum(p.numel() for p in reference.parameters()) / 20
        assert torch.isfinite(small(TOKENS).logits).all()

    def test_shrinking_depth_keeps_every_layer_kind_the_checkpoint_has(self) -> None:
        config = SmolLM3Config(**_DIMS, num_hidden_layers=8, no_rope_layer_interval=4)
        assert config.no_rope_layers == [1, 1, 1, 0, 1, 1, 1, 0]
        _, reference = _build(SMOLLM3, config)
        small, note = coverage.shrunken_reference(reference)
        assert note["edits"]["no_rope_layers"] == [1, 1, 1, 0]
        assert note["edits"]["num_hidden_layers"] == 4
        assert torch.isfinite(small(TOKENS).logits).all()

    def test_scaling_check_catches_the_fault_on_a_shrunken_rebuild(self) -> None:
        config = GraniteConfig(**self.WIDE, num_hidden_layers=6)
        cases: list[tuple[Fault, str]] = [(None, "pass"), (_unscaled_residual, "fail")]
        for fault, expected in cases:
            bridge, reference = _build(GRANITE, config, fault)

            def boot(hf_model: Any, fault: Fault = fault) -> Any:
                return _boot(GRANITE, hf_model, fault)

            result = coverage.benchmark_scaling_factor_coverage(
                bridge, reference, boot, TOKENS, max_variant_params=1e5
            )
            assert _status(result) == expected, result.message
            assert result.details is not None
            assert result.details["variant_base"]["weights"] == "random"
            parts = [entry["part"] for entry in result.details["blocking"]]
            assert "shrunken config: logit_parity" in parts

    def test_a_small_checkpoint_keeps_its_own_weights(self) -> None:
        bridge, reference = _build(GRANITE, _granite())
        result = coverage.benchmark_scaling_factor_coverage(
            bridge,
            reference,
            lambda hf_model: _boot(GRANITE, hf_model),
            TOKENS,
            max_variant_params=1e9,
        )
        assert result.details is not None and "variant_base" not in result.details


class TestReferenceComparison:
    def test_passes_against_itself_and_reports_differences_both_ways(self) -> None:
        full, _ = _build(GRANITE, _granite())
        shell, _ = _build(GRANITE, _granite(), _shell_blocks)
        full_surface = coverage.adapter_surface(full)
        assert (
            _status(coverage.benchmark_reference_comparison(full_surface, full_surface)) == "pass"
        )
        shell_surface = coverage.adapter_surface(shell)
        result = coverage.benchmark_reference_comparison(shell_surface, full_surface)
        assert _status(result) == "gaps"
        assert "blocks.{i}.attn.hook_out" in _gaps(result, "missing_from_reference")
        assert 0 < coverage.reference_coverage(shell_surface, full_surface) < 1
        reverse = coverage.benchmark_reference_comparison(full_surface, shell_surface)
        assert reverse.details is not None
        assert "blocks.{i}.attn.hook_out" in reverse.details["hooks"]["only_in_adapter"]

    def test_names_only_the_adapter_under_test_has_cost_nothing(self) -> None:
        full, _ = _build(GRANITE, _granite())
        reference = coverage.adapter_surface(full)
        draft = {key: names | {f"extra.{key}"} for key, names in reference.items()}
        result = coverage.benchmark_reference_comparison(draft, reference)
        assert _status(result) == "pass"
        assert coverage.reference_coverage(draft, reference) == 1.0
        assert result.details is not None
        assert result.details["hooks"]["only_in_adapter"] == ["extra.hooks"]


class TestResidualFloor:
    def test_holds_on_a_fully_mapped_adapter(self) -> None:
        bridge, _ = _build(GRANITE, _granite(residual_multiplier=0.5))
        result = coverage.benchmark_residual_floor(bridge, TOKENS)
        assert _status(result) == "pass", result.message

    def test_fails_on_a_shell_and_names_what_to_wrap(self) -> None:
        bridge, reference = _build(GRANITE, _granite(), _shell_blocks)
        result = coverage.benchmark_residual_floor(bridge, TOKENS)
        assert _status(result) == "fail"
        assert result.details is not None
        assert result.details["wrap"] == [
            "model.layers.{i}.input_layernorm",
            "model.layers.{i}.mlp",
            "model.layers.{i}.post_attention_layernorm",
            "model.layers.{i}.self_attn",
        ]
        assert _report(bridge, reference)["passed"] is False

    def test_fails_when_state_space_layers_are_unmapped_and_names_them(self) -> None:
        bridge, reference = _build(BAMBA, _bamba(), _unmapped_mixer)
        result = coverage.benchmark_residual_floor(bridge, TOKENS)
        assert _status(result) == "fail"
        assert result.details is not None
        # The attention layer in the middle still accounts for itself.
        assert sorted(sum(result.details["failing_blocks"].values(), [])) == [0, 2]
        assert result.details["wrap"] == ["model.layers.{i}.mamba"]
        assert "model.layers.{i}.mamba.in_proj" in result.details["modules_without_output_hook"]
        report = _report(bridge, reference)
        assert report["passed"] is False
        assert "residual_floor" in {finding["check"] for finding in report["blocking_findings"]}

    def test_a_generic_wrapper_on_each_state_space_layer_meets_the_floor(self) -> None:
        bridge, reference = _build(BAMBA, _bamba(), _generic_mixer)
        assert _status(coverage.benchmark_residual_floor(bridge, TOKENS)) == "pass"
        report = _report(bridge, reference)
        assert report["passed"] is True, report["blocking_findings"]
        assert 0 < report["coverage"] < 1
        internals = {gap["where"] for gap in report["gaps"] if gap["kind"] == "generic"}
        assert {"model.layers.{i}.mamba.in_proj", "model.layers.{i}.mamba.out_proj"} <= internals

    def test_the_correct_hybrid_adapter_passes_with_full_coverage(self) -> None:
        bridge, reference = _build(BAMBA, _bamba())
        report = _report(bridge, reference)
        assert report["passed"] is True, report["blocking_findings"]
        assert report["coverage"] == 1.0 and report["gaps"] == []

    def test_a_post_norm_block_meets_the_floor_through_its_own_norms(self) -> None:
        # GPT-1 norms each sum: n = ln_1(x + attn), out = ln_2(n + mlp). Nothing adds up
        # across the block, yet every step is a hooked output entering a hooked norm.
        config = OpenAIGPTConfig(vocab_size=128, n_embd=32, n_layer=2, n_head=4, n_positions=64)
        bridge, reference = _build("OpenAIGPTLMHeadModel", config)
        assert _status(coverage.benchmark_residual_floor(bridge, TOKENS)) == "pass"
        consistency = coverage.benchmark_hook_consistency(bridge, reference, TOKENS)
        assert consistency.details is not None
        assert consistency.details["residual_sum"]["status"] == "skip"
        assert _report(bridge, reference)["passed"] is True

    def test_a_term_inferred_from_the_stream_does_not_count(self) -> None:
        bridge, reference = _build(GRANITE, _granite(), _no_mlp_out_hook)
        # The library derives the MLP term as resid_post - resid_mid, so the sum cannot be wrong.
        consistency = coverage.benchmark_hook_consistency(bridge, reference, TOKENS)
        assert consistency.details is not None
        assert consistency.details["residual_sum"]["status"] == "pass"
        result = coverage.benchmark_residual_floor(bridge, TOKENS)
        assert _status(result) == "fail"
        assert "inferred from the stream" in result.message


class TestDeclaredGaps:
    def test_a_declared_gap_carries_its_reason_and_still_lowers_coverage(self) -> None:
        bridge, reference = _build(GRANITE, _granite(), _generic_mlp)
        undeclared = _report(bridge, reference)
        setattr(
            bridge.adapter,
            coverage.DECLARED_GAPS_ATTRIBUTE,
            {"model.layers.{i}.mlp": "delegated to HuggingFace's fused kernel"},
        )
        report = _report(bridge, reference)
        assert report["gaps"] and all(gap["declared"] for gap in report["gaps"])
        assert {gap["reason"] for gap in report["gaps"]} == {
            "delegated to HuggingFace's fused kernel"
        }
        assert report["coverage"] == undeclared["coverage"] < 1
        assert report["stale_declarations"] == []

    def test_a_declaration_for_something_fully_exposed_is_reported_as_stale(self) -> None:
        bridge, reference = _build(GRANITE, _granite())
        setattr(bridge.adapter, coverage.DECLARED_GAPS_ATTRIBUTE, {"model.layers.{i}.mlp": "n/a"})
        report = _report(bridge, reference)
        assert report["stale_declarations"] == ["model.layers.{i}.mlp"]
        assert report["gaps"] == []

    def test_a_declaration_cannot_clear_a_blocking_finding(self) -> None:
        bridge, reference = _build(GRANITE, _granite(), _shell_blocks)
        setattr(
            bridge.adapter,
            coverage.DECLARED_GAPS_ATTRIBUTE,
            {"model.layers": "blocks are opaque by design", "hook_resid_mid": "not exposed"},
        )
        report = _report(bridge, reference)
        assert report["passed"] is False
        assert all(gap["declared"] for gap in report["gaps"] if gap["kind"] == "unmapped")
        assert "residual_floor" in {finding["check"] for finding in report["blocking_findings"]}

    def test_an_adapter_without_the_attribute_or_with_a_malformed_one_still_runs(self) -> None:
        bridge, _ = _build(GRANITE, _granite())
        assert coverage.declared_gaps(bridge.adapter) == {}
        setattr(bridge.adapter, coverage.DECLARED_GAPS_ATTRIBUTE, ["not", "a", "mapping"])
        assert coverage.declared_gaps(bridge.adapter) == {}


class TestRunningOnABridgeDirectly:
    """The checks need nothing but a bridge: no checkpoint name, adapter file or tokenizer."""

    def test_reports_on_a_locally_built_bridge_with_no_other_inputs(self) -> None:
        bridge, _ = _build(GRANITE, _granite())
        report = runner.run_coverage_checks(bridge, test_input=TOKENS, source="local")
        assert report["passed"] is True, report["blocking_findings"]
        assert (report["coverage"], report["coverage_basis"]) == (1.0, "ownership")
        assert report["source"] == "local"
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        assert statuses == {
            "structural_coverage": "pass",
            "expected_hooks": "pass",
            "residual_floor": "pass",
            "layer_type_coverage": "skip",
            "hook_consistency": "pass",
            "scaling_factor_coverage": "pass",
            "reference_comparison": "skip",
        }
        # The multiplier variants were rebuilt and booted without a checkpoint to name.
        assert report["n_blocking_parts"] > 5

    def test_fails_a_misleading_adapter_the_same_way(self) -> None:
        config = _granite(residual_multiplier=0.5)
        bridge, _ = _build(GRANITE, config, _unscaled_residual)
        report = runner.run_coverage_checks(bridge, test_input=TOKENS)
        assert report["passed"] is False
        assert {"residual_floor", "hook_consistency"} <= {
            finding["check"] for finding in report["blocking_findings"]
        }

    def test_compares_with_a_reference_when_given_its_surface(self) -> None:
        full, _ = _build(GRANITE, _granite())
        generic, _ = _build(GRANITE, _granite(), _generic_mlp)
        report = runner.run_coverage_checks(
            generic, test_input=TOKENS, reference_surface=coverage.adapter_surface(full)
        )
        assert report["passed"] is True
        assert report["coverage_basis"] == "reference" and 0 < report["coverage"] < 1


def test_a_check_that_cannot_run_fails_instead_of_skipping() -> None:
    bridge, _ = _build(GRANITE, _granite())
    result = coverage.benchmark_hook_consistency(bridge, torch.nn.Linear(2, 2), TOKENS)
    assert _status(result) == "fail"
    assert result.message.startswith("could not run")


@pytest.mark.parametrize(
    "module,kind",
    [
        (torch.nn.Linear(2, 2), "linear"),
        (torch.nn.Embedding(4, 2), "embedding"),
        (torch.nn.LayerNorm(2), "norm"),
        (torch.nn.Conv1d(2, 2, 1), "conv"),
        (torch.nn.MultiheadAttention(2, 1), "parameters"),
    ],
)
def test_module_kind(module: torch.nn.Module, kind: str) -> None:
    assert coverage.module_kind(module) == kind


def test_every_library_component_has_a_role_in_the_ownership_map() -> None:
    import transformer_lens.model_bridge.generalized_components as components

    placed = (
        coverage.SUBTREE_OWNERS
        + coverage.CONTAINER_COMPONENTS
        + coverage.BOUNDARY_COMPONENTS
        + tuple(owner for owners in coverage.MODULE_KIND_OWNERS.values() for owner in owners)
    )
    unplaced = [
        name
        for name, value in vars(components).items()
        if isinstance(value, type)
        and issubclass(value, GeneralizedComponent)
        and not issubclass(value, placed)
    ]
    assert not unplaced, f"give these components a role in adapter_coverage: {unplaced}"
