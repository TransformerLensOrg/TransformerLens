"""Causal-swap trial-runner tests: real forward passes and hook installs on ``_ToyBridge``."""

from typing import Dict, Tuple

import pytest
import torch

from tests.unit.tools.conftest import D_MODEL, D_VOCAB, _ToyBridge
from transformer_lens.tools.analysis import JacobianLens
from transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark import (
    AnswerMetrics,
    BenchmarkCorpus,
    FunctionSpec,
    PromptTrialSpec,
    compute_answer_metrics,
    run_causal_swap_benchmark,
    run_causal_swap_trial,
)
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
)

# coordinate_patch_hooks warns on every call; test_jacobian_lens_coordinate_patch_hooks.py
# covers that warning.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

PROMPT = "a toy prompt"
SOLVE_K = 8
# Generous: these tests cover runner mechanics; tolerance matching is tested model-free.
CONTROL_TOLERANCE = 4.0


def _word_for_token_id(token_id: int) -> str:
    """Word whose ``" " + word`` resolves to ``token_id`` under the toy tokenizer, which maps a
    string to ``len(s) % D_VOCAB``."""
    length = (token_id - 1) % D_VOCAB
    if length == 0:
        length = D_VOCAB
    return "x" * length


@pytest.fixture(scope="module")
def toy_bridge() -> _ToyBridge:
    return _ToyBridge()


@pytest.fixture(scope="module")
def toy_lens() -> JacobianLens:
    return JacobianLens(
        {0: torch.eye(D_MODEL), 1: torch.eye(D_MODEL), 2: torch.eye(D_MODEL)},
        n_prompts=1,
        d_model=D_MODEL,
    )


def _spec_for_source_id(source_id: int) -> PromptTrialSpec:
    return PromptTrialSpec(
        function="f",
        source=_word_for_token_id(source_id),
        target=_word_for_token_id((source_id + 1) % D_VOCAB),
        prompt=PROMPT,
        source_answer=_word_for_token_id((source_id + 2) % D_VOCAB),
        target_answer=_word_for_token_id((source_id + 3) % D_VOCAB),
    )


def _spec_with_active_source(lens: JacobianLens, model: _ToyBridge, layer: int) -> PromptTrialSpec:
    """Spec whose source is in ``layer``'s support for ``PROMPT``, found by decomposing because
    the toy activations are not hand-computable."""
    support = lens.decompose(model, PROMPT, layer=layer, position=-1, k=SOLVE_K).support
    return _spec_for_source_id(int(support[0]))


def _spec_with_inactive_source(
    lens: JacobianLens, model: _ToyBridge, layer: int
) -> PromptTrialSpec:
    """Spec whose source is outside ``layer``'s support; ``SOLVE_K < D_VOCAB`` guarantees one."""
    support = {
        int(token)
        for token in lens.decompose(model, PROMPT, layer=layer, position=-1, k=SOLVE_K).support
    }
    return _spec_for_source_id(
        next(token_id for token_id in range(D_VOCAB) if token_id not in support)
    )


def test_run_causal_swap_trial_shares_decomposition_cache_between_real_and_control(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    cache: Dict[Tuple[int, int, int], JSpaceDecomposition] = {}
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=1,
        decomposition_cache=cache,
        control_tolerance=CONTROL_TOLERANCE,
        k=SOLVE_K,
    )
    assert result.status == "ok"
    assert len(cache) == 1  # one (layer, batch, position) key, reused by the control call
    # A key mismatch would silently double the solve cost without failing anything else.
    seq_len = toy_bridge.to_tokens(spec.prompt).shape[1]
    assert set(cache) == {(1, 0, seq_len - 1)}


def test_run_causal_swap_trial_conditions_match_independently_patched_passes(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    # Compared with hand-installed hooks, not the baseline: the baseline scores the source
    # answer, so it would miss a hardcoded alpha=0.0 or swapped real/control assignments.
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=1,
        control_tolerance=CONTROL_TOLERANCE,
        k=SOLVE_K,
    )
    assert result.status == "ok"
    assert result.control_token_id is not None

    source_id = toy_bridge.to_single_token(f" {spec.source}")
    target_id = toy_bridge.to_single_token(f" {spec.target}")
    target_answer_id = toy_bridge.to_single_token(f" {spec.target_answer}")
    tokens = toy_bridge.to_tokens(spec.prompt)

    def _patched_metrics(condition_target_id: int) -> AnswerMetrics:
        cache: Dict[Tuple[int, int, int], JSpaceDecomposition] = {}
        cache[(1, 0, tokens.shape[1] - 1)] = toy_lens.decompose(
            toy_bridge, spec.prompt, layer=1, position=-1, k=SOLVE_K
        )
        hooks = toy_lens.coordinate_patch_hooks(
            toy_bridge,
            source_id,
            condition_target_id,
            layers=[1],
            positions=[-1],
            decomposition_cache=cache,
            k=SOLVE_K,
            alpha=1.0,
        )
        with toy_bridge.hooks(fwd_hooks=hooks), torch.no_grad():
            logits = toy_bridge(tokens)[0, -1].float()
        return compute_answer_metrics(logits, target_answer_id)

    real_metrics = _patched_metrics(target_id)
    control_metrics = _patched_metrics(result.control_token_id)
    assert result.real_target_metrics == real_metrics
    assert result.control_target_metrics == control_metrics
    # Otherwise the equalities above could hold vacuously under a no-op patch.
    with torch.no_grad():
        unpatched = compute_answer_metrics(toy_bridge(tokens)[0, -1].float(), target_answer_id)
    assert real_metrics != unpatched
    assert control_metrics != unpatched


def test_run_causal_swap_trial_records_skip_without_raising_on_inactive_source(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    spec = _spec_with_inactive_source(toy_lens, toy_bridge, layer=2)
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=2,
        control_tolerance=CONTROL_TOLERANCE,
        k=SOLVE_K,
    )
    assert result.status == "skipped_source_inactive"
    assert result.real_target_metrics is None
    assert result.control_target_metrics is None
    assert result.error is not None


def test_run_causal_swap_trial_records_skip_when_no_control_token_matches(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    # Zero tolerance empties the toy pool; the trial must skip, not raise or take the nearest.
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=1,
        control_tolerance=0.0,
        k=SOLVE_K,
    )
    assert result.status == "skipped_no_control_token"
    assert result.control_token_id is None
    assert result.real_target_metrics is None
    assert result.control_target_metrics is None
    assert result.error is not None


def test_run_causal_swap_trial_populates_baseline_regardless_of_status(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    spec = _spec_with_inactive_source(toy_lens, toy_bridge, layer=2)
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=2,
        control_tolerance=CONTROL_TOLERANCE,
        k=SOLVE_K,
    )
    # The baseline pass is unhooked, so an inactive source does not affect it.
    assert result.baseline is not None


def test_run_causal_swap_trial_skips_before_installing_hooks(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A skip derived from catching ValueError inside the hooked pass would reach this stub.
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("coordinate_patch_hooks must not run for an inactive source")

    monkeypatch.setattr(toy_lens, "coordinate_patch_hooks", _boom)
    spec = _spec_with_inactive_source(toy_lens, toy_bridge, layer=2)
    cache: Dict[Tuple[int, int, int], JSpaceDecomposition] = {}
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=2,
        decomposition_cache=cache,
        control_tolerance=CONTROL_TOLERANCE,
        k=SOLVE_K,
    )
    assert result.status == "skipped_source_inactive"
    # The up-front check seeds the cache before the early return.
    seq_len = toy_bridge.to_tokens(spec.prompt).shape[1]
    assert set(cache) == {(2, 0, seq_len - 1)}


def test_run_causal_swap_trial_propagates_same_id_source_and_target(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    # A protocol error, not a skip: a skip would drop it from both denominators.
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    same_id_spec = PromptTrialSpec(
        function=spec.function,
        source=spec.source,
        target=spec.source,
        prompt=spec.prompt,
        source_answer=spec.source_answer,
        target_answer=spec.target_answer,
    )
    with pytest.raises(ValueError, match="same token id"):
        run_causal_swap_trial(
            toy_lens,
            toy_bridge,
            same_id_spec,
            layer=1,
            control_tolerance=CONTROL_TOLERANCE,
            k=SOLVE_K,
        )


def test_run_causal_swap_trial_records_one_control_draw_per_seed(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    cache: Dict[Tuple[int, int, int], JSpaceDecomposition] = {}
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=1,
        decomposition_cache=cache,
        control_tolerance=CONTROL_TOLERANCE,
        control_seeds=(0, 1, 2),
        k=SOLVE_K,
    )
    assert result.status == "ok"
    assert [draw.seed for draw in result.control_draws] == [0, 1, 2]
    # Extra draws add forward passes but no vocabulary-scale solves.
    assert len(cache) == 1


def test_run_causal_swap_trial_primary_control_mirrors_the_first_seed(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=1,
        control_tolerance=CONTROL_TOLERANCE,
        control_seeds=(3, 4),
        k=SOLVE_K,
    )
    assert result.status == "ok"
    assert result.control_draws[0].seed == 3
    assert result.control_token_id == result.control_draws[0].token_id
    assert result.control_target_metrics == result.control_draws[0].metrics


def test_run_causal_swap_trial_control_draws_can_differ_across_seeds(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    # If every seed collapsed to one token the control arm's spread would be unmeasurable.
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=1,
        control_tolerance=CONTROL_TOLERANCE,
        control_seeds=tuple(range(8)),
        k=SOLVE_K,
    )
    assert result.status == "ok"
    assert len({draw.token_id for draw in result.control_draws}) > 1


def test_run_causal_swap_trial_skips_when_a_seed_lacks_a_control(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    # Zero tolerance empties every seed's pool; no partial draw list is reported.
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    result = run_causal_swap_trial(
        toy_lens,
        toy_bridge,
        spec,
        layer=1,
        control_tolerance=0.0,
        control_seeds=(0, 1),
        k=SOLVE_K,
    )
    assert result.status == "skipped_no_control_token"
    assert result.control_draws == []


def test_run_causal_swap_trial_rejects_an_empty_seed_list(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    spec = _spec_with_active_source(toy_lens, toy_bridge, layer=1)
    with pytest.raises(ValueError, match="at least one seed"):
        run_causal_swap_trial(
            toy_lens,
            toy_bridge,
            spec,
            layer=1,
            control_tolerance=CONTROL_TOLERANCE,
            control_seeds=(),
            k=SOLVE_K,
        )


def _rigged_corpus(model: _ToyBridge) -> BenchmarkCorpus:
    """Two concepts whose answers are rigged so one is baseline-capable and one is not."""
    template = "prompt {arg} end"
    correct_concept = "AAAA"
    wrong_concept = "BBBBBB"

    def _actual_top1(concept: str) -> int:
        prompt = template.format(arg=concept)
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
        return int(logits.argmax().item())

    correct_answer = _word_for_token_id(_actual_top1(correct_concept))
    wrong_answer = _word_for_token_id((_actual_top1(wrong_concept) + 1) % D_VOCAB)
    return BenchmarkCorpus(
        name="toy",
        concepts=(correct_concept, wrong_concept),
        functions=(
            FunctionSpec(
                name="f",
                template=template,
                answers={correct_concept: correct_answer, wrong_concept: wrong_answer},
            ),
        ),
    )


def test_run_causal_swap_benchmark_excludes_baseline_incapable_prompts(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    corpus = _rigged_corpus(toy_bridge)
    trials, excluded = run_causal_swap_benchmark(
        toy_lens,
        toy_bridge,
        corpus,
        layers=[1],
        control_tolerance=CONTROL_TOLERANCE,
        k=SOLVE_K,
    )
    assert len(excluded) == 1
    assert excluded[0].function == "f" and excluded[0].source == "BBBBBB"
    assert all(not (t.function == "f" and t.source == "BBBBBB") for t in trials)
    assert all(t.function == "f" and t.source == "AAAA" for t in trials)


def test_run_causal_swap_benchmark_is_deterministic_given_seed(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    corpus = _rigged_corpus(toy_bridge)
    first, first_excluded = run_causal_swap_benchmark(
        toy_lens,
        toy_bridge,
        corpus,
        layers=[1],
        control_tolerance=CONTROL_TOLERANCE,
        control_seeds=(7, 8),
        k=SOLVE_K,
    )
    second, second_excluded = run_causal_swap_benchmark(
        toy_lens,
        toy_bridge,
        corpus,
        layers=[1],
        control_tolerance=CONTROL_TOLERANCE,
        control_seeds=(7, 8),
        k=SOLVE_K,
    )
    assert first == second
    assert first_excluded == second_excluded
