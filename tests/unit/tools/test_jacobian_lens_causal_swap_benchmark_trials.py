"""Trial-runner tests for the causal coordinate-swap benchmark.

Unlike the corpus/metrics/filter/control-token tests, these exercise real forward passes and
real ``model.hooks(fwd_hooks=...)`` installs against the shared ``_ToyBridge`` fixture -- a real
``TransformerBridge`` subclass, no Hugging Face download required.
"""

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

# ``coordinate_patch_hooks`` always warns once per call, naming the layer/position counts that
# will perform a live vocabulary-scale solve; this is expected and already covered by its own
# tests (test_jacobian_lens_coordinate_patch_hooks.py), so it is not re-asserted here.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

PROMPT = "a toy prompt"
SOLVE_K = 8
# Generous on purpose: these tests exercise runner mechanics (cache sharing, skip recording,
# baseline filtering, determinism), not tolerance-matching precision, which
# test_jacobian_lens_causal_swap_benchmark.py already covers directly.
CONTROL_TOLERANCE = 4.0


def _word_for_token_id(token_id: int) -> str:
    """Builds a word that resolves back to ``token_id`` under the toy bridge's length-based
    tokenizer (``to_single_token(s) == len(s) % D_VOCAB``, so a " "-prefixed word of length
    ``token_id - 1`` (mod ``D_VOCAB``) resolves to ``token_id``)."""
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


def _spec_with_active_source(lens: JacobianLens, model: _ToyBridge, layer: int) -> PromptTrialSpec:
    """Builds a spec whose source concept is a real active atom in ``layer``'s support for
    ``PROMPT`` -- discovered via a real decomposition rather than guessed, since the toy model's
    activations are not hand-computable in advance."""
    support = lens.decompose(model, PROMPT, layer=layer, position=-1, k=SOLVE_K).support
    source_id = int(support[0])
    target_id = (source_id + 1) % D_VOCAB
    source_answer_id = (source_id + 2) % D_VOCAB
    target_answer_id = (source_id + 3) % D_VOCAB
    return PromptTrialSpec(
        function="f",
        source=_word_for_token_id(source_id),
        target=_word_for_token_id(target_id),
        prompt=PROMPT,
        source_answer=_word_for_token_id(source_answer_id),
        target_answer=_word_for_token_id(target_answer_id),
    )


def _spec_with_inactive_source(
    lens: JacobianLens, model: _ToyBridge, layer: int
) -> PromptTrialSpec:
    """Builds a spec whose source concept is provably absent from ``layer``'s active support
    for ``PROMPT`` -- ``k=SOLVE_K < D_VOCAB`` guarantees at least one token id is left out."""
    support = {
        int(token)
        for token in lens.decompose(model, PROMPT, layer=layer, position=-1, k=SOLVE_K).support
    }
    source_id = next(token_id for token_id in range(D_VOCAB) if token_id not in support)
    target_id = (source_id + 1) % D_VOCAB
    source_answer_id = (source_id + 2) % D_VOCAB
    target_answer_id = (source_id + 3) % D_VOCAB
    return PromptTrialSpec(
        function="f",
        source=_word_for_token_id(source_id),
        target=_word_for_token_id(target_id),
        prompt=PROMPT,
        source_answer=_word_for_token_id(source_answer_id),
        target_answer=_word_for_token_id(target_answer_id),
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
    # The runner seeds the cache under the hook's own key before installing hooks, so the
    # first firing is a hit rather than a second vocabulary-scale solve. Pin the key format:
    # a mismatch would silently double the solve cost without failing any other assertion.
    seq_len = toy_bridge.to_tokens(spec.prompt).shape[1]
    assert set(cache) == {(1, 0, seq_len - 1)}


def test_run_causal_swap_trial_conditions_match_independently_patched_passes(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    # Each condition must be exactly the pass its hooks produce when installed by hand.
    # Comparing against the trial's baseline would catch neither a hardcoded alpha=0.0 nor
    # swapped real/control assignments, because the baseline is scored on the source answer
    # rather than the target's.
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
    # A no-op patch would leave both conditions at the unpatched pass, so pin that each
    # condition actually moves the target's metrics; otherwise the equalities above could
    # hold vacuously.
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
    # A zero tolerance admits only atoms at the target's exact displacement from the source,
    # which the toy dictionary does not contain, so the pool is empty. The trial must record
    # its own skip status rather than raising or falling back to the nearest atom.
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
    # The baseline forward pass has no hooks installed, so it is unaffected by the source
    # being inactive for the (unrelated) intervention conditions.
    assert result.baseline is not None


def test_run_causal_swap_trial_skips_before_installing_hooks(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deciding the skip up front means the hooks are never built for an inactive source. If
    # the status instead came from catching a ValueError raised inside the hooked forward
    # pass, this stub would be reached and the test would fail. The cache is already seeded
    # by the up-front check, so the early return leaves it populated.
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
    seq_len = toy_bridge.to_tokens(spec.prompt).shape[1]
    assert set(cache) == {(2, 0, seq_len - 1)}


def test_run_causal_swap_trial_propagates_same_id_source_and_target(
    toy_lens: JacobianLens, toy_bridge: _ToyBridge
) -> None:
    # A same-id source and target is a protocol error, not a per-trial skip. Filing it as a
    # skip would drop it from both denominators and hide the mistake.
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
    # Every draw reuses the one seeded decomposition, so the extra draws add forward passes
    # but no vocabulary-scale solves.
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
    # The point of drawing several seeds is that the control arm has its own spread. If every
    # seed collapsed to one token the spread would be unmeasurable and this test would fail.
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
    # A zero tolerance empties the pool for every seed, so the trial skips rather than
    # reporting a partial set of draws.
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
    # Without a seed there is no control arm at all, so the trial cannot report a
    # real-vs-control gap. Fail loudly rather than indexing an empty draw list.
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
    """One function, two concepts: "correct"'s own baseline answer is rigged to be the model's
    actual top1 (baseline-capable); "wrong"'s is rigged to not be (baseline-incapable)."""
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
