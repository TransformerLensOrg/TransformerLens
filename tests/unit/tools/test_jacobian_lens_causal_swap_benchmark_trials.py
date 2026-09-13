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
    BenchmarkCorpus,
    FunctionSpec,
    PromptTrialSpec,
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
        control_seed=7,
        k=SOLVE_K,
    )
    second, second_excluded = run_causal_swap_benchmark(
        toy_lens,
        toy_bridge,
        corpus,
        layers=[1],
        control_tolerance=CONTROL_TOLERANCE,
        control_seed=7,
        k=SOLVE_K,
    )
    assert first == second
    assert first_excluded == second_excluded
