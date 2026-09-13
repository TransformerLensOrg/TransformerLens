"""Causal coordinate-swap benchmark for ``JacobianLens.coordinate_patch_hooks``.

Measures whether an anchored J-space coordinate edit installed live inside a forward pass
via ``coordinate_patch_hooks`` causes a directional change in model output, under three
controls: baseline-capability filtering (only intervene on prompts the model already
answers correctly), norm-matched random-atom controls (isolate "this concept mattered"
from "any edit of similar magnitude would have mattered"), and bootstrap uncertainty on
every reported rate.

This module is layered bottom-up and built out across several stages. This stage is
model-free: it establishes the prompt corpus schema and the rank/margin metric shared by
every later stage (baseline filtering, control-token selection, the trial runner, and
artifact serialization).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Container, Dict, Iterator, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class FunctionSpec:
    """A templated prompt function evaluated over a shared set of concepts.

    ``template`` takes a single ``{arg}`` placeholder, e.g. ``"The capital of {arg} is"``.
    ``answers`` maps each concept to its answer word under this function, e.g.
    ``{"France": "Paris"}``.
    """

    name: str
    template: str
    answers: Dict[str, str]


@dataclass(frozen=True)
class BenchmarkCorpus:
    """A named set of concepts and the prompt functions evaluated over them."""

    name: str
    concepts: Sequence[str]
    functions: Sequence[FunctionSpec]


@dataclass(frozen=True)
class PromptTrialSpec:
    """One (function, ordered source/target concept pair) prompt instance."""

    function: str
    source: str
    target: str
    prompt: str
    source_answer: str
    target_answer: str


def iter_prompt_trials(corpus: BenchmarkCorpus) -> Iterator[PromptTrialSpec]:
    """Yields one spec per (function, ordered source/target concept pair).

    Ordered pairs are every element of ``itertools.permutations(corpus.concepts, 2)``, the
    same cross product Jacobian_Lens_Demo.ipynb's country benchmark already uses, now as
    tested library code instead of a notebook cell.
    """
    for function in corpus.functions:
        for source, target in itertools.permutations(corpus.concepts, 2):
            yield PromptTrialSpec(
                function=function.name,
                source=source,
                target=target,
                prompt=function.template.format(arg=source),
                source_answer=function.answers[source],
                target_answer=function.answers[target],
            )


@dataclass(frozen=True)
class AnswerMetrics:
    """Rank/margin/tie metrics for one target token against one next-token logit vector."""

    top1_token_id: int
    target_rank: int
    target_is_top1: bool
    target_tied_for_top: bool
    target_logit_margin: float


def compute_answer_metrics(logits: torch.Tensor, target_token_id: int) -> AnswerMetrics:
    """Computes rank/margin/tie metrics for ``target_token_id`` against ``logits``.

    Ports Jacobian_Lens_Demo.ipynb's ``_target_metrics`` cell verbatim (arithmetic
    unchanged, only renamed and restructured into a dataclass), so results stay directly
    comparable with that notebook's already-reviewed success/rank definitions.
    """
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("expected one-dimensional next-token logits")
    if not 0 <= target_token_id < logits.shape[0]:
        raise ValueError("target token id is outside the vocabulary")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")

    target_logit = logits[target_token_id]
    top_logit = logits.max()
    top1_token_id = int(logits.argmax().item())
    top_logit_tie_count = int((logits == top_logit).sum().item())
    competitors = torch.cat((logits[:target_token_id], logits[target_token_id + 1 :]))
    return AnswerMetrics(
        top1_token_id=top1_token_id,
        target_rank=int((logits > target_logit).sum().item()) + 1,
        target_is_top1=top1_token_id == target_token_id,
        target_tied_for_top=bool(target_logit == top_logit) and top_logit_tie_count > 1,
        target_logit_margin=float((target_logit - competitors.max()).item()),
    )


@dataclass(frozen=True)
class BaselineRecord:
    """A source prompt's own-answer metrics under the unperturbed baseline forward pass."""

    function: str
    source: str
    prompt: str
    metrics: AnswerMetrics


def filter_baseline_capable(
    baselines: Sequence[BaselineRecord],
) -> Tuple[List[BaselineRecord], List[BaselineRecord]]:
    """Splits baseline records into (capable, excluded) prompts.

    A prompt is baseline-capable when the model's own deterministic argmax already matches
    the source's answer (``metrics.target_is_top1``); only such prompts are eligible for
    later intervention trials, so an edit's effect is never measured against a prompt the
    unperturbed model already gets wrong. Order-preserving in both outputs; never mutates
    ``baselines``.
    """
    capable = [record for record in baselines if record.metrics.target_is_top1]
    excluded = [record for record in baselines if not record.metrics.target_is_top1]
    return capable, excluded


def select_norm_matched_control_token(
    dictionary: torch.Tensor,
    target_token_id: int,
    excluded_ids: Container[int],
    *,
    tolerance: float = 0.1,
    seed: int = 0,
) -> int:
    """Deterministically selects a norm-matched control token id.

    A candidate token id ``t`` qualifies when its ``dictionary`` atom norm is within
    ``tolerance`` (relative to the target token's atom norm) and ``t`` is neither
    ``target_token_id`` nor a member of ``excluded_ids``. One qualifying candidate is picked
    with a seeded ``torch.Generator`` so the same ``seed`` always yields the same control
    token. Raises ``ValueError`` if no candidate qualifies; the tolerance is never silently
    widened and selection never falls back to the globally nearest atom.
    """
    if dictionary.ndim != 2:
        raise ValueError(
            f"dictionary must be 2-D [num_atoms, d_model], got shape {tuple(dictionary.shape)}"
        )
    atom_norms = dictionary.float().norm(dim=1)
    target_norm = atom_norms[target_token_id]
    within_tolerance = (atom_norms - target_norm).abs() <= tolerance * target_norm
    candidates = [
        token_id
        for token_id in range(dictionary.shape[0])
        if token_id != target_token_id
        and token_id not in excluded_ids
        and bool(within_tolerance[token_id])
    ]
    if not candidates:
        raise ValueError(
            "no candidate token within relative tolerance "
            f"{tolerance} of target_token_id={target_token_id}'s atom norm "
            f"({float(target_norm):.4f})"
        )
    generator = torch.Generator(device=dictionary.device).manual_seed(seed)
    pick = int(torch.randint(len(candidates), (1,), generator=generator).item())
    return candidates[pick]
