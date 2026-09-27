"""Integration guard: attribution patching and faithfulness on a real Bridge. Bridge.

The model-free unit suite runs against ``_LinearToyBridge``, which overrides
``hook_dict``, ``hooks()``, and ``check_hooks_to_add`` — the three behaviours the
two real-Bridge failures depend on. Its hook points have no conversion and its
gate check is a no-op, so a green unit suite says nothing about a real Bridge.

This test boots a real GPT-2 Bridge and exercises the paths the toy bridge
hides:

- gradients captured *through* hook conversions — ``blocks.*.attn.hook_z`` hands a
  reshaped ``[batch, seq, n_heads, d_head]`` view to the forward hook, so the
  ``attn_head_out`` family only scores if the backward hook delivers the gradient
  in that converted shape; and
- the default ``names_filter`` (``None``) falling back to the node hook set on a
  real ``hook_dict``, which also exposes gated points (``hook_mlp_in``,
  ``attn.hook_result``, split-QKV inputs) that ``add_hook`` would reject; and
- the edge-ablation rewrite, whose per-head fork hooks and pre-LN placement only
  exist on a real attention bridge.
"""

from __future__ import annotations

import math

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.attribution_patching import (
    EdgeAttributionConfig,
    attribution_patch,
    faithfulness,
)

CLEAN_PROMPT = "The capital of France is"
CORRUPT_PROMPT = "The capital of Russia is"


@pytest.fixture(scope="module")
def gpt2_bridge():
    return TransformerBridge.boot_transformers("gpt2", device="cpu", dtype=torch.float32)


def _logit_diff_metric(answer_id: int, wrong_id: int):
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logits[0, -1, answer_id] - logits[0, -1, wrong_id]

    return metric


def test_attribution_patch_scores_every_node_finite_on_real_bridge(gpt2_bridge) -> None:
    clean = gpt2_bridge.to_tokens(CLEAN_PROMPT)
    corrupt = gpt2_bridge.to_tokens(CORRUPT_PROMPT)
    assert clean.shape == corrupt.shape, "prompts must tokenize to the same length"

    answer_id = int(gpt2_bridge.to_tokens(" Paris")[0, -1].item())
    wrong_id = int(gpt2_bridge.to_tokens(" Moscow")[0, -1].item())
    metric_fn = _logit_diff_metric(answer_id, wrong_id)

    # Default config, no explicit names_filter: exercises the node-hook-set fallback.
    result = attribution_patch(gpt2_bridge, clean, corrupt, metric_fn)

    # Scores exactly the node graph: embed at every position, plus per-layer
    # per-head attention outputs and per-layer MLP outputs.
    seq_len = int(clean.shape[1])
    n_layers = int(gpt2_bridge.cfg.n_layers)
    n_heads = int(gpt2_bridge.cfg.n_heads)
    expected_count = seq_len * (1 + n_layers * (n_heads + 1))
    assert len(result.node_scores) == expected_count
    assert result.edge_scores == {}

    for node, score in result.node_scores.items():
        assert math.isfinite(score), f"non-finite score for {node}"

    # All three node families are present — attn_head_out is the family the hook
    # conversion bug broke, mlp_out and embed round out the node graph.
    families = {node.kind for node in result.node_scores}
    assert families == {"embed", "attn_head_out", "mlp_out"}


def test_faithfulness_recovers_most_of_the_metric_on_gpt2_small(gpt2_bridge) -> None:
    """A ranked edge circuit recovers far more of the gap than a random one.

    The ablation rewrites each reader's input at its pre-LN fork hook, so this
    only works if the real Bridge's split-QKV and MLP-entry hooks sit where the
    rewrite assumes. A circuit built from the top-ranked edges should recover a
    large share of the clean-to-corrupt gap; a random edge set of the same size
    should recover essentially none, which is what makes the number meaningful
    rather than an artifact of ablating almost nothing.

    Thresholds are set from measured values on this exact model and prompt pair
    (gpt2-small, fp32, CPU): the top 200 edges recover about 0.62 of the gap,
    while a random 200 of the 194,946 edges recover about 0.00. Recovery grows
    with the budget -- roughly 0.29 at 50 edges and 0.997 at 1000 -- so the
    budget is fixed here rather than left to drift.
    """
    clean = gpt2_bridge.to_tokens(CLEAN_PROMPT)
    corrupt = gpt2_bridge.to_tokens(CORRUPT_PROMPT)
    assert clean.shape == corrupt.shape, "prompts must tokenize to the same length"

    answer_id = int(gpt2_bridge.to_tokens(" Paris")[0, -1].item())
    wrong_id = int(gpt2_bridge.to_tokens(" Moscow")[0, -1].item())
    metric_fn = _logit_diff_metric(answer_id, wrong_id)

    result = attribution_patch(
        gpt2_bridge,
        clean,
        corrupt,
        metric_fn,
        config=EdgeAttributionConfig(granularity="edge"),
    )
    budget = 200
    ranked = result.top_edges(k=budget)
    assert len(ranked) == budget

    report = faithfulness(gpt2_bridge, clean, corrupt, metric_fn, ranked)

    assert math.isfinite(report.recovered)
    assert report.circuit_size == budget
    assert report.total_edges > report.circuit_size
    assert report.full_metric != pytest.approx(report.corrupt_metric)
    assert report.recovered > 0.5

    # A random edge set of the same size recovers essentially nothing, so the
    # ranking is doing real work rather than the number being an artifact of the
    # circuit's size.
    all_edges = list(result.edge_scores)
    generator = torch.Generator().manual_seed(0)
    sampled = torch.randperm(len(all_edges), generator=generator)[:budget].tolist()
    random_report = faithfulness(
        gpt2_bridge, clean, corrupt, metric_fn, [all_edges[index] for index in sampled]
    )

    assert random_report.circuit_size == report.circuit_size
    assert random_report.recovered < 0.1
    assert report.recovered > random_report.recovered + 0.4
