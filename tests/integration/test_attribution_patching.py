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
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.attribution_patching import (
    EdgeAttributionConfig,
    GradientCache,
    Node,
    NodeKind,
    _ablate_edges,
    _edge_hook_flags,
    _edge_hook_names,
    attribution_patch,
    enumerate_edges,
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


def _capture_hook(name: str, cache: dict[str, torch.Tensor]) -> Callable[..., None]:
    def capture(tensor: torch.Tensor, *, hook: Any) -> None:
        del hook
        cache[name] = tensor.detach().clone()

    return capture


def _capture_gpt2_forward(
    model: TransformerBridge,
    tokens: torch.Tensor,
    overrides: list[tuple[str, Callable]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cache: dict[str, torch.Tensor] = {}
    observers = [
        (name, _capture_hook(name, cache)) for name in _edge_hook_names(model.cfg.n_layers)
    ]
    with torch.no_grad(), model.hooks(fwd_hooks=(overrides or []) + observers):
        logits = model(tokens)
    return logits, cache


@pytest.fixture(scope="module")
def gpt2_edge_endpoints(gpt2_bridge):
    clean = gpt2_bridge.to_tokens(CLEAN_PROMPT)
    corrupt = gpt2_bridge.to_tokens(CORRUPT_PROMPT)
    assert clean.shape == corrupt.shape
    assert gpt2_bridge.original_model.config._attn_implementation == "eager"
    with _edge_hook_flags(gpt2_bridge):
        clean_logits, clean_values = _capture_gpt2_forward(gpt2_bridge, clean)
        _, corrupt_values = _capture_gpt2_forward(gpt2_bridge, corrupt)
    cache = GradientCache(clean_values, {}, torch.tensor(0.0))
    edges = enumerate_edges(gpt2_bridge, cache)
    return clean, clean_logits, clean_values, corrupt_values, edges


def _override_reader(index: tuple[int, ...], value: torch.Tensor) -> Callable[..., torch.Tensor]:
    def override(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
        del hook
        replaced = tensor.clone()
        replaced[index] = value
        return replaced

    return override


def _capture_gpt2_ablation(
    model: TransformerBridge,
    tokens: torch.Tensor,
    edges: list[tuple[Node, Node]],
    circuit: list[tuple[Node, Node]],
    replacements: dict[str, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cache: dict[str, torch.Tensor] = {}
    original_hooks = model.hooks

    @contextmanager
    def hooks(fwd_hooks: list[tuple[str, Callable]], **kwargs: Any) -> Iterator[Any]:
        # Appending observers preserves the production correction order.
        observers = [
            (name, _capture_hook(name, cache)) for name in _edge_hook_names(model.cfg.n_layers)
        ]
        with original_hooks(fwd_hooks=fwd_hooks + observers, **kwargs):
            yield model

    before = _bridge_hook_state(model)
    with monkeypatch.context() as patch:
        patch.setattr(model, "hooks", hooks)
        logits = _ablate_edges(model, tokens, edges, circuit, replacements)
    assert _bridge_hook_state(model) == before
    return logits, cache


@pytest.mark.parametrize(
    "kind,name,layer,head",
    [
        ("q_input", "blocks.1.attn.hook_q_input", 1, 3),
        ("k_input", "blocks.1.attn.hook_k_input", 1, 4),
        ("v_input", "blocks.1.attn.hook_v_input", 1, 5),
        ("mlp_in", "blocks.1.hook_mlp_in", 1, None),
        ("logits", "blocks.11.hook_resid_post", 11, None),
    ],
)
def test_partial_gpt2_ablation_matches_explicit_reader_override(
    gpt2_bridge,
    gpt2_edge_endpoints,
    monkeypatch: pytest.MonkeyPatch,
    kind: NodeKind,
    name: str,
    layer: int,
    head: int | None,
) -> None:
    clean, clean_logits, clean_values, corrupt_values, edges = gpt2_edge_endpoints
    position = clean.shape[1] - 1
    writer = Node("attn_head_out", position, layer=0, head=1)
    reader = Node(kind, position, layer=layer, head=head)
    excluded = (writer, reader)
    assert excluded in edges
    circuit = [edge for edge in edges if edge != excluded]
    assert len(circuit) == len(edges) - 1
    index = (0, position) if head is None else (0, position, head)
    delta = (
        corrupt_values["blocks.0.attn.hook_result"][0, position, 1]
        - clean_values["blocks.0.attn.hook_result"][0, position, 1]
    )
    assert delta.abs().max() > 1e-5
    expected_reader = clean_values[name].clone()
    expected_reader[index] = (
        clean_values[name][index]
        - clean_values["blocks.0.attn.hook_result"][0, position, 1]
        + corrupt_values["blocks.0.attn.hook_result"][0, position, 1]
    )
    with _edge_hook_flags(gpt2_bridge):
        expected, reference = _capture_gpt2_forward(
            gpt2_bridge, clean, [(name, _override_reader(index, expected_reader[index]))]
        )
        actual, observed = _capture_gpt2_ablation(
            gpt2_bridge, clean, edges, circuit, corrupt_values, monkeypatch
        )
    # Both routes apply identical fp32 slice arithmetic to the same clean reader.
    torch.testing.assert_close(reference[name], expected_reader, atol=0, rtol=0)
    torch.testing.assert_close(observed[name], expected_reader, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (actual - clean_logits).abs().max() > 1e-5


def test_coupled_partial_gpt2_ablation_matches_staged_reader_overrides(
    gpt2_bridge,
    gpt2_edge_endpoints,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean, clean_logits, clean_values, corrupt_values, edges = gpt2_edge_endpoints
    position = clean.shape[1] - 1
    upstream_name = "blocks.1.attn.hook_q_input"
    downstream_name = "blocks.2.attn.hook_k_input"
    upstream_index = (0, position, 3)
    downstream_index = (0, position, 4)
    upstream_value = (
        clean_values[upstream_name][upstream_index]
        - clean_values["blocks.0.attn.hook_result"][0, position, 1]
        + corrupt_values["blocks.0.attn.hook_result"][0, position, 1]
    )
    upstream_override = (upstream_name, _override_reader(upstream_index, upstream_value))
    with _edge_hook_flags(gpt2_bridge):
        _, staged = _capture_gpt2_forward(gpt2_bridge, clean, [upstream_override])
        live = staged["blocks.1.attn.hook_result"][0, position, 3]
        assert (live - clean_values["blocks.1.attn.hook_result"][0, position, 3]).abs().max() > 1e-6
        downstream_value = (
            staged[downstream_name][downstream_index]
            - live
            + corrupt_values["blocks.1.attn.hook_result"][0, position, 3]
        )
        overrides = [
            upstream_override,
            (downstream_name, _override_reader(downstream_index, downstream_value)),
        ]
        expected, reference = _capture_gpt2_forward(gpt2_bridge, clean, overrides)
        excluded = {
            (
                Node("attn_head_out", position, layer=0, head=1),
                Node("q_input", position, layer=1, head=3),
            ),
            (
                Node("attn_head_out", position, layer=1, head=3),
                Node("k_input", position, layer=2, head=4),
            ),
        }
        assert excluded <= set(edges)
        circuit = [edge for edge in edges if edge not in excluded]
        assert len(circuit) == len(edges) - 2
        actual, observed = _capture_gpt2_ablation(
            gpt2_bridge, clean, edges, circuit, corrupt_values, monkeypatch
        )
    for name in (upstream_name, downstream_name, "blocks.1.attn.hook_result"):
        torch.testing.assert_close(observed[name], reference[name], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (actual - clean_logits).abs().max() > 1e-5


def _bridge_hook_state(model: TransformerBridge) -> tuple:
    flags = tuple(
        getattr(model.cfg, name)
        for name in ("use_attn_result", "use_split_qkv_input", "use_hook_mlp_in", "use_attn_in")
    )
    handles = {
        name: (
            tuple(id(handle) for handle in point.fwd_hooks),
            tuple(id(handle) for handle in point.bwd_hooks),
        )
        for name, point in model.hook_dict.items()
    }
    return flags, handles, model.context_level


@pytest.mark.parametrize("fail_at", [None, 1, 3])
def test_forward_only_faithfulness_restores_gpt2_hooks_after_metric_error(
    gpt2_bridge,
    gpt2_edge_endpoints,
    fail_at: int | None,
) -> None:
    clean, _, _, _, edges = gpt2_edge_endpoints
    corrupt = gpt2_bridge.to_tokens(CORRUPT_PROMPT)
    answer_id = int(gpt2_bridge.to_tokens(" Paris")[0, -1])
    wrong_id = int(gpt2_bridge.to_tokens(" Moscow")[0, -1])
    metric = _logit_diff_metric(answer_id, wrong_id)
    calls = 0

    def checked_metric(logits: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        assert not logits.requires_grad
        if calls == fail_at:
            raise RuntimeError("metric failed")
        return metric(logits)

    original_state = _bridge_hook_state(gpt2_bridge)
    with _edge_hook_flags(gpt2_bridge):
        gpt2_bridge.set_use_split_qkv_input(False)
        gpt2_bridge.set_use_hook_mlp_in(False)
        gpt2_bridge.set_use_attn_in(True)
        try:
            with gpt2_bridge.hooks(fwd_hooks=[("hook_embed", lambda tensor, **kwargs: None)]):
                before = _bridge_hook_state(gpt2_bridge)
                with torch.no_grad():
                    if fail_at is None:
                        report = faithfulness(
                            gpt2_bridge, clean, corrupt, checked_metric, edges[:-1]
                        )
                        assert math.isfinite(report.recovered)
                    else:
                        with pytest.raises(RuntimeError, match="metric failed"):
                            faithfulness(gpt2_bridge, clean, corrupt, checked_metric, edges[:-1])
                assert _bridge_hook_state(gpt2_bridge) == before
        finally:
            gpt2_bridge.set_use_attn_in(False)
    assert _bridge_hook_state(gpt2_bridge) == original_state


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


def test_edge_class_breakout_marks_into_qk_as_the_least_faithful_class(gpt2_bridge) -> None:
    """Removing the into-Q/K edges is what breaks the circuit's faithfulness.

    Edges into Q and K pass through the softmax, so the linearized ranking is
    least trustworthy there; edges into V, the MLP, and the terminal readout are
    linear. The breakout measures that by keeping every edge outside one class,
    so a class whose removal collapses recovery is the one carrying the
    nonlinearity.

    Thresholds are set from measured values on this exact model and prompt pair
    (gpt2-small, fp32, CPU): keeping everything except into-Q/K recovers about
    1.09 of the gap, while removing into-V drops it to about -0.18 and removing
    into-logits to about 0.00.
    """
    clean = gpt2_bridge.to_tokens(CLEAN_PROMPT)
    corrupt = gpt2_bridge.to_tokens(CORRUPT_PROMPT)
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
    report = faithfulness(gpt2_bridge, clean, corrupt, metric_fn, result.top_edges(k=200))

    breakout = report.edge_class_recovered
    assert set(breakout) == {"into_qk", "into_v", "into_mlp", "into_logits"}
    assert all(math.isfinite(value) for value in breakout.values())
    assert sum(report.edge_class_counts.values()) == report.total_edges

    # Dropping the softmax-fed class leaves the circuit intact; dropping the
    # linear classes does not.
    assert breakout["into_qk"] > 0.9
    assert breakout["into_v"] < 0.5
    assert breakout["into_logits"] < 0.5
