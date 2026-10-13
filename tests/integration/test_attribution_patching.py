"""Attribution patching and faithfulness on a raw GPT-2 Bridge.

Covers converted gradient caches, gated reader forks, independent partial
intervention references, and circuit-relative class removal.
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
    EdgeClass,
    FaithfulnessConfig,
    GradientCache,
    Node,
    NodeKind,
    _ablate_edges,
    _edge_hook_flags,
    _edge_hook_names,
    _mean_writer_contributions,
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


def test_partial_mean_gpt2_ablation_matches_held_out_reader_override(
    gpt2_bridge,
    gpt2_edge_endpoints,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean, clean_logits, clean_values, _, edges = gpt2_edge_endpoints
    corrupt = gpt2_bridge.to_tokens(CORRUPT_PROMPT)
    calibration = gpt2_bridge.to_tokens(["The capital of Spain is", "The capital of Italy is"])
    assert calibration.shape == (2, clean.shape[1])
    position = clean.shape[1] - 1
    name = "blocks.1.attn.hook_q_input"
    index = (0, position, 3)
    excluded = (
        Node("attn_head_out", position, layer=0, head=1),
        Node("q_input", position, layer=1, head=3),
    )
    assert excluded in edges
    circuit = [edge for edge in edges if edge != excluded]
    with _edge_hook_flags(gpt2_bridge):
        _, first = _capture_gpt2_forward(gpt2_bridge, calibration[:1])
        _, second = _capture_gpt2_forward(gpt2_bridge, calibration[1:])
        names = ["hook_embed"]
        for layer in range(gpt2_bridge.cfg.n_layers):
            names.extend([f"blocks.{layer}.attn.hook_result", f"blocks.{layer}.hook_mlp_out"])
        reference_means = {hook: (first[hook] + second[hook]) / 2 for hook in names}
        actual_means = _mean_writer_contributions(gpt2_bridge, calibration)
        assert actual_means.keys() == reference_means.keys()
        for hook in names:
            torch.testing.assert_close(actual_means[hook], reference_means[hook], atol=0, rtol=0)
        expected_reader = clean_values[name].clone()
        expected_reader[index] = (
            clean_values[name][index]
            - clean_values["blocks.0.attn.hook_result"][0, position, 1]
            + reference_means["blocks.0.attn.hook_result"][0, position, 1]
        )
        assert (expected_reader - clean_values[name]).abs().max() > 1e-5
        expected, reference = _capture_gpt2_forward(
            gpt2_bridge, clean, [(name, _override_reader(index, expected_reader[index]))]
        )
        actual, observed = _capture_gpt2_ablation(
            gpt2_bridge, clean, edges, circuit, actual_means, monkeypatch
        )
        corrupt_logits, _ = _capture_gpt2_forward(gpt2_bridge, corrupt)
        answer_id = int(gpt2_bridge.to_tokens(" Paris")[0, -1])
        wrong_id = int(gpt2_bridge.to_tokens(" Moscow")[0, -1])
        metric = _logit_diff_metric(answer_id, wrong_id)
        report = faithfulness(
            gpt2_bridge,
            clean,
            corrupt,
            metric,
            circuit,
            FaithfulnessConfig(ablation="mean"),
            mean_tokens=calibration,
        )
    torch.testing.assert_close(reference[name], expected_reader, atol=0, rtol=0)
    torch.testing.assert_close(observed[name], expected_reader, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (actual - clean_logits).abs().max() > 1e-5
    recovered = float(
        (metric(expected) - metric(corrupt_logits))
        / (metric(clean_logits) - metric(corrupt_logits))
    )
    assert report.recovered == pytest.approx(recovered, abs=1e-6)


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

    # All three node families are present -- attn_head_out is the family the hook
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


def _gpt2_class_reader_names(edge_class: EdgeClass, n_layers: int) -> list[str]:
    suffixes = {
        "into_qk": ("attn.hook_q_input", "attn.hook_k_input"),
        "into_v": ("attn.hook_v_input",),
        "into_mlp": ("hook_mlp_in",),
    }
    if edge_class == "into_logits":
        return [f"blocks.{n_layers - 1}.hook_resid_post"]
    return [
        f"blocks.{layer}.{suffix}" for layer in range(n_layers) for suffix in suffixes[edge_class]
    ]


def _gpt2_class_overrides(
    clean: torch.Tensor,
    clean_values: dict[str, torch.Tensor],
    corrupt_values: dict[str, torch.Tensor],
    edge_class: EdgeClass,
    n_layers: int,
) -> tuple[tuple[Node, Node], list[tuple[str, Callable]]]:
    position = clean.shape[1] - 2
    kind: NodeKind = "q_input" if edge_class == "into_v" else "v_input"
    name = "blocks.0.attn.hook_q_input" if kind == "q_input" else "blocks.0.attn.hook_v_input"
    index = (0, position, 0)
    base_value = (
        clean_values[name][index]
        - clean_values["hook_embed"][0, position]
        + corrupt_values["hook_embed"][0, position]
    )
    base_edge = (Node("embed", position), Node(kind, position, layer=0, head=0))
    overrides = [(name, _override_reader(index, base_value))]
    # Replacing every writer in a class preserves the prompt-independent residual background.
    overrides.extend(
        (reader_name, _override_reader((), corrupt_values[reader_name]))
        for reader_name in _gpt2_class_reader_names(edge_class, n_layers)
    )
    return base_edge, overrides


@pytest.mark.parametrize("edge_class", ["into_qk", "into_v", "into_mlp", "into_logits"])
def test_edge_class_gpt2_removal_keeps_candidate_exclusions(
    gpt2_bridge,
    gpt2_edge_endpoints,
    monkeypatch: pytest.MonkeyPatch,
    edge_class: EdgeClass,
) -> None:
    clean, _, clean_values, corrupt_values, edges = gpt2_edge_endpoints
    corrupt = gpt2_bridge.to_tokens(CORRUPT_PROMPT)
    base_edge, overrides = _gpt2_class_overrides(
        clean, clean_values, corrupt_values, edge_class, gpt2_bridge.cfg.n_layers
    )
    class_names = _gpt2_class_reader_names(edge_class, gpt2_bridge.cfg.n_layers)
    assert base_edge in edges
    assert base_edge[1].hook_name not in class_names
    circuit = [edge for edge in edges if edge != base_edge]
    reduced = [edge for edge in circuit if edge[1].hook_name not in class_names]
    answer_id = int(gpt2_bridge.to_tokens(" Paris")[0, -1])
    wrong_id = int(gpt2_bridge.to_tokens(" Moscow")[0, -1])
    metric = _logit_diff_metric(answer_id, wrong_id)
    with _edge_hook_flags(gpt2_bridge):
        expected, reference = _capture_gpt2_forward(gpt2_bridge, clean, overrides)
        restored, _ = _capture_gpt2_forward(gpt2_bridge, clean, overrides[1:])
        actual, observed = _capture_gpt2_ablation(
            gpt2_bridge, clean, edges, reduced, corrupt_values, monkeypatch
        )
        report = faithfulness(gpt2_bridge, clean, corrupt, metric, circuit)
    # Dense corrections and cached reader overrides use different reduction orders.
    for name in class_names + [base_edge[1].hook_name]:
        torch.testing.assert_close(observed[name], reference[name], atol=2e-4, rtol=0)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)
    expected_recovery = (float(metric(expected)) - report.corrupt_metric) / (
        report.full_metric - report.corrupt_metric
    )
    assert report.edge_class_recovered[edge_class] == pytest.approx(expected_recovery, abs=1e-5)
    assert report.circuit_size == len(edges) - 1
    assert sum(report.edge_class_counts.values()) == len(edges)
    if edge_class != "into_logits":
        gap = report.full_metric - report.corrupt_metric
        assert abs(float(metric(expected) - metric(restored)) / gap) > 1e-4
