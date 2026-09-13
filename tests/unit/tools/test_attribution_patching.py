"""Model-free unit tests for the attribution-patching substrate.

These tests target the ``TransformerBridge`` API exclusively (TransformerLens v4
deprecates ``HookedTransformer``). They use a tiny, deliberately *linear*
``TransformerBridge`` subclass so gradients have a closed form and, later, the
first-order attribution identity holds exactly.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest
import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.attribution_patching import (
    AttributionResult,
    EdgeAttributionConfig,
    GradientCache,
    Node,
    _assert_edges_unique,
    _check_required_hooks,
    _ensure_edge_hook_flags,
    _node_effects,
    _required_hook_names,
    _writer_hook_name,
    attribution_patch,
    cache_activation_and_gradient,
    enumerate_edges,
    enumerate_nodes,
)

D_MODEL = 4
D_VOCAB = 6
N_LAYERS = 2
SEQ_LEN = 3


class _LinearBlock(nn.Module):
    """Position-wise residual-linear block with a single output hook point."""

    def __init__(self, d_model: int, layer: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        nn.init.normal_(self.linear.weight, std=0.2)
        self.hook_out = HookPoint()
        self.hook_out.name = f"blocks.{layer}.hook_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hook_out(residual + self.linear(residual))


class _LinearToyBridge(TransformerBridge):
    """Tiny fully-linear ``TransformerBridge`` with Bridge-native hook points.

    The production constructor needs a Hugging Face model and an architecture
    adapter; unit tests only need the hook graph, a forward pass, and the
    ``hooks()`` context, so this subclass initializes ``nn.Module`` directly while
    keeping the concrete ``TransformerBridge`` isinstance contract. Every layer is
    linear and ``ln_final`` is the identity, so the residual->metric map is linear.
    """

    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        nn.Module.__init__(self)
        # TransformerBridge.__setattr__ registers HookPoints here; create it before
        # any HookPoint attribute is assigned.
        self._hook_registry: dict[str, HookPoint] = {}
        self.context_level = 0
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            d_vocab=D_VOCAB,
            d_vocab_out=D_VOCAB,
            model_name="linear-toy-bridge",
            dtype=dtype,
            device="cpu",
        )
        self.compatibility_mode = False
        self._weights_processed = False
        self.forward_calls = 0
        self.embed = nn.Embedding(D_VOCAB, D_MODEL, dtype=dtype)
        nn.init.normal_(self.embed.weight, std=0.2)
        self.hook_embed = HookPoint()
        self.hook_embed.name = "hook_embed"
        self.blocks = nn.ModuleList(
            [_LinearBlock(D_MODEL, layer, dtype) for layer in range(N_LAYERS)]
        )
        self.ln_final = nn.Identity()
        self.unembed = nn.Linear(D_MODEL, D_VOCAB, bias=False, dtype=dtype)
        nn.init.normal_(self.unembed.weight, std=0.2)
        self.eval()

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        hooks: dict[str, HookPoint] = {"hook_embed": self.hook_embed}
        for layer, block in enumerate(self.blocks):
            hooks[f"blocks.{layer}.hook_out"] = block.hook_out
        return hooks

    def check_hooks_to_add(self, name: str) -> None:  # pragma: no cover - trivial
        del name

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        # A production bridge delegates this to its wrapped HF model; this toy owns
        # its small modules directly, so enumerate the nn.Module tree.
        return nn.Module.parameters(self, recurse=recurse)

    def named_parameters(  # type: ignore[override]
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        return nn.Module.named_parameters(
            self, prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    def to_tokens(self, prompt: str) -> torch.Tensor:
        ids = [(3 * index + len(prompt)) % D_VOCAB for index in range(SEQ_LEN)]
        return torch.tensor([ids], dtype=torch.long)

    def forward(
        self, tokens: torch.Tensor, return_type: str | None = "logits"
    ) -> torch.Tensor | None:
        self.forward_calls += 1
        residual = self.hook_embed(self.embed(tokens))
        for block in self.blocks:
            residual = block(residual)
        if return_type is None:
            return None
        return self.unembed(self.ln_final(residual))

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[str, Any]] = [],
        bwd_hooks: list[tuple[str, Any]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ) -> Iterator["_LinearToyBridge"]:
        del clear_contexts
        added: list[tuple[HookPoint, str, Any]] = []
        for direction, specs in (("fwd", fwd_hooks), ("bwd", bwd_hooks)):
            for name, hook_fn in specs:
                hook_point = self.hook_dict[name]
                hook_point.add_hook(hook_fn, dir=direction)
                handles = hook_point.fwd_hooks if direction == "fwd" else hook_point.bwd_hooks
                added.append((hook_point, direction, handles[-1]))
        try:
            yield self
        finally:
            if reset_hooks_end:
                for hook_point, direction, handle in added:
                    handle.hook.remove()
                    handles = hook_point.fwd_hooks if direction == "fwd" else hook_point.bwd_hooks
                    if handle in handles:
                        handles.remove(handle)


def _metric_fn(answer: int, wrong: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logits[0, -1, answer] - logits[0, -1, wrong]

    return metric


def test_gradient_cache_only_covers_filtered_names() -> None:
    model = _LinearToyBridge()
    tokens = model.to_tokens("prompt")
    metric = _metric_fn(answer=1, wrong=2)

    result = cache_activation_and_gradient(
        model, tokens, metric, names_filter=["blocks.0.hook_out"]
    )

    assert set(result.activations) == {"blocks.0.hook_out"}
    assert set(result.gradients) == {"blocks.0.hook_out"}
    grad = result.gradients["blocks.0.hook_out"]
    assert grad is not None
    assert grad.shape == result.activations["blocks.0.hook_out"].shape
    assert torch.isfinite(grad).all()


def test_gradient_cache_matches_closed_form_linear_gradient() -> None:
    model = _LinearToyBridge()
    tokens = model.to_tokens("prompt")
    answer, wrong = 1, 2
    metric = _metric_fn(answer, wrong)

    result = cache_activation_and_gradient(
        model, tokens, metric, names_filter=["blocks.0.hook_out"]
    )
    grad = result.gradients["blocks.0.hook_out"]

    # Closed form: metric = (u_a - u_b) . (I + W1) h0[-1]; blocks are position-wise
    # so the gradient is zero except at the final position.
    direction = model.unembed.weight[answer] - model.unembed.weight[wrong]  # [d_model]
    jac = torch.eye(D_MODEL) + model.blocks[1].linear.weight  # d h1 / d h0
    expected_last = jac.T @ direction
    expected = torch.zeros(1, SEQ_LEN, D_MODEL)
    expected[0, -1] = expected_last

    torch.testing.assert_close(grad, expected)


def test_gradient_cache_activation_only_skips_gradients() -> None:
    model = _LinearToyBridge()
    tokens = model.to_tokens("prompt")
    metric = _metric_fn(answer=1, wrong=2)

    result = cache_activation_and_gradient(
        model, tokens, metric, names_filter=["blocks.0.hook_out"], compute_gradient=False
    )

    # Activation-only pass: activations populated, every gradient left None.
    assert set(result.activations) == {"blocks.0.hook_out"}
    assert torch.isfinite(result.activations["blocks.0.hook_out"]).all()
    assert set(result.gradients) == {"blocks.0.hook_out"}
    assert result.gradients["blocks.0.hook_out"] is None


# ---------------------------------------------------------------------------
# Commit 2 — typed computational-graph node model
# ---------------------------------------------------------------------------

N_HEADS = 2
D_HEAD = 2


def _synthetic_node_cache(
    n_layers: int = N_LAYERS,
    seq_len: int = SEQ_LEN,
    n_heads: int = N_HEADS,
    d_head: int = D_HEAD,
    d_model: int = D_MODEL,
) -> GradientCache:
    """A cache whose keys/shapes carry the standard node-granularity hook points.

    Node enumeration only reads hook names and tensor shapes, so the contents can
    be zeros; this keeps the graph test model-free and independent of any forward
    pass.
    """
    activations: dict[str, torch.Tensor] = {"hook_embed": torch.zeros(1, seq_len, d_model)}
    for layer in range(n_layers):
        activations[f"blocks.{layer}.attn.hook_z"] = torch.zeros(1, seq_len, n_heads, d_head)
        activations[f"blocks.{layer}.hook_mlp_out"] = torch.zeros(1, seq_len, d_model)
    return GradientCache(
        activations=activations,
        gradients={name: None for name in activations},
        metric=torch.tensor(0.0),
    )


def _cfg_stub(n_layers: int = N_LAYERS) -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(n_layers=n_layers))


def test_node_hook_name_and_key_validation() -> None:
    assert Node(kind="embed", position=0).hook_name == "hook_embed"
    assert (
        Node(kind="attn_head_out", layer=1, head=0, position=2).hook_name == "blocks.1.attn.hook_z"
    )
    assert Node(kind="mlp_out", layer=0, position=1).hook_name == "blocks.0.hook_mlp_out"

    # The typed key rejects malformed nodes rather than building a wrong graph.
    with pytest.raises(ValueError):
        Node(kind="embed", position=0, layer=0)
    with pytest.raises(ValueError):
        Node(kind="attn_head_out", layer=0, position=0)  # head missing
    with pytest.raises(ValueError):
        Node(kind="mlp_out", layer=0, head=0, position=0)  # head not allowed


def test_enumerate_nodes_returns_expected_keys() -> None:
    nodes = enumerate_nodes(_cfg_stub(), _synthetic_node_cache())

    # embed(pos) + per layer [head*pos attn-head-out + pos mlp-out]
    expected = SEQ_LEN + N_LAYERS * (N_HEADS * SEQ_LEN + SEQ_LEN)
    assert len(nodes) == expected
    assert len(set(nodes)) == expected  # nodes are unique + hashable

    embed_nodes = [n for n in nodes if n.kind == "embed"]
    assert {n.position for n in embed_nodes} == set(range(SEQ_LEN))
    assert all(n.layer is None and n.head is None for n in embed_nodes)

    attn_nodes = [n for n in nodes if n.kind == "attn_head_out"]
    assert {(n.layer, n.head) for n in attn_nodes} == {
        (layer, head) for layer in range(N_LAYERS) for head in range(N_HEADS)
    }

    mlp_nodes = [n for n in nodes if n.kind == "mlp_out"]
    assert {(n.layer, n.position) for n in mlp_nodes} == {
        (layer, pos) for layer in range(N_LAYERS) for pos in range(SEQ_LEN)
    }
    assert all(n.head is None for n in mlp_nodes)


def test_enumerate_nodes_raises_on_missing_hook() -> None:
    cache = _synthetic_node_cache()
    del cache.activations["blocks.1.hook_mlp_out"]

    with pytest.raises(ValueError, match="blocks.1.hook_mlp_out"):
        enumerate_nodes(_cfg_stub(), cache)


# ---------------------------------------------------------------------------
# Commit 3 — config + result API
# ---------------------------------------------------------------------------


def test_config_defaults_are_the_supported_node_sweep() -> None:
    config = EdgeAttributionConfig()
    assert config.granularity == "node"
    assert config.ig_steps == 1


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="ig_steps"):
        EdgeAttributionConfig(ig_steps=0)


def test_config_accepts_edge_granularity() -> None:
    config = EdgeAttributionConfig(granularity="edge")
    assert config.granularity == "edge"
    assert config.ig_steps == 1


def test_config_ig_steps_above_one_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="integrated gradient"):
        EdgeAttributionConfig(ig_steps=5)


def test_top_nodes_ranks_by_effect_magnitude() -> None:
    small = Node(kind="embed", position=0)
    big_negative = Node(kind="mlp_out", layer=0, position=1)
    medium = Node(kind="attn_head_out", layer=1, head=0, position=2)
    result = AttributionResult(
        node_scores={small: 0.1, big_negative: -5.0, medium: 2.0},
    )

    ranked = result.top_nodes(k=2)
    assert [node for node, _ in ranked] == [big_negative, medium]

    # k beyond the node count returns every node, still magnitude-ordered.
    assert [node for node, _ in result.top_nodes(k=10)] == [big_negative, medium, small]


def test_top_edges_ranks_by_effect_magnitude() -> None:
    small = (Node(kind="embed", position=0), Node(kind="mlp_in", layer=0, position=0))
    big_negative = (
        Node(kind="embed", position=0),
        Node(kind="q_input", layer=0, head=0, position=0),
    )
    medium = (
        Node(kind="attn_head_out", layer=0, head=0, position=0),
        Node(kind="mlp_in", layer=1, position=0),
    )
    result = AttributionResult(
        node_scores={},
        edge_scores={small: 0.1, big_negative: -5.0, medium: 2.0},
    )

    ranked = result.top_edges(k=2)
    assert [(writer, reader) for writer, reader, _ in ranked] == [big_negative, medium]

    # k beyond the edge count returns every edge, still magnitude-ordered.
    full = result.top_edges(k=10)
    assert [(writer, reader) for writer, reader, _ in full] == [big_negative, medium, small]
    assert [score for _, _, score in full] == [-5.0, 2.0, 0.1]


def test_top_edges_empty_when_no_edge_sweep_has_run() -> None:
    result = AttributionResult(node_scores={})
    assert result.edge_scores == {}
    assert result.top_edges() == []


# ---------------------------------------------------------------------------
# Commit 4 — node attribution_patch entry point
# ---------------------------------------------------------------------------


class _AttnMlpBlock(nn.Module):
    """A block exposing the standard node-granularity hook points.

    ``hook_z`` carries the per-head output ``[batch, seq, n_heads, d_head]`` and
    ``hook_mlp_out`` the MLP write ``[batch, seq, d_model]`` — the two per-layer
    hook families ``enumerate_nodes`` reads. The block is linear; the test only
    needs a real hook graph with gradients flowing to those points, not attention.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, layer: int, dtype: torch.dtype):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.w_z = nn.Linear(d_model, n_heads * d_head, bias=False, dtype=dtype)
        self.w_o = nn.Linear(n_heads * d_head, d_model, bias=False, dtype=dtype)
        self.w_mlp = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(p=0.5)
        for linear in (self.w_z, self.w_o, self.w_mlp):
            nn.init.normal_(linear.weight, std=0.2)
        self.hook_z = HookPoint()
        self.hook_z.name = f"blocks.{layer}.attn.hook_z"
        self.hook_mlp_out = HookPoint()
        self.hook_mlp_out.name = f"blocks.{layer}.hook_mlp_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = residual.shape
        z = self.hook_z(self.w_z(residual).reshape(batch, seq, self.n_heads, self.d_head))
        attn_out = self.w_o(z.reshape(batch, seq, self.n_heads * self.d_head))
        residual = residual + self.dropout(attn_out)
        mlp_out = self.hook_mlp_out(self.w_mlp(residual))
        return residual + mlp_out


class _NodeGraphToyBridge(_LinearToyBridge):
    """A tiny ``TransformerBridge`` carrying the full node-granularity hook graph.

    Inherits ``_LinearToyBridge``'s ``hooks()`` / parameter plumbing but swaps in
    blocks with ``attn.hook_z`` + ``hook_mlp_out`` so ``attribution_patch`` can
    enumerate and score every node.
    """

    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        nn.Module.__init__(self)
        self._hook_registry: dict[str, HookPoint] = {}
        self.context_level = 0
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            d_vocab=D_VOCAB,
            d_vocab_out=D_VOCAB,
            model_name="node-graph-toy-bridge",
            dtype=dtype,
            device="cpu",
        )
        self.compatibility_mode = False
        self._weights_processed = False
        self.forward_calls = 0
        self.embed = nn.Embedding(D_VOCAB, D_MODEL, dtype=dtype)
        nn.init.normal_(self.embed.weight, std=0.2)
        self.hook_embed = HookPoint()
        self.hook_embed.name = "hook_embed"
        self.blocks = nn.ModuleList(
            [_AttnMlpBlock(D_MODEL, N_HEADS, D_HEAD, layer, dtype) for layer in range(N_LAYERS)]
        )
        self.ln_final = nn.Identity()
        self.unembed = nn.Linear(D_MODEL, D_VOCAB, bias=False, dtype=dtype)
        nn.init.normal_(self.unembed.weight, std=0.2)
        self.eval()

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        hooks: dict[str, HookPoint] = {"hook_embed": self.hook_embed}
        for layer, block in enumerate(self.blocks):
            hooks[f"blocks.{layer}.attn.hook_z"] = block.hook_z
            hooks[f"blocks.{layer}.hook_mlp_out"] = block.hook_mlp_out
        return hooks

    def forward(
        self, tokens: torch.Tensor, return_type: str | None = "logits"
    ) -> torch.Tensor | None:
        self.forward_calls += 1
        residual = self.hook_embed(self.embed(tokens))
        for block in self.blocks:
            residual = block(residual)
        if return_type is None:
            return None
        return self.unembed(self.ln_final(residual))


def _expected_node_count() -> int:
    return SEQ_LEN + N_LAYERS * (N_HEADS * SEQ_LEN + SEQ_LEN)


def test_gradient_cache_defaults_to_the_node_hook_set() -> None:
    """``names_filter=None`` caches the node hook set, not every hook point.

    On a real Bridge, "every hook point" raises: ``hook_dict`` exposes gated points
    (``hook_mlp_in``, ``attn.hook_result``, split-QKV inputs) that ``add_hook``
    rejects unless the matching ``set_use_*`` flag is on. The default filter falls
    back to the node graph's hook set so it works out of the box.
    """
    model = _NodeGraphToyBridge()
    tokens = torch.tensor([[1, 2, 3]])
    metric = _metric_fn(answer=1, wrong=2)

    result = cache_activation_and_gradient(model, tokens, metric)

    expected = {
        "hook_embed",
        "blocks.0.attn.hook_z",
        "blocks.0.hook_mlp_out",
        "blocks.1.attn.hook_z",
        "blocks.1.hook_mlp_out",
    }
    assert set(result.activations) == expected
    assert set(result.gradients) == expected


def test_attribution_patch_scores_every_node_with_finite_values() -> None:
    model = _NodeGraphToyBridge()
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[3, 2, 1]])

    result = attribution_patch(model, clean, corrupt, _metric_fn(answer=1, wrong=2))

    assert isinstance(result, AttributionResult)
    assert len(result.node_scores) == _expected_node_count()
    assert all(isinstance(score, float) for score in result.node_scores.values())
    assert all(math.isfinite(score) for score in result.node_scores.values())
    assert result.edge_scores == {}
    # top_nodes ranks the scored graph by magnitude.
    assert len(result.top_nodes(k=3)) == 3


def test_attribution_patch_averages_scores_across_the_batch() -> None:
    model = _NodeGraphToyBridge()
    clean = torch.tensor([[1, 2, 3], [0, 4, 5]])
    corrupt = torch.tensor([[3, 2, 1], [5, 4, 0]])
    metric = _metric_fn(answer=1, wrong=2)

    batched = attribution_patch(model, clean, corrupt, metric)
    per_example = [
        attribution_patch(model, clean[i : i + 1], corrupt[i : i + 1], metric) for i in range(2)
    ]

    assert set(batched.node_scores) == set(per_example[0].node_scores)
    for node in batched.node_scores:
        expected = (per_example[0].node_scores[node] + per_example[1].node_scores[node]) / 2
        assert batched.node_scores[node] == pytest.approx(expected)


def test_attribution_patch_raises_on_token_length_mismatch() -> None:
    model = _NodeGraphToyBridge()
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[1, 2]])

    with pytest.raises(ValueError, match="same length"):
        attribution_patch(model, clean, corrupt, _metric_fn(answer=1, wrong=2))


def test_attribution_patch_raises_on_batch_size_mismatch() -> None:
    model = _NodeGraphToyBridge()
    clean = torch.tensor([[1, 2, 3], [3, 2, 1]])
    corrupt = torch.tensor([[3, 2, 1]])

    with pytest.raises(ValueError, match="same number of prompt pairs"):
        attribution_patch(model, clean, corrupt, _metric_fn(answer=1, wrong=2))


def test_attribution_patch_rejects_model_in_training_mode() -> None:
    model = _NodeGraphToyBridge()
    model.train()
    tokens = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match=r"model\.eval\(\)"):
        attribution_patch(model, tokens, tokens, _metric_fn(answer=1, wrong=2))
    assert model.training is True
    assert model.forward_calls == 0


def test_attribution_patch_rejects_nested_submodule_in_training_mode() -> None:
    model = _NodeGraphToyBridge()
    model.blocks[1].dropout.train()
    tokens = torch.tensor([[1, 2, 3]])
    assert model.training is False
    assert model.blocks[1].training is False

    with pytest.raises(ValueError, match=r"model\.eval\(\)"):
        attribution_patch(model, tokens, tokens, _metric_fn(answer=1, wrong=2))
    assert model.training is False
    assert model.blocks[1].training is False
    assert model.blocks[1].dropout.training is True
    assert model.forward_calls == 0


def test_attribution_patch_rejects_hidden_original_model_training_mode() -> None:
    model = _NodeGraphToyBridge()
    original_model = nn.Sequential(nn.Linear(D_MODEL, D_MODEL))
    original_model.eval()
    original_model[0].train()
    model.__dict__["original_model"] = original_model
    tokens = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="original_model"):
        attribution_patch(model, tokens, tokens, _metric_fn(answer=1, wrong=2))
    assert original_model.training is False
    assert original_model[0].training is True
    assert model.forward_calls == 0


def test_attribution_patch_identical_inputs_are_exactly_zero_in_eval_mode() -> None:
    model = _NodeGraphToyBridge()
    tokens = torch.tensor([[1, 2, 3]])

    result = attribution_patch(model, tokens, tokens, _metric_fn(answer=1, wrong=2))

    assert result.node_scores
    assert all(score == 0.0 for score in result.node_scores.values())


# ---------------------------------------------------------------------------
# Commit 5 — linear-model reconstruction identity
# ---------------------------------------------------------------------------
#
# ``_NodeGraphToyBridge`` is fully linear by construction — the remaining
# nonlinearities on the residual->metric path are neutralized without extra
# freezing: ``ln_final`` is ``nn.Identity``, the
# attention projection is a plain linear map with no softmax pattern, the MLP has
# no activation, and ``_metric_fn`` is linear in the logits. The first-order
# Taylor estimate each node score uses is therefore *exact*, so the reconstruction
# and single-node identities below hold under equality (tight ``atol``) rather than
# the ``atol``-slack approximation a stock nonlinear model would require.


def _metric_delta(
    model: _NodeGraphToyBridge,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    with torch.no_grad():
        return float(metric_fn(model(clean)) - metric_fn(model(corrupt)))


def _patch_node_toward_clean(
    model: _NodeGraphToyBridge,
    corrupt: torch.Tensor,
    node: Node,
    clean_activations: dict[str, torch.Tensor],
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    """Run the corrupt forward with ``node``'s activation replaced by its clean value."""
    clean_value = clean_activations[node.hook_name]

    def hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
        del hook
        if node.kind == "attn_head_out":
            tensor[0, node.position, node.head] = clean_value[0, node.position, node.head]
        else:
            tensor[0, node.position] = clean_value[0, node.position]
        return tensor

    with torch.no_grad(), model.hooks(fwd_hooks=[(node.hook_name, hook)]):
        return float(metric_fn(model(corrupt)))


def test_linear_reconstruction_identity_holds_on_a_complete_cut() -> None:
    """Node attribution reconstructs ``m(clean) - m(corrupt)`` exactly on a complete cut.

    The identity holds over a **complete cut** of the graph, not the full node set.
    Each residual write flows through the writes downstream of it (the MLP reads the
    post-attention residual, which already contains the embed and attention writes),
    so a node's full-model gradient re-counts the paths of every node upstream of
    it: summing embed + attention + MLP scores overcounts. The embed layer is the
    input-side complete cut — every path to the metric passes through exactly one
    embed position — so its scores alone reconstruct the exact metric delta for a
    linear model.
    """
    model = _NodeGraphToyBridge()
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[3, 2, 1]])
    metric = _metric_fn(answer=1, wrong=2)

    result = attribution_patch(model, clean, corrupt, metric)
    embed_reconstruction = sum(
        score for node, score in result.node_scores.items() if node.kind == "embed"
    )

    assert embed_reconstruction == pytest.approx(
        _metric_delta(model, clean, corrupt, metric), abs=1e-6
    )


def test_linear_single_node_patch_matches_score_and_sign() -> None:
    """Patching one node corrupt->clean moves the metric by exactly its score.

    For a linear model the first-order attribution of a single node equals the exact
    effect of patching only that node from its corrupt value to its clean value
    (everything upstream held at corrupt). This pins the sign/direction convention
    end to end across all three node families: a positive score corresponds to the
    metric moving in the positive direction under the denoising patch.
    """
    model = _NodeGraphToyBridge()
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[3, 2, 1]])
    metric = _metric_fn(answer=1, wrong=2)

    # names_filter=None defaults to the node hook set, which for this toy is every hook.
    clean_cache = cache_activation_and_gradient(model, clean, metric)
    result = attribution_patch(model, clean, corrupt, metric)
    with torch.no_grad():
        m_corrupt = float(metric(model(corrupt)))

    covered: set[str] = set()
    for node, score in result.top_nodes(k=len(result.node_scores)):
        delta_m = _patch_node_toward_clean(model, corrupt, node, clean_cache.activations, metric)
        delta_m -= m_corrupt
        assert delta_m == pytest.approx(score, abs=1e-6)
        if abs(score) > 1e-6:
            assert (delta_m > 0) == (score > 0)  # denoising sign convention
        covered.add(node.kind)

    assert covered == {"embed", "attn_head_out", "mlp_out"}


# ---------------------------------------------------------------------------
# The denoising convention on a nonlinear model
# ---------------------------------------------------------------------------
#
# Every toy bridge above is linear, so its Jacobian is input-independent and the
# clean and corrupt runs share a gradient. That hides the "gradient is taken from
# the corrupt run" half of the denoising convention: repointing ``_node_effects``
# at ``clean_cache.gradients`` would leave every linear test green.
# ``_NonlinearNodeGraphToyBridge`` adds a GELU MLP so the two runs' gradients
# diverge, which makes the corrupt-vs-clean choice observable.


class _NonlinearAttnMlpBlock(_AttnMlpBlock):
    """``_AttnMlpBlock`` with a GELU MLP so the residual->metric map is nonlinear.

    The extra ``w_mlp_in`` + GELU make ``d(metric)/d(activation)`` input-dependent,
    so the clean and corrupt runs no longer share a Jacobian. Without a
    nonlinearity the two caches hold identical gradients and which run the score
    reads from cannot be told apart.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, layer: int, dtype: torch.dtype):
        super().__init__(d_model, n_heads, d_head, layer, dtype)
        self.w_mlp_in = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        nn.init.normal_(self.w_mlp_in.weight, std=0.6)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = residual.shape
        z = self.hook_z(self.w_z(residual).reshape(batch, seq, self.n_heads, self.d_head))
        residual = residual + self.w_o(z.reshape(batch, seq, self.n_heads * self.d_head))
        hidden = torch.nn.functional.gelu(self.w_mlp_in(residual))
        mlp_out = self.hook_mlp_out(self.w_mlp(hidden))
        return residual + mlp_out


class _NonlinearNodeGraphToyBridge(_NodeGraphToyBridge):
    """``_NodeGraphToyBridge`` whose MLPs carry a GELU, so gradients are input-dependent.

    Reuses the parent's hook graph and ``hooks()`` plumbing but swaps the linear
    blocks for :class:`_NonlinearAttnMlpBlock`, giving the clean and corrupt runs
    genuinely different gradients.
    """

    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__(dtype=dtype)
        torch.manual_seed(1)
        self.blocks = nn.ModuleList(
            [
                _NonlinearAttnMlpBlock(D_MODEL, N_HEADS, D_HEAD, layer, dtype)
                for layer in range(N_LAYERS)
            ]
        )
        self.eval()


def test_nonlinear_node_scores_read_the_corrupt_run_gradient() -> None:
    """On a nonlinear model the score reads the corrupt run's gradient, not the clean one.

    With an input-dependent Jacobian the clean and corrupt caches hold *different*
    gradients, so pairing the clean-activation delta with the clean gradient (the
    reversed convention) yields different scores from the corrupt-gradient one the
    docstring promises. This pins ``attribution_patch`` to the corrupt gradient — a
    guard the linear tests structurally cannot provide.
    """
    model = _NonlinearNodeGraphToyBridge()
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[3, 2, 1]])
    metric = _metric_fn(answer=1, wrong=2)

    # Both caches carry gradients so the two conventions can be contrasted directly.
    clean_cache = cache_activation_and_gradient(model, clean, metric)
    corrupt_cache = cache_activation_and_gradient(model, corrupt, metric)
    nodes = enumerate_nodes(model, corrupt_cache)

    # Fixture guard: the nonlinearity must actually make the runs' gradients differ,
    # otherwise this test would silently pass on a still-linear model.
    assert any(
        not torch.allclose(clean_cache.gradients[name], corrupt_cache.gradients[name])
        for name in corrupt_cache.gradients
    )

    result = attribution_patch(model, clean, corrupt, metric)

    # attribution_patch scores the corrupt-gradient convention...
    corrupt_grad_scores = _node_effects(clean_cache, corrupt_cache, nodes)
    for node in nodes:
        assert result.node_scores[node] == pytest.approx(corrupt_grad_scores[node])

    # ...which genuinely diverges from the clean-gradient convention (same delta,
    # clean run's gradient). If _node_effects read the clean gradient instead, the
    # two would coincide and this assertion would fail.
    clean_grad_scores = _node_effects(
        clean_cache,
        GradientCache(
            activations=corrupt_cache.activations,
            gradients=clean_cache.gradients,
            metric=corrupt_cache.metric,
        ),
        nodes,
    )
    assert any(
        corrupt_grad_scores[node] != pytest.approx(clean_grad_scores[node]) for node in nodes
    )

    # Behavioural pin, independent of _node_effects: a single-node corrupt->clean
    # patch moves the metric in the score's direction. The estimate is first-order
    # on a nonlinear model, so assert the sign (not the exact value) across every
    # non-negligible node and confirm at least one was actually checked.
    with torch.no_grad():
        m_corrupt = float(metric(model(corrupt)))
    checked = 0
    for node, score in result.top_nodes(k=len(result.node_scores)):
        if abs(score) < 1e-3:
            continue
        delta_m = _patch_node_toward_clean(model, corrupt, node, clean_cache.activations, metric)
        delta_m -= m_corrupt
        assert (delta_m > 0) == (score > 0)  # denoising sign convention
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Edge-granularity hook requirements
# ---------------------------------------------------------------------------
#
# Edges into and out of an attention head read hook points a node sweep never
# needs: the per-head ``attn.hook_result`` (writer) and the split
# ``attn.hook_q_input``/``hook_k_input``/``hook_v_input`` (reader). Both
# families exist on the Bridge unconditionally but only fire once their owning
# config flag is on.


class _GatedEdgeHookBlock(nn.Module):
    """A block exposing the per-head hook points edge granularity reads.

    ``hook_result`` and the split ``hook_q_input``/``hook_k_input``/
    ``hook_v_input`` mirror the real Bridge's fire-time gating: the HookPoints
    exist unconditionally, but the block only calls them when the matching
    ``cfg`` flag is on, so their activation is absent from a cache built while
    the flag is off rather than present with a placeholder value.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        layer: int,
        dtype: torch.dtype,
        cfg: SimpleNamespace,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = n_heads
        self.d_head = d_head
        self.w_z = nn.Linear(d_model, n_heads * d_head, bias=False, dtype=dtype)
        self.w_o = nn.Linear(n_heads * d_head, d_model, bias=False, dtype=dtype)
        self.w_mlp = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        for linear in (self.w_z, self.w_o, self.w_mlp):
            nn.init.normal_(linear.weight, std=0.2)
        self.hook_z = HookPoint()
        self.hook_z.name = f"blocks.{layer}.attn.hook_z"
        self.hook_mlp_out = HookPoint()
        self.hook_mlp_out.name = f"blocks.{layer}.hook_mlp_out"
        self.hook_result = HookPoint()
        self.hook_result.name = f"blocks.{layer}.attn.hook_result"
        self.hook_q_input = HookPoint()
        self.hook_q_input.name = f"blocks.{layer}.attn.hook_q_input"
        self.hook_k_input = HookPoint()
        self.hook_k_input.name = f"blocks.{layer}.attn.hook_k_input"
        self.hook_v_input = HookPoint()
        self.hook_v_input.name = f"blocks.{layer}.attn.hook_v_input"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        batch, seq, d_model = residual.shape
        if self.cfg.use_split_qkv_input:
            per_head_residual = residual.unsqueeze(-2).expand(batch, seq, self.n_heads, d_model)
            self.hook_q_input(per_head_residual)
            self.hook_k_input(per_head_residual)
            self.hook_v_input(per_head_residual)

        z = self.hook_z(self.w_z(residual).reshape(batch, seq, self.n_heads, self.d_head))
        if self.cfg.use_attn_result:
            # Distributive over per-head weight slicing: summing this per-head
            # decomposition equals self.w_o(z_flat) exactly (no bias term to split).
            w_o_per_head = self.w_o.weight.reshape(d_model, self.n_heads, self.d_head).permute(
                1, 2, 0
            )
            per_head_out = self.hook_result(torch.einsum("bshd,hdm->bshm", z, w_o_per_head))
            attn_out = per_head_out.sum(dim=-2)
        else:
            attn_out = self.w_o(z.reshape(batch, seq, self.n_heads * self.d_head))

        residual = residual + attn_out
        mlp_out = self.hook_mlp_out(self.w_mlp(residual))
        return residual + mlp_out


class _EdgeHookToyBridge(_LinearToyBridge):
    """A tiny ``TransformerBridge`` whose blocks carry the edge-granularity hook points.

    ``cfg.use_attn_result``/``cfg.use_split_qkv_input`` default to ``False``,
    matching a real Bridge; ``set_use_attn_result``/``set_use_split_qkv_input``
    toggle them.
    """

    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        nn.Module.__init__(self)
        self._hook_registry: dict[str, HookPoint] = {}
        self.context_level = 0
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            d_vocab=D_VOCAB,
            d_vocab_out=D_VOCAB,
            model_name="edge-hook-toy-bridge",
            dtype=dtype,
            device="cpu",
            use_attn_result=False,
            use_split_qkv_input=False,
            use_hook_mlp_in=False,
        )
        self.compatibility_mode = False
        self._weights_processed = False
        self.embed = nn.Embedding(D_VOCAB, D_MODEL, dtype=dtype)
        nn.init.normal_(self.embed.weight, std=0.2)
        self.hook_embed = HookPoint()
        self.hook_embed.name = "hook_embed"
        self.blocks = nn.ModuleList(
            [
                _GatedEdgeHookBlock(D_MODEL, N_HEADS, D_HEAD, layer, dtype, self.cfg)
                for layer in range(N_LAYERS)
            ]
        )
        self.ln_final = nn.Identity()
        self.unembed = nn.Linear(D_MODEL, D_VOCAB, bias=False, dtype=dtype)
        nn.init.normal_(self.unembed.weight, std=0.2)

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        hooks: dict[str, HookPoint] = {"hook_embed": self.hook_embed}
        for layer, block in enumerate(self.blocks):
            hooks[f"blocks.{layer}.attn.hook_z"] = block.hook_z
            hooks[f"blocks.{layer}.hook_mlp_out"] = block.hook_mlp_out
            hooks[f"blocks.{layer}.attn.hook_result"] = block.hook_result
            hooks[f"blocks.{layer}.attn.hook_q_input"] = block.hook_q_input
            hooks[f"blocks.{layer}.attn.hook_k_input"] = block.hook_k_input
            hooks[f"blocks.{layer}.attn.hook_v_input"] = block.hook_v_input
        return hooks

    def forward(
        self, tokens: torch.Tensor, return_type: str | None = "logits"
    ) -> torch.Tensor | None:
        residual = self.hook_embed(self.embed(tokens))
        for block in self.blocks:
            residual = block(residual)
        if return_type is None:
            return None
        return self.unembed(self.ln_final(residual))

    def set_use_attn_result(self, use_attn_result: bool) -> None:
        self.cfg.use_attn_result = use_attn_result

    def set_use_split_qkv_input(self, use_split_qkv_input: bool) -> None:
        self.cfg.use_split_qkv_input = use_split_qkv_input

    def set_use_hook_mlp_in(self, use_hook_mlp_in: bool) -> None:
        self.cfg.use_hook_mlp_in = use_hook_mlp_in


def test_required_hook_names_edge_granularity_adds_per_head_families() -> None:
    node_names = set(_required_hook_names(N_LAYERS))
    edge_names = set(_required_hook_names(N_LAYERS, granularity="edge"))

    assert node_names < edge_names
    for layer in range(N_LAYERS):
        assert f"blocks.{layer}.attn.hook_result" in edge_names
        assert f"blocks.{layer}.attn.hook_q_input" in edge_names
        assert f"blocks.{layer}.attn.hook_k_input" in edge_names
        assert f"blocks.{layer}.attn.hook_v_input" in edge_names


def test_ensure_edge_hook_flags_enables_required_bridge_flags() -> None:
    model = _EdgeHookToyBridge()
    assert model.cfg.use_attn_result is False
    assert model.cfg.use_split_qkv_input is False
    assert model.cfg.use_hook_mlp_in is False

    _ensure_edge_hook_flags(model)

    assert model.cfg.use_attn_result is True
    assert model.cfg.use_split_qkv_input is True
    assert model.cfg.use_hook_mlp_in is True


def test_edge_hook_flags_populate_the_per_head_cache() -> None:
    model = _EdgeHookToyBridge()
    _ensure_edge_hook_flags(model)
    tokens = model.to_tokens("prompt")
    metric = _metric_fn(answer=1, wrong=2)
    edge_names = _required_hook_names(N_LAYERS, granularity="edge")

    cache = cache_activation_and_gradient(model, tokens, metric, names_filter=edge_names)

    assert set(cache.activations) == set(edge_names)
    for layer in range(N_LAYERS):
        assert cache.activations[f"blocks.{layer}.attn.hook_result"].shape == (
            1,
            SEQ_LEN,
            N_HEADS,
            D_MODEL,
        )
        for input_hook in ("hook_q_input", "hook_k_input", "hook_v_input"):
            assert cache.activations[f"blocks.{layer}.attn.{input_hook}"].shape == (
                1,
                SEQ_LEN,
                N_HEADS,
                D_MODEL,
            )

    # No missing hook points: the shared check is a no-op.
    _check_required_hooks(cache, edge_names, "edge graph", "hint unused when nothing is missing")


def test_missing_edge_hook_raises_instead_of_silently_shrinking_the_graph() -> None:
    model = _EdgeHookToyBridge()  # flags left off: per-head hooks never fire
    tokens = model.to_tokens("prompt")
    metric = _metric_fn(answer=1, wrong=2)
    edge_names = _required_hook_names(N_LAYERS, granularity="edge")

    cache = cache_activation_and_gradient(model, tokens, metric, names_filter=edge_names)

    assert "blocks.0.attn.hook_result" not in cache.activations

    with pytest.raises(ValueError, match="attn.hook_result"):
        _check_required_hooks(cache, edge_names, "edge graph", "enable use_attn_result.")


# ---------------------------------------------------------------------------
# Edge enumeration in the typed graph
# ---------------------------------------------------------------------------
#
# An edge u -> v carries writer u's residual-stream contribution into reader
# v's input. Readers are the per-head split-QKV inputs (q_input/k_input/
# v_input) and the MLP entry (mlp_in); writers are the existing node kinds
# (embed, attn_head_out, mlp_out). At a fixed sequence position, a reader
# connects to every writer that precedes it in the residual stream so far --
# the graph never mixes across positions, matching the per-position Node key.


def _synthetic_edge_cache(
    n_layers: int = N_LAYERS,
    seq_len: int = SEQ_LEN,
    n_heads: int = N_HEADS,
    d_head: int = D_HEAD,
    d_model: int = D_MODEL,
) -> GradientCache:
    """A cache whose keys/shapes carry the edge-granularity hook points.

    Edge enumeration only reads hook names and tensor shapes, so the contents
    can be zeros; this keeps the graph test model-free and independent of any
    forward pass.
    """
    cache = _synthetic_node_cache(
        n_layers=n_layers, seq_len=seq_len, n_heads=n_heads, d_head=d_head, d_model=d_model
    )
    for layer in range(n_layers):
        cache.activations[f"blocks.{layer}.attn.hook_result"] = torch.zeros(
            1, seq_len, n_heads, d_model
        )
        for input_hook in ("hook_q_input", "hook_k_input", "hook_v_input"):
            cache.activations[f"blocks.{layer}.attn.{input_hook}"] = torch.zeros(
                1, seq_len, n_heads, d_model
            )
        cache.activations[f"blocks.{layer}.hook_mlp_in"] = torch.zeros(1, seq_len, d_model)
    cache.gradients = {name: None for name in cache.activations}
    return cache


def test_reader_node_hook_name_and_key_validation() -> None:
    assert (
        Node(kind="q_input", layer=0, head=1, position=2).hook_name == "blocks.0.attn.hook_q_input"
    )
    assert (
        Node(kind="k_input", layer=1, head=0, position=0).hook_name == "blocks.1.attn.hook_k_input"
    )
    assert (
        Node(kind="v_input", layer=0, head=0, position=0).hook_name == "blocks.0.attn.hook_v_input"
    )
    assert Node(kind="mlp_in", layer=1, position=0).hook_name == "blocks.1.hook_mlp_in"

    # The typed key rejects malformed reader nodes rather than building a wrong graph.
    with pytest.raises(ValueError):
        Node(kind="q_input", layer=0, position=0)  # head missing
    with pytest.raises(ValueError):
        Node(kind="mlp_in", layer=0, head=0, position=0)  # head not allowed


def _expected_edge_count(n_layers: int, n_heads: int, seq_len: int) -> int:
    """Independent count of writer -> reader pairs for the dense per-position graph.

    At each position, a reader connects to every writer enumerated so far: the
    three per-head QKV readers at a layer see everything upstream of that
    layer's attention, and the MLP reader additionally sees that layer's own
    attention writes.
    """
    total_per_position = 0
    available = 1  # embed
    for _ in range(n_layers):
        total_per_position += 3 * n_heads * available  # q/k/v readers
        available += n_heads  # this layer's attn_head_out writers
        total_per_position += available  # mlp_in reader
        available += 1  # this layer's mlp_out writer
    return seq_len * total_per_position


def test_enumerate_edges_returns_expected_writer_reader_pairs() -> None:
    cache = _synthetic_edge_cache()
    edges = enumerate_edges(_cfg_stub(), cache)

    assert len(edges) == _expected_edge_count(N_LAYERS, N_HEADS, SEQ_LEN)
    assert len(set(edges)) == len(edges)  # no duplicate (u, v) pairs
    assert all(writer.position == reader.position for writer, reader in edges)

    # embed is the first writer at its position, so every reader at that
    # position has a direct edge from it.
    readers_at_0 = {reader for writer, reader in edges if writer.position == 0}
    for reader in readers_at_0:
        assert (Node(kind="embed", position=0), reader) in edges

    # the final layer's MLP output has no downstream reader in this graph.
    terminal_writer = Node(kind="mlp_out", layer=N_LAYERS - 1, position=0)
    assert all(writer != terminal_writer for writer, _ in edges)


def test_enumerate_edges_raises_on_missing_reader_hook() -> None:
    cache = _synthetic_edge_cache()
    del cache.activations["blocks.0.hook_mlp_in"]

    with pytest.raises(ValueError, match="blocks.0.hook_mlp_in"):
        enumerate_edges(_cfg_stub(), cache)


def test_assert_edges_unique_raises_on_duplicate() -> None:
    reader = Node(kind="mlp_in", layer=0, position=0)
    writer = Node(kind="embed", position=0)

    with pytest.raises(ValueError, match="more than once"):
        _assert_edges_unique([(writer, reader), (writer, reader)])


# ---------------------------------------------------------------------------
# Commit 3 - edge scoring in attribution_patch
# ---------------------------------------------------------------------------
#
# Edge scoring needs every reader hook to actually participate in the forward
# computation, unlike _GatedEdgeHookBlock's probes above, which fire but are
# discarded: a reader with no forward-graph path to the metric would receive
# no gradient, leaving nothing genuine for _edge_effects to score.
# _EdgeScoringBlock wires every per-head QKV input and the MLP entry into the
# actual computation, so a cache built from it carries real, distinguishable
# per-edge gradients.


class _EdgeScoringBlock(nn.Module):
    """A block whose split-QKV inputs and MLP entry are read, not just probed.

    Each head's z is a linear combination of that head's own hook_q_input /
    hook_k_input / hook_v_input, through independent per-head weights, so an
    edge into one input family is distinguishable from an edge into another.
    Whether a cfg flag is on only changes whether the corresponding HookPoint
    fires (and so whether it is visible in a cache); the per-head
    decomposition of the residual is distributive, so the computed value is
    identical either way, mirroring _GatedEdgeHookBlock's hook_result split.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        layer: int,
        dtype: torch.dtype,
        cfg: SimpleNamespace,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = n_heads
        self.d_head = d_head
        self.w_q = nn.Linear(d_model, n_heads * d_head, bias=False, dtype=dtype)
        self.w_k = nn.Linear(d_model, n_heads * d_head, bias=False, dtype=dtype)
        self.w_v = nn.Linear(d_model, n_heads * d_head, bias=False, dtype=dtype)
        self.w_o = nn.Linear(n_heads * d_head, d_model, bias=False, dtype=dtype)
        self.w_mlp = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        for linear in (self.w_q, self.w_k, self.w_v, self.w_o, self.w_mlp):
            nn.init.normal_(linear.weight, std=0.2)
        self.hook_z = HookPoint()
        self.hook_z.name = f"blocks.{layer}.attn.hook_z"
        self.hook_mlp_out = HookPoint()
        self.hook_mlp_out.name = f"blocks.{layer}.hook_mlp_out"
        self.hook_result = HookPoint()
        self.hook_result.name = f"blocks.{layer}.attn.hook_result"
        self.hook_q_input = HookPoint()
        self.hook_q_input.name = f"blocks.{layer}.attn.hook_q_input"
        self.hook_k_input = HookPoint()
        self.hook_k_input.name = f"blocks.{layer}.attn.hook_k_input"
        self.hook_v_input = HookPoint()
        self.hook_v_input.name = f"blocks.{layer}.attn.hook_v_input"
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_in.name = f"blocks.{layer}.hook_mlp_in"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        batch, seq, d_model = residual.shape
        per_head_residual = residual.unsqueeze(-2).expand(batch, seq, self.n_heads, d_model)
        if self.cfg.use_split_qkv_input:
            q_input = self.hook_q_input(per_head_residual)
            k_input = self.hook_k_input(per_head_residual)
            v_input = self.hook_v_input(per_head_residual)
        else:
            q_input = k_input = v_input = per_head_residual

        w_q_per_head = self.w_q.weight.reshape(self.n_heads, self.d_head, d_model)
        w_k_per_head = self.w_k.weight.reshape(self.n_heads, self.d_head, d_model)
        w_v_per_head = self.w_v.weight.reshape(self.n_heads, self.d_head, d_model)
        z = self.hook_z(
            torch.einsum("bshm,hdm->bshd", q_input, w_q_per_head)
            + torch.einsum("bshm,hdm->bshd", k_input, w_k_per_head)
            + torch.einsum("bshm,hdm->bshd", v_input, w_v_per_head)
        )

        w_o_per_head = self.w_o.weight.reshape(d_model, self.n_heads, self.d_head).permute(1, 2, 0)
        per_head_out_raw = torch.einsum("bshd,hdm->bshm", z, w_o_per_head)
        if self.cfg.use_attn_result:
            per_head_out = self.hook_result(per_head_out_raw)
        else:
            per_head_out = per_head_out_raw
        residual = residual + per_head_out.sum(dim=-2)

        mlp_in = self.hook_mlp_in(residual) if self.cfg.use_hook_mlp_in else residual
        mlp_out = self.hook_mlp_out(self.w_mlp(mlp_in))
        return residual + mlp_out


class _EdgeScoringToyBridge(_LinearToyBridge):
    """A tiny ``TransformerBridge`` whose edge-granularity hooks all sit on the data path.

    Every reader hook point (``hook_q_input``/``hook_k_input``/``hook_v_input``,
    ``hook_mlp_in``) and writer hook point (``hook_result``) that
    :func:`enumerate_edges` connects is read by the forward computation itself,
    so ``cache_activation_and_gradient``'s backward hooks capture real,
    per-edge gradients.
    """

    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        nn.Module.__init__(self)
        self._hook_registry: dict[str, HookPoint] = {}
        self.context_level = 0
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            d_vocab=D_VOCAB,
            d_vocab_out=D_VOCAB,
            model_name="edge-scoring-toy-bridge",
            dtype=dtype,
            device="cpu",
            use_attn_result=False,
            use_split_qkv_input=False,
            use_hook_mlp_in=False,
        )
        self.compatibility_mode = False
        self._weights_processed = False
        self.embed = nn.Embedding(D_VOCAB, D_MODEL, dtype=dtype)
        nn.init.normal_(self.embed.weight, std=0.2)
        self.hook_embed = HookPoint()
        self.hook_embed.name = "hook_embed"
        self.blocks = nn.ModuleList(
            [
                _EdgeScoringBlock(D_MODEL, N_HEADS, D_HEAD, layer, dtype, self.cfg)
                for layer in range(N_LAYERS)
            ]
        )
        self.ln_final = nn.Identity()
        self.unembed = nn.Linear(D_MODEL, D_VOCAB, bias=False, dtype=dtype)
        nn.init.normal_(self.unembed.weight, std=0.2)

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        hooks: dict[str, HookPoint] = {"hook_embed": self.hook_embed}
        for layer, block in enumerate(self.blocks):
            hooks[f"blocks.{layer}.attn.hook_z"] = block.hook_z
            hooks[f"blocks.{layer}.hook_mlp_out"] = block.hook_mlp_out
            hooks[f"blocks.{layer}.attn.hook_result"] = block.hook_result
            hooks[f"blocks.{layer}.attn.hook_q_input"] = block.hook_q_input
            hooks[f"blocks.{layer}.attn.hook_k_input"] = block.hook_k_input
            hooks[f"blocks.{layer}.attn.hook_v_input"] = block.hook_v_input
            hooks[f"blocks.{layer}.hook_mlp_in"] = block.hook_mlp_in
        return hooks

    def forward(
        self, tokens: torch.Tensor, return_type: str | None = "logits"
    ) -> torch.Tensor | None:
        residual = self.hook_embed(self.embed(tokens))
        for block in self.blocks:
            residual = block(residual)
        if return_type is None:
            return None
        return self.unembed(self.ln_final(residual))

    def set_use_attn_result(self, use_attn_result: bool) -> None:
        self.cfg.use_attn_result = use_attn_result

    def set_use_split_qkv_input(self, use_split_qkv_input: bool) -> None:
        self.cfg.use_split_qkv_input = use_split_qkv_input

    def set_use_hook_mlp_in(self, use_hook_mlp_in: bool) -> None:
        self.cfg.use_hook_mlp_in = use_hook_mlp_in


def test_attribution_patch_edge_granularity_scores_every_edge_with_finite_values() -> None:
    model = _EdgeScoringToyBridge()
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[3, 2, 1]])
    metric = _metric_fn(answer=1, wrong=2)

    result = attribution_patch(
        model, clean, corrupt, metric, config=EdgeAttributionConfig(granularity="edge")
    )

    assert isinstance(result, AttributionResult)
    assert len(result.edge_scores) == _expected_edge_count(N_LAYERS, N_HEADS, SEQ_LEN)
    assert all(isinstance(score, float) for score in result.edge_scores.values())
    assert all(math.isfinite(score) for score in result.edge_scores.values())

    top = result.top_edges(k=5)
    assert len(top) == 5
    magnitudes = [abs(score) for _, _, score in top]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for writer, reader, score in top:
        assert (writer, reader) in result.edge_scores
        assert result.edge_scores[(writer, reader)] == score


def test_attribution_patch_edge_aggregation_reconstructs_direct_node_scores() -> None:
    """Edge aggregation plus the residual stream's escape term equals the direct node score.

    A writer reaches the metric two ways: through every edge enumerate_edges gives
    it (what the aggregation sums), and by surviving unread in the residual stream
    all the way to the final MLP write -- the direct skip-connection carry-through,
    which none of enumerate_edges' four reader kinds model (there is no "final
    readout" reader in this graph). That escape term is the same shared vector for
    every writer at a given position: the corrupt-run gradient at the last layer's
    hook_mlp_out equals d(metric)/d(the final residual) exactly, since hook_mlp_out
    feeds the final residual by pure addition and nothing else reads it. Dotting a
    writer's own delta (measured at its _writer_hook_name, matching _edge_effects)
    against that shared vector and adding it to the writer's edge-score sum
    reconstructs its direct node score exactly, including for the final layer's MLP
    output itself, which has no outgoing edge and so is reconstructed by the escape
    term alone.
    """
    model = _EdgeScoringToyBridge()
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[3, 2, 1]])
    metric = _metric_fn(answer=1, wrong=2)

    edge_result = attribution_patch(
        model, clean, corrupt, metric, config=EdgeAttributionConfig(granularity="edge")
    )
    node_result = attribution_patch(
        model, clean, corrupt, metric, config=EdgeAttributionConfig(granularity="node")
    )

    edge_hook_names = _required_hook_names(N_LAYERS, granularity="edge")
    clean_cache = cache_activation_and_gradient(
        model, clean, metric, names_filter=edge_hook_names, compute_gradient=False
    )
    corrupt_cache = cache_activation_and_gradient(
        model, corrupt, metric, names_filter=edge_hook_names
    )
    escape_grad = corrupt_cache.gradients[f"blocks.{N_LAYERS - 1}.hook_mlp_out"]

    terminal_writer = Node(kind="mlp_out", layer=N_LAYERS - 1, position=0)
    assert terminal_writer not in edge_result.node_scores

    for node, direct_score in node_result.node_scores.items():
        name = _writer_hook_name(node)
        delta = clean_cache.activations[name] - corrupt_cache.activations[name]
        if node.kind == "attn_head_out":
            delta_vec = delta[0, node.position, node.head]
        else:
            delta_vec = delta[0, node.position]
        escape_term = float((delta_vec * escape_grad[0, node.position]).sum())

        edge_sum = edge_result.node_scores.get(node, 0.0)
        assert direct_score == pytest.approx(edge_sum + escape_term, abs=1e-5)
