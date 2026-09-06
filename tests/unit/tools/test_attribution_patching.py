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
    attribution_patch,
    cache_activation_and_gradient,
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

    # The typed key rejects malformed nodes (Risk 1: explicit graph).
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


def test_config_unsupported_paths_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="PR2"):
        EdgeAttributionConfig(granularity="edge")
    with pytest.raises(NotImplementedError, match="PR3"):
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


def test_top_edges_not_implemented_until_pr2() -> None:
    result = AttributionResult(node_scores={})
    assert result.edge_scores == {}
    with pytest.raises(NotImplementedError, match="PR2"):
        result.top_edges()


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
        for linear in (self.w_z, self.w_o, self.w_mlp):
            nn.init.normal_(linear.weight, std=0.2)
        self.hook_z = HookPoint()
        self.hook_z.name = f"blocks.{layer}.attn.hook_z"
        self.hook_mlp_out = HookPoint()
        self.hook_mlp_out.name = f"blocks.{layer}.hook_mlp_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = residual.shape
        z = self.hook_z(self.w_z(residual).reshape(batch, seq, self.n_heads, self.d_head))
        residual = residual + self.w_o(z.reshape(batch, seq, self.n_heads * self.d_head))
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
        residual = self.hook_embed(self.embed(tokens))
        for block in self.blocks:
            residual = block(residual)
        if return_type is None:
            return None
        return self.unembed(self.ln_final(residual))


def _expected_node_count() -> int:
    return SEQ_LEN + N_LAYERS * (N_HEADS * SEQ_LEN + SEQ_LEN)


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
