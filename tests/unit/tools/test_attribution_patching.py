"""Model-free unit tests for the attribution-patching substrate.

These tests target the ``TransformerBridge`` API exclusively (TransformerLens v4
deprecates ``HookedTransformer``). They use a tiny, deliberately *linear*
``TransformerBridge`` subclass so gradients have a closed form and, later, the
first-order attribution identity holds exactly.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.attribution_patching import (
    cache_activation_and_gradient,
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
