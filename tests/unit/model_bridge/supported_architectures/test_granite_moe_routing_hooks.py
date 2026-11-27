"""GraniteMoe router hooks land on the right tuple elements on every transformers.

transformers 5.22 changed the GraniteMoe-family router return from
``(top_k_index, top_k_weights, router_logits)`` to
``(router_logits, top_k_weights, top_k_index)``. The router bridge must hook the
logits at ``hook_out`` and the top-k indices at ``hook_expert_indices`` either way.
"""

from __future__ import annotations

import copy

import torch
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

NUM_EXPERTS = 4
TOP_K = 2
LOGITS = "blocks.0.mlp.gate.hook_out"
WEIGHTS = "blocks.0.mlp.gate.hook_expert_weights"
INDICES = "blocks.0.mlp.gate.hook_expert_indices"


def _tiny_bridge() -> TransformerBridge:
    torch.manual_seed(0)
    cfg = GraniteMoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    model = GraniteMoeForCausalLM(cfg).eval()
    return build_bridge_from_module(
        model,
        "GraniteMoeForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()


def _tokens() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 128, (1, 6))


def test_router_hooks_get_logits_weights_and_indices():
    bridge, tokens = _tiny_bridge(), _tokens()
    _, cache = bridge.run_with_cache(tokens)

    logits, weights, indices = cache[LOGITS], cache[WEIGHTS], cache[INDICES]
    assert logits.is_floating_point() and logits.shape == (tokens.numel(), NUM_EXPERTS)
    assert not indices.is_floating_point() and indices.shape == (tokens.numel(), TOP_K)
    assert weights.shape == (tokens.numel(), NUM_EXPERTS)
    # The router keeps the top-k logits, so the hooked indices are their positions.
    assert torch.equal(
        indices.long().sort(dim=-1).values,
        logits.topk(TOP_K, dim=-1).indices.sort(dim=-1).values,
    )
    assert (weights != 0).sum(-1).eq(TOP_K).all()


def test_editing_expert_indices_reroutes():
    bridge, tokens = _tiny_bridge(), _tokens()
    baseline = bridge(tokens)

    bridge.add_hook(INDICES, lambda t, hook=None: torch.zeros_like(t))
    edited = bridge(tokens)
    bridge.reset_hooks()

    assert not torch.equal(baseline, edited)


def test_routing_hooks_are_observe_only():
    bridge, tokens = _tiny_bridge(), _tokens()
    baseline = bridge(tokens)
    cached_logits, _ = bridge.run_with_cache(tokens)

    assert torch.equal(baseline, cached_logits)
