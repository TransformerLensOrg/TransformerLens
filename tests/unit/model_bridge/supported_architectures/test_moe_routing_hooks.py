"""MoE routing observables on the bridge (hook_expert_weights / hook_expert_indices).

Mirrors HookedTransformer's MoE routing hooks: weights are exposed in HT's
``[tokens, num_experts]`` layout, indices as ``[tokens, top_k]``, on both MoE
families (5.13 ``TopKRouter`` blocks and GPT-OSS).
"""

from __future__ import annotations

import copy

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

NUM_EXPERTS = 4
TOP_K = 2
WEIGHTS = "blocks.0.mlp.gate.hook_expert_weights"
INDICES = "blocks.0.mlp.gate.hook_expert_indices"


def _tiny_bridge() -> TransformerBridge:
    torch.manual_seed(0)
    cfg = Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        max_position_embeddings=64,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    cfg._attn_implementation = "eager"
    model = Qwen2MoeForCausalLM(cfg).eval()
    bridge = build_bridge_from_module(
        model,
        "Qwen2MoeForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    bridge.adapter.setup_component_testing(model, bridge_model=bridge)
    return bridge


def _tokens() -> torch.Tensor:
    return torch.randint(0, 128, (1, 6))


def test_routing_hooks_are_cached_with_hooked_transformer_shapes():
    """Weights arrive at HT's [tokens, num_experts]; indices at [tokens, top_k]."""
    bridge, tokens = _tiny_bridge(), _tokens()
    _, cache = bridge.run_with_cache(tokens)

    assert cache[WEIGHTS].shape == (tokens.numel(), NUM_EXPERTS)
    assert cache[INDICES].shape == (tokens.numel(), TOP_K)


def test_routing_hooks_are_observe_only():
    """Firing the hooks does not perturb the forward pass."""
    bridge, tokens = _tiny_bridge(), _tokens()
    baseline = bridge(tokens)
    cached_logits, _ = bridge.run_with_cache(tokens)

    assert torch.equal(baseline, cached_logits)


def test_expanded_weights_carry_only_the_selected_experts():
    """The scatter puts each top-k score at its expert id and zeroes the rest."""
    bridge, tokens = _tiny_bridge(), _tokens()
    _, cache = bridge.run_with_cache(tokens)

    weights, indices = cache[WEIGHTS], cache[INDICES]
    assert (weights != 0).sum(-1).eq(TOP_K).all()
    gathered = weights.gather(-1, indices.long())
    assert torch.equal(
        gathered.sort(dim=-1).values, weights.topk(TOP_K, dim=-1).values.sort(dim=-1).values
    )


def test_editing_expert_weights_reaches_the_model():
    """An edit in the weights hook changes the output — not a dead hook."""
    bridge, tokens = _tiny_bridge(), _tokens()
    baseline = bridge(tokens)

    bridge.add_hook(WEIGHTS, lambda t, hook=None: torch.zeros_like(t))
    edited = bridge(tokens)
    bridge.reset_hooks()

    assert not torch.equal(baseline, edited)


def test_editing_expert_indices_reroutes():
    """An edit in the indices hook re-routes tokens to different experts."""
    bridge, tokens = _tiny_bridge(), _tokens()
    baseline = bridge(tokens)

    bridge.add_hook(INDICES, lambda t, hook=None: torch.zeros_like(t))
    edited = bridge(tokens)
    bridge.reset_hooks()

    assert not torch.equal(baseline, edited)


def test_hooked_transformer_names_alias_onto_the_router():
    """HT places these on the MoE block itself; migrated code must find them there."""
    bridge, tokens = _tiny_bridge(), _tokens()
    _, cache = bridge.run_with_cache(tokens)

    assert torch.equal(cache["blocks.0.mlp.hook_expert_weights"], cache[WEIGHTS])
    assert torch.equal(cache["blocks.0.mlp.hook_expert_indices"], cache[INDICES])


class _StubRouter(torch.nn.Module):
    """Router returning a fixed (logits, top-k weights, top-k indices) tuple."""

    def __init__(self, logits, weights, indices):
        super().__init__()
        self._out = (logits, weights, indices)

    def forward(self, hidden_states):  # noqa: ARG002 - fixed output by design
        return self._out


def test_rerouting_picks_up_the_weight_at_the_newly_selected_expert():
    """The gather runs after the indices hook, as HookedTransformer does.

    Re-routing a token to a different expert must pick up the weight sitting at
    that expert (zero when it was not originally selected), not carry the old
    expert's weight over to the new one.
    """
    from transformer_lens.model_bridge.generalized_components.moe import MoERouterBridge

    logits = torch.zeros(1, NUM_EXPERTS)
    weights = torch.tensor([[0.7, 0.3]])
    indices = torch.tensor([[1, 2]])

    router = MoERouterBridge(name="router")
    router.set_original_component(_StubRouter(logits, weights, indices))
    # Route both slots to expert 0, which held no weight in the original top-k.
    router.hook_expert_indices.add_hook(lambda t, hook=None: torch.zeros_like(t))

    _, out_weights, out_indices = router(torch.zeros(1, 8))

    assert torch.equal(out_indices, torch.zeros_like(indices))
    assert torch.equal(out_weights, torch.zeros_like(weights)), (
        "weights must be gathered at the edited indices, so a re-route to an "
        f"unselected expert yields 0; got {out_weights.tolist()}"
    )


def test_boosting_a_suppressed_expert_reroutes_the_selection():
    """A weight edit outside the top-k re-derives the selection from the edited
    tensor — HT's pre-top-k contract — instead of being discarded by the gather.
    """
    from transformer_lens.model_bridge.generalized_components.moe import MoERouterBridge

    logits = torch.zeros(1, NUM_EXPERTS)
    weights = torch.tensor([[0.7, 0.3]])
    indices = torch.tensor([[1, 2]])

    router = MoERouterBridge(name="router")
    router.set_original_component(_StubRouter(logits, weights, indices))

    def boost_expert_zero(t, hook=None):
        t = t.clone()
        t[:, 0] = 5.0  # expert 0 held no weight in the original top-k
        return t

    router.hook_expert_weights.add_hook(boost_expert_zero)
    _, out_weights, out_indices = router(torch.zeros(1, 8))

    assert out_indices[0].tolist() == [0, 1], out_indices
    assert torch.allclose(out_weights, torch.tensor([[5.0, 0.7]])), out_weights


def test_suppressed_expert_boost_reaches_the_model_output():
    """End-to-end: the edit changes logits, where the pre-fix gather no-opped it."""
    bridge = _tiny_bridge()
    tokens = torch.tensor([[3, 17, 42, 9]])

    with torch.no_grad():
        base = bridge(tokens, return_type="logits")

    def boost_all_nonselected(t, hook=None):
        t = t.clone()
        t[t == 0.0] = 100.0
        return t

    with torch.no_grad():
        edited = bridge.run_with_hooks(tokens, fwd_hooks=[(WEIGHTS, boost_all_nonselected)])

    assert not torch.allclose(edited, base), "off-top-k weight edit must reach the model"
