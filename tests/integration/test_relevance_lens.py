"""End-to-end relevance-rule backend on tiny, offline HF fixtures.

Per-component unit tests wrap one bridge component directly in a hand-built
single-block harness, which only ever registers a canonical mount name
(``ln1``, ``mlp``) once. That harness cannot see what happens on a real,
fully assembled ``TransformerBridge``, where the same component is *also*
reachable through the raw HF module tree under its own HF attribute name
(for example ``blocks.0._original_component.input_layernorm``). These tests
build a tiny random Qwen2 (the opaque gated-MLP recompute path) and a tiny
random Phi-3 (``JointGateUpMLPBridge``'s already-reconstructed forward)
fully offline -- random weights from a programmatic HF config, no network
access, no checkpoint download -- and exercise all three rules together on
the real block stack: forward stays bit-identical, canonical mounts are
actually found and installed, and the gradient each rule-active node passes
upstream matches its closed-form VJP given the gradient it actually
received downstream in the real graph.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRules,
    half_rule,
    identity_rule,
    ln_rule_grad,
    use_relevance_rules,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.gated_mlp import (
    resolve_activation_fn,
)
from transformer_lens.model_bridge.sources import build_bridge_config_from_hf
from transformer_lens.model_bridge.supported_architectures.phi3 import (
    Phi3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)

TOKENS = torch.tensor([[1, 5, 7, 42, 9]])
N_LAYERS = 2
TINY_DIMS = dict(
    vocab_size=97,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=N_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)


class _MockTokenizer:
    """Stand-in to satisfy TransformerBridge(tokenizer=...)."""


def _build_qwen2_bridge() -> TransformerBridge:
    hf_config = AutoConfig.for_model("qwen2", **TINY_DIMS)
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "Qwen2ForCausalLM", "qwen2-tiny", torch.float32
    )
    adapter = Qwen2ArchitectureAdapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_MockTokenizer())


def _build_phi3_bridge() -> TransformerBridge:
    hf_config = AutoConfig.for_model("phi3", **TINY_DIMS)
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "Phi3ForCausalLM", "phi3-tiny", torch.float32
    )
    adapter = Phi3ArchitectureAdapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_MockTokenizer())


# Qwen2 exercises GatedMLPBridge's opaque recompute-from-weights path (the
# primary path per real usage); Phi-3 exercises JointGateUpMLPBridge's
# already-reconstructed forward, isolating rule bugs from Bridge-integration
# bugs on the fused-projection family.
FIXTURE_BUILDERS = {
    "qwen2": _build_qwen2_bridge,
    "phi3": _build_phi3_bridge,
}


@pytest.fixture(scope="module", params=sorted(FIXTURE_BUILDERS), ids=sorted(FIXTURE_BUILDERS))
def bridge(request: pytest.FixtureRequest) -> TransformerBridge:
    return FIXTURE_BUILDERS[request.param]()


def _expected_canonical_mounts() -> set[str]:
    ln_mounts = {f"blocks.{i}.ln1" for i in range(N_LAYERS)} | {
        f"blocks.{i}.ln2" for i in range(N_LAYERS)
    }
    mlp_mounts = {f"blocks.{i}.mlp" for i in range(N_LAYERS)}
    return ln_mounts | mlp_mounts


class TestForwardIdentity:
    def test_active_forward_matches_baseline_under_all_three_rules(
        self, bridge: TransformerBridge
    ) -> None:
        with torch.no_grad():
            baseline = bridge(TOKENS)
        with use_relevance_rules(
            bridge, RelevanceRules(normalization=True, activation=True, multiplicative_gate=True)
        ):
            with torch.no_grad():
                active = bridge(TOKENS)
        assert torch.equal(active, baseline)


class TestCoverage:
    def test_every_canonical_mount_is_installed_and_none_skipped(
        self, bridge: TransformerBridge
    ) -> None:
        with use_relevance_rules(
            bridge, RelevanceRules(normalization=True, activation=True, multiplicative_gate=True)
        ) as coverage:
            pass
        assert set(coverage.installed) == _expected_canonical_mounts()
        assert coverage.skipped == ()


class TestGradientMatchesClosedFormOracle:
    """Each rule-active node's local VJP, checked against the gradient it
    actually receives in the real graph -- not a hand-rederived whole-model
    oracle. The rule Functions compute a purely local closed-form VJP (already
    proven against analytic oracles in the primitive and single-component
    tests), so the only thing a real multi-block graph can newly break is the
    wiring: the wrong node's weights, a disconnected graph path, or a mount
    that silently never resolves so its rule never activates. Tapping
    hook_in/hook_out -- outside the LN-rule's fail-closed
    hook_scale/hook_normalized guard -- exposes exactly the gradient each node
    passes upstream and the gradient it receives from downstream, with no
    need to reconstruct attention or the rest of the stack by hand.
    """

    def test_ln_rule_and_gated_mlp_rule_match_local_oracles(
        self, bridge: TransformerBridge
    ) -> None:
        ln1 = bridge.blocks[0].ln1
        mlp = bridge.blocks[0].mlp
        captured: dict[str, torch.Tensor] = {}

        def _capture(key: str):
            def _hook(tensor: torch.Tensor, hook=None) -> None:
                captured[key] = tensor.detach().clone()

            return _hook

        ln1.hook_in.add_hook(_capture("ln_x"))
        ln1.hook_in.add_hook(_capture("ln_grad_in"), dir="bwd")
        ln1.hook_out.add_hook(_capture("ln_grad_out"), dir="bwd")
        mlp.hook_in.add_hook(_capture("mlp_x"))
        mlp.hook_in.add_hook(_capture("mlp_grad_in"), dir="bwd")
        mlp.hook_out.add_hook(_capture("mlp_grad_out"), dir="bwd")
        ln1_weight_grad = None
        try:
            with use_relevance_rules(
                bridge,
                RelevanceRules(normalization=True, activation=True, multiplicative_gate=True),
            ):
                logits = bridge(TOKENS)
                logits.sum().backward()
            ln1_weight_grad = ln1.weight.grad.clone()
        finally:
            ln1.hook_in.remove_hooks(dir="both")
            ln1.hook_out.remove_hooks(dir="both")
            mlp.hook_in.remove_hooks(dir="both")
            mlp.hook_out.remove_hooks(dir="both")
            for parameter in bridge.parameters():
                parameter.grad = None

        eps = getattr(ln1.original_component, "variance_epsilon", 1e-6)
        weight = ln1.weight.detach()
        denom = (captured["ln_x"].pow(2).mean(-1, keepdim=True) + eps).sqrt()
        expected_ln_grad_in = ln_rule_grad(captured["ln_grad_out"] * weight, denom)
        torch.testing.assert_close(
            captured["ln_grad_in"], expected_ln_grad_in, atol=1e-5, rtol=1e-4
        )
        # weight keeps its ordinary gradient: the rule only redefines the x-path
        # VJP, not d(output)/d(weight), given the same grad_out the rule-active
        # forward actually produced.
        expected_weight_grad = (captured["ln_grad_out"] * (captured["ln_x"] / denom)).sum(
            dim=(0, 1)
        )
        torch.testing.assert_close(ln1_weight_grad, expected_weight_grad, atol=1e-5, rtol=1e-4)

        x = captured["mlp_x"].detach().requires_grad_(True)
        w_gate, w_in, w_out = mlp.W_gate.detach(), mlp.W_in.detach(), mlp.W_out.detach()
        b_gate = getattr(mlp.gate, "bias", None)
        b_in = getattr(getattr(mlp, "in"), "bias", None)
        b_out = getattr(mlp.out, "bias", None)
        act_fn = resolve_activation_fn(mlp.config)
        with torch.enable_grad():
            gate_output = x @ w_gate
            if b_gate is not None:
                gate_output = gate_output + b_gate
            up_output = x @ w_in
            if b_in is not None:
                up_output = up_output + b_in
            activated = identity_rule(gate_output, act_fn)
            gated = half_rule(activated, up_output)
            down = gated @ w_out
            if b_out is not None:
                down = down + b_out
        (expected_mlp_grad_in,) = torch.autograd.grad(
            down, x, grad_outputs=captured["mlp_grad_out"]
        )
        torch.testing.assert_close(
            captured["mlp_grad_in"], expected_mlp_grad_in, atol=1e-5, rtol=1e-4
        )
