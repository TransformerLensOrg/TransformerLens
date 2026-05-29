"""Acceptance gate for the Inspect driver: gpt2 through our HF provider must match
``boot_transformers`` to fp tolerance (same backend + dtype + eager attention), with
exact next-token argmax, and interventions must take effect.

Gated on ``inspect_ai`` being installed (the ``inspect`` extra); runs on CPU.
"""
from __future__ import annotations

import pytest
import torch

try:
    import inspect_ai  # noqa: F401

    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False

pytestmark = [
    pytest.mark.inspect,
    pytest.mark.skipif(
        not INSPECT_AVAILABLE, reason="inspect_ai not installed (pip install '.[inspect]')"
    ),
]

MODEL = "gpt2"
PROMPT = "The quick brown fox"
# A spread across boundary kinds and layers — all must match boot_transformers to fp
# tolerance. TransformerBridge-native names (the bridge cache carries these too).
PARITY_HOOKS = [
    "blocks.0.hook_in",  # resid_pre
    "blocks.0.attn.hook_out",  # attn_out
    "blocks.0.ln2.hook_in",  # resid_mid
    "blocks.0.mlp.hook_out",  # mlp_out
    "blocks.0.hook_out",  # resid_post
    "blocks.6.attn.hook_out",
    "blocks.11.hook_out",
]

# Boundary kind -> TransformerBridge-native hook suffix.
KIND_SUFFIX = {
    "resid_pre": "hook_in",
    "resid_mid": "ln2.hook_in",
    "resid_post": "hook_out",
    "attn_out": "attn.hook_out",
    "mlp_out": "mlp.hook_out",
}

# Token-free tiny-random checkpoints, one per structural-check code path: standard
# sequential (nothing gated), parallel-residual (resid_mid gated by the causal probe),
# post-norm (resid_mid gated by the linear-identity probe). Pins cross-family behavior
# so a detector/load-path regression on non-gpt2 archs fails CI instead of shipping silently.
STRUCTURAL_FAMILIES = [
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", frozenset()),
    ("hf-internal-testing/tiny-random-GPTJForCausalLM", frozenset({"resid_mid"})),
    ("hf-internal-testing/tiny-random-Gemma2ForCausalLM", frozenset({"resid_mid"})),
]


@pytest.fixture(scope="module")
def hf_bridge():
    from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

    return TransformerBridge.boot_transformers(MODEL)


@pytest.fixture(scope="module")
def inspect_bridge():
    from transformer_lens.model_bridge.remote_bridge import RemoteBridge

    bridge = RemoteBridge.boot_inspect(MODEL)
    yield bridge
    bridge.close()


@pytest.fixture(scope="module")
def tokens(hf_bridge):
    # Shared token ids so the only variable under test is the backend path, not
    # tokenization (to_tokens BOS parity is covered by test_string_input_parity).
    return hf_bridge.to_tokens(PROMPT)


class TestInspectParity:
    def test_argmax_matches_boot_transformers(self, hf_bridge, inspect_bridge, tokens):
        hf_logits = hf_bridge.forward(tokens)
        i_logits = inspect_bridge.forward(tokens)
        assert int(hf_logits[0, -1].argmax()) == int(i_logits[0, -1].argmax())

    def test_string_input_parity(self, hf_bridge, inspect_bridge):
        # Same STRING (not shared tokens): exercises RemoteBridge.to_tokens BOS handling.
        # A bare encode (no BOS) would diverge from boot_transformers here.
        hf_argmax = int(hf_bridge.forward(PROMPT)[0, -1].argmax())
        i_argmax = int(inspect_bridge.forward(PROMPT)[0, -1].argmax())
        assert hf_argmax == i_argmax

    def test_hooks_match(self, hf_bridge, inspect_bridge, tokens):
        _, hf_cache = hf_bridge.run_with_cache(tokens)
        _, i_cache = inspect_bridge.run_with_cache(tokens)
        for hook in PARITY_HOOKS:
            a, b = hf_cache[hook].float(), i_cache[hook].float()
            assert a.shape == b.shape
            assert torch.allclose(
                a, b, atol=1e-3, rtol=1e-3
            ), f"{hook} diverges: max {(a - b).abs().max().item():.2e}"

    def test_loss_matches(self, hf_bridge, inspect_bridge, tokens):
        # Full-sequence logits ⇒ loss is computable and matches boot_transformers.
        hf_loss = hf_bridge.forward(tokens, return_type="loss")
        i_loss = inspect_bridge.forward(tokens, return_type="loss")
        assert torch.allclose(hf_loss, i_loss, atol=1e-3)


# Each intervenable boundary kind at layer 0, with whether suppress flips the top token
# (mlp_out shifts the logits but not the argmax at layer 0). resid_pre exercises the
# forward_pre_hook path; the others the forward_hook path.
INTERVENE_HOOKS = [
    ("blocks.0.hook_in", True),  # resid_pre
    ("blocks.0.attn.hook_out", True),  # attn_out
    ("blocks.0.mlp.hook_out", False),  # mlp_out
    ("blocks.0.hook_out", True),  # resid_post
]


class TestInspectInterventions:
    @pytest.mark.parametrize("hook,flips_argmax", INTERVENE_HOOKS)
    def test_suppress_applies_at_each_boundary(self, inspect_bridge, tokens, hook, flips_argmax):
        clean_logits = inspect_bridge.forward(tokens)
        clean_argmax = int(clean_logits[0, -1].argmax())

        supp_logits, supp_cache = inspect_bridge.run_with_cache(
            tokens, intervene={hook: {"op": "suppress"}}
        )
        assert supp_cache[hook].abs().max().item() == 0.0  # capture reflects the intervention
        assert not torch.allclose(supp_logits, clean_logits)  # and it propagated to the logits
        if flips_argmax:
            assert int(supp_logits[0, -1].argmax()) != clean_argmax

    def test_intervention_reverts(self, inspect_bridge, tokens):
        clean_argmax = int(inspect_bridge.forward(tokens)[0, -1].argmax())
        inspect_bridge.forward(tokens, intervene={"blocks.0.hook_out": {"op": "suppress"}})
        assert int(inspect_bridge.forward(tokens)[0, -1].argmax()) == clean_argmax  # not sticky


class TestPreHookKwargs:
    """The resid_pre pre-hook runs with_kwargs=True so it modifies the right tensor
    whether hidden_states arrives positionally (args[0]) or as a kwarg."""

    def _hook(self):
        from transformer_lens.model_bridge.sources.inspect.provider import _pre_hook

        return _pre_hook(layer=0, want_capture=True, spec={"op": "suppress"}, raw={})

    def test_positional_hidden_states(self):
        hidden = torch.ones(1, 2, 4)
        new_args, new_kwargs = self._hook()(None, (hidden, "mask"), {})
        assert torch.allclose(new_args[0], torch.zeros_like(hidden)) and new_args[1] == "mask"
        assert new_kwargs == {}

    def test_kwarg_hidden_states(self):
        hidden = torch.ones(1, 2, 4)
        new_args, new_kwargs = self._hook()(None, (), {"hidden_states": hidden, "use_cache": True})
        assert new_args == ()
        assert torch.allclose(new_kwargs["hidden_states"], torch.zeros_like(hidden))  # suppressed
        assert new_kwargs["use_cache"] is True  # other kwargs preserved


class TestStructuralProbe:
    """The boot-time structural self-check: resid_mid is offered only when attn feeds mlp.
    Toy modules (no download) exercise the causal probe for sequential vs parallel blocks."""

    def _toy_model(self, parallel: bool = False, resid_scale: float = 1.0):
        import torch.nn as nn

        class Sub(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.lin = nn.Linear(4, 4)
                self.k = k

            def forward(self, x):
                return self.lin(x) * self.k

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Sub(0.5)
                self.mlp = Sub(0.1)

            def forward(self, x):
                a = self.self_attn(x)
                m = self.mlp(x) if parallel else self.mlp(x + a)  # parallel reads block input
                # resid_scale != 1 mimics post-norm/residual-multiplier archs (Gemma2/OLMo2/
                # Granite): outputs don't add linearly, so resid_pre + attn_out is wrong.
                return x + resid_scale * a + resid_scale * m

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 4)
                self.layers = nn.ModuleList([Block()])

            def forward(self, ids):
                x = self.embed(ids)
                for b in self.layers:
                    x = b(x)
                return x

        torch.manual_seed(0)
        return Model()

    def test_sequential_offers_resid_mid(self):
        from transformer_lens.model_bridge.sources.inspect.provider import (
            _detect_capabilities,
        )

        m = self._toy_model()  # standard sequential, linear residual
        kinds, note = _detect_capabilities(m, m.layers)
        assert "resid_mid" in kinds
        assert note == ""  # nothing gated

    def test_parallel_gates_resid_mid(self):
        from transformer_lens.model_bridge.sources.inspect.provider import (
            _detect_capabilities,
        )

        m = self._toy_model(parallel=True)  # mlp reads block input, not attn output
        kinds, note = _detect_capabilities(m, m.layers)
        assert "resid_mid" not in kinds
        assert {"resid_pre", "resid_post", "attn_out", "mlp_out"} <= kinds  # rest still served
        assert "resid_mid" in note  # note explains the gate

    def test_nonlinear_residual_gates_resid_mid(self):
        from transformer_lens.model_bridge.sources.inspect.provider import (
            _detect_capabilities,
        )

        # Sequential (attn feeds mlp) but outputs are scaled before the residual add, so
        # resid_post != resid_pre + attn_out + mlp_out — resid_mid must still be gated.
        m = self._toy_model(resid_scale=2.0)
        kinds, note = _detect_capabilities(m, m.layers)
        assert "resid_mid" not in kinds
        assert {"resid_pre", "resid_post", "attn_out", "mlp_out"} <= kinds
        assert "resid_mid" in note


class TestStructuralCheckAcrossFamilies:
    """Real tiny-random models, one per detector code path, run in CI (no token, seconds).
    Locks both the gating decision and offered-boundary parity vs boot_transformers, so a
    structural-check or load-path regression on non-gpt2 architectures fails here."""

    @pytest.mark.parametrize("model_id,expected_gated", STRUCTURAL_FAMILIES)
    def test_gating_and_parity(self, model_id, expected_gated):
        from transformer_lens.model_bridge.remote_bridge import RemoteBridge
        from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

        # Matched fp32 + eager attention (boot_inspect's defaults) so the only variable
        # is the boundary mapping.
        hf = TransformerBridge.boot_transformers(model_id, device="cpu", dtype=torch.float32)
        inspect = RemoteBridge.boot_inspect(model_id, dtype=torch.float32)
        try:
            supported = inspect._driver.supported_hook_points
            offered = {k for k, suf in KIND_SUFFIX.items() if f"blocks.0.{suf}" in supported}
            gated = frozenset(set(KIND_SUFFIX) - offered)
            assert gated == expected_gated, f"{model_id}: gated {sorted(gated)}"
            # gated kinds are reported non-fireable, not silently dropped
            for kind in expected_gated:
                assert f"blocks.0.{KIND_SUFFIX[kind]}" in inspect._driver.non_fireable_hook_points

            n_layers = int(hf.cfg.n_layers)
            toks = hf.to_tokens(PROMPT)
            _, hf_cache = hf.run_with_cache(toks)
            _, i_cache = inspect.run_with_cache(toks)
            assert int(hf.forward(toks)[0, -1].argmax()) == int(
                inspect.forward(toks)[0, -1].argmax()
            )
            for layer in sorted({0, n_layers - 1}):
                for kind in offered:
                    hk = f"blocks.{layer}.{KIND_SUFFIX[kind]}"
                    a, b = hf_cache[hk].float(), i_cache[hk].float()
                    assert torch.allclose(
                        a, b, atol=1e-3, rtol=1e-3
                    ), f"{model_id} {hk} diverges: {(a - b).abs().max().item():.2e}"
        finally:
            inspect.close()
