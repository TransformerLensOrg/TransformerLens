"""Acceptance gate for the Inspect driver: gpt2 through our HF provider must match
``boot_transformers`` exactly (same backend), and interventions must take effect.

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
# A spread across boundary kinds and layers — all must match boot_transformers
# exactly. TransformerBridge-native names (the bridge cache carries these too).
PARITY_HOOKS = [
    "blocks.0.hook_in",  # resid_pre
    "blocks.0.attn.hook_out",  # attn_out
    "blocks.0.ln2.hook_in",  # resid_mid
    "blocks.0.mlp.hook_out",  # mlp_out
    "blocks.0.hook_out",  # resid_post
    "blocks.6.attn.hook_out",
    "blocks.11.hook_out",
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
    # Shared token ids (BOS-prepended, the TL convention) so the only variable is
    # the backend path — RemoteBridge.to_tokens doesn't prepend BOS on its own.
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
