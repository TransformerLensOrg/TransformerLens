"""SymbolicBridge hook mirroring: fc-split archs (OPT/XGLM) must fire the placeholder's
own ``mlp.hook_in``/``mlp.hook_out`` — wired at setup from the ``in``/``out`` subcomponents
(see ``component_setup._wire_symbolic_hooks``). Without the mirror those HookPoints exist
in the registry but never fire, so caching misses them and interventions silently no-op.

Runs on CPU with token-free tiny-random checkpoints.
"""
from __future__ import annotations

import pytest
import torch

OPT = "hf-internal-testing/tiny-random-OPTForCausalLM"
XGLM = "hf-internal-testing/tiny-random-XGLMForCausalLM"


@pytest.fixture(scope="module")
def opt_bridge():
    from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

    return TransformerBridge.boot_transformers(OPT, device="cpu", dtype=torch.float32)


class TestSymbolicMirror:
    def test_mlp_hooks_fire_and_match_subcomponents(self, opt_bridge):
        tokens = opt_bridge.to_tokens("Hello world")
        _, cache = opt_bridge.run_with_cache(tokens)
        assert "blocks.0.mlp.hook_out" in cache.cache_dict
        assert "blocks.0.mlp.hook_in" in cache.cache_dict
        # The mirror re-fires the subcomponent tensors: out = fc2's output, in = fc1's input.
        assert torch.equal(cache["blocks.0.mlp.hook_out"], cache["blocks.0.mlp.out.hook_out"])
        assert torch.equal(cache["blocks.0.mlp.hook_in"], cache["blocks.0.mlp.in.hook_in"])

    def test_intervention_via_symbolic_hook_propagates(self, opt_bridge):
        tokens = opt_bridge.to_tokens("Hello world")
        base = opt_bridge.forward(tokens)
        patched = opt_bridge.run_with_hooks(
            tokens, fwd_hooks=[("blocks.0.mlp.hook_out", lambda t, hook: torch.zeros_like(t))]
        )
        assert not torch.allclose(base, patched), "symbolic-hook edit did not reach the logits"

    def test_mirror_survives_reset_hooks(self, opt_bridge):
        """The wiring is permanent — a reset_hooks (any run_with_* teardown) must not
        sever it, else the second run_with_cache silently loses the mlp entries."""
        tokens = opt_bridge.to_tokens("Hello world")
        opt_bridge.reset_hooks()
        _, cache = opt_bridge.run_with_cache(tokens)
        assert "blocks.0.mlp.hook_out" in cache.cache_dict

    def test_xglm_mirror_fires(self):
        from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

        bridge = TransformerBridge.boot_transformers(XGLM, device="cpu", dtype=torch.float32)
        _, cache = bridge.run_with_cache(bridge.to_tokens("Hi"))
        assert "blocks.0.mlp.hook_out" in cache.cache_dict
        assert torch.equal(cache["blocks.0.mlp.hook_out"], cache["blocks.0.mlp.out.hook_out"])
