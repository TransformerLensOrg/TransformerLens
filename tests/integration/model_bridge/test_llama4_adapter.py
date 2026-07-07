"""Integration tests for the Llama 4 (text) architecture adapter.

The only public Llama4ForCausalLM checkpoint
(trl-internal-testing/tiny-Llama4ForCausalLM) ships uninitialized
feed_forward.experts tensors (NaN / ~1e38), so the fixture re-initializes
those four tensors deterministically and tests against the local snapshot.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "trl-internal-testing/tiny-Llama4ForCausalLM"


@pytest.fixture(scope="module")
def snapshot_path(tmp_path_factory):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = tmp_path_factory.mktemp("llama4") / "tiny-llama4-sanitized"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    torch.manual_seed(42)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "feed_forward.experts" in name:
                param.copy_(torch.randn_like(param) * 0.02)
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def llama4_bridge(snapshot_path):
    return TransformerBridge.boot_transformers(snapshot_path, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(llama4_bridge):
    torch.manual_seed(0)
    return torch.randint(0, llama4_bridge.cfg.d_vocab - 10, (1, 12))


class TestLlama4BridgeCreation:
    def test_adapter_selected(self, llama4_bridge):
        from transformer_lens.model_bridge.supported_architectures.llama4 import (
            Llama4ArchitectureAdapter,
        )

        assert isinstance(llama4_bridge.adapter, Llama4ArchitectureAdapter)


class TestLlama4ForwardEquivalence:
    def test_forward_matches_fresh_hf(self, llama4_bridge, snapshot_path, sample_tokens):
        from transformers import AutoModelForCausalLM

        fresh = AutoModelForCausalLM.from_pretrained(
            snapshot_path, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = llama4_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestLlama4Hooks:
    def test_hooks_fire_in_tl_shapes(self, llama4_bridge, sample_tokens):
        """Llama4TextMoe flattens to [batch*seq, d]; hook_out must still be 3D."""
        d_model = llama4_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.1.mlp.hook_out"]
        with torch.no_grad():
            llama4_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"

    def test_router_scores_hook(self, llama4_bridge, sample_tokens):
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            llama4_bridge.run_with_hooks(
                sample_tokens,
                fwd_hooks=[("blocks.0.mlp.hook_router_scores", grab)],
            )
        n_experts = llama4_bridge.original_model.config.num_local_experts
        assert captured.get("blocks.0.mlp.hook_router_scores") == (
            sample_tokens.shape[1],
            n_experts,
        )

    def test_mlp_hook_edit_propagates(self, llama4_bridge, sample_tokens):
        with torch.no_grad():
            baseline = llama4_bridge(sample_tokens)
            edited = llama4_bridge.run_with_hooks(
                sample_tokens,
                fwd_hooks=[("blocks.1.mlp.hook_out", lambda t, hook: torch.zeros_like(t))],
            )
        assert not torch.allclose(edited, baseline)

    def test_backward_through_moe(self, llama4_bridge, sample_tokens):
        """HF accumulates routed output in place on the shared-expert result;
        the adapter clones under grad so backward hooks survive."""
        grads = {}

        def grab_grad(tensor, hook):
            grads[hook.name] = tensor.detach().clone()

        with llama4_bridge.hooks(bwd_hooks=[("blocks.0.mlp.hook_out", grab_grad)]):
            out = llama4_bridge(sample_tokens)
            out.sum().backward()
        assert "blocks.0.mlp.hook_out" in grads
        assert grads["blocks.0.mlp.hook_out"].shape == (
            1,
            sample_tokens.shape[1],
            llama4_bridge.cfg.d_model,
        )


class TestLlama4Generation:
    def test_generate(self, llama4_bridge):
        text = llama4_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
