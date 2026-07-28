"""Integration tests for the AFMoE (Arcee Trinity) architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "onnx-internal-testing/tiny-random-AfmoeForCausalLM"


@pytest.fixture(scope="module")
def afmoe_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(afmoe_bridge):
    torch.manual_seed(0)
    return torch.randint(0, afmoe_bridge.cfg.d_vocab - 10, (1, 12))


class TestAfmoeBridgeCreation:
    def test_adapter_selected(self, afmoe_bridge):
        from transformer_lens.model_bridge.supported_architectures.afmoe import (
            AfmoeArchitectureAdapter,
        )

        assert isinstance(afmoe_bridge.adapter, AfmoeArchitectureAdapter)


class TestAfmoeForwardEquivalence:
    def test_forward_matches_fresh_hf(self, afmoe_bridge, sample_tokens):
        from transformers import AutoModelForCausalLM

        fresh = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = afmoe_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestAfmoeHooks:
    def test_hooks_fire(self, afmoe_bridge, sample_tokens):
        """Covers a dense layer (0), a MoE layer (2), sandwich norms, and the
        attention output gate."""
        d_model = afmoe_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
            "blocks.2.mlp.hook_out": (1, seq, d_model),
            "blocks.0.ln1_post.hook_out": (1, seq, d_model),
            "blocks.0.attn.gate.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            afmoe_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"

    def test_router_gate_hook_moe_layers_only(self, afmoe_bridge, sample_tokens):
        """Router only exists at layer >= num_dense_layers; HF flattens its
        input so the hook is [batch*seq, num_experts]."""
        n_experts = afmoe_bridge.original_model.config.num_experts
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            afmoe_bridge.run_with_hooks(
                sample_tokens,
                fwd_hooks=[("blocks.2.mlp.router_gate.hook_out", grab)],
            )
        assert captured.get("blocks.2.mlp.router_gate.hook_out") == (
            sample_tokens.shape[1],
            n_experts,
        )

    def test_mlp_hook_edit_propagates(self, afmoe_bridge, sample_tokens):
        with torch.no_grad():
            baseline = afmoe_bridge(sample_tokens)
            edited = afmoe_bridge.run_with_hooks(
                sample_tokens,
                fwd_hooks=[("blocks.2.mlp.hook_out", lambda t, hook: torch.zeros_like(t))],
            )
        assert not torch.allclose(edited, baseline)


class TestAfmoeGeneration:
    def test_generate(self, afmoe_bridge):
        text = afmoe_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
