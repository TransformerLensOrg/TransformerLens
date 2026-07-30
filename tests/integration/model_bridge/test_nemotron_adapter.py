"""Integration tests for the dense Nemotron architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "badaoui/tiny-random-NemotronForCausalLM"


@pytest.fixture(scope="module")
def nemotron_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(nemotron_bridge):
    torch.manual_seed(0)
    return torch.randint(0, nemotron_bridge.cfg.d_vocab - 10, (1, 8))


class TestNemotronBridgeCreation:
    def test_adapter_selected(self, nemotron_bridge):
        from transformer_lens.model_bridge.supported_architectures.nemotron import (
            NemotronArchitectureAdapter,
        )

        assert isinstance(nemotron_bridge.adapter, NemotronArchitectureAdapter)

    def test_layernorm1p_offset_is_live(self, nemotron_bridge):
        """The wrapped norm must be NVIDIA's LayerNorm1P (weight + 1 gamma)."""
        ln = nemotron_bridge.blocks[0].ln1.original_component
        assert type(ln).__name__ == "NemotronLayerNorm1P"


class TestNemotronForwardEquivalence:
    def test_forward_matches_hf(self, nemotron_bridge, sample_tokens):
        hf_model = nemotron_bridge.original_model
        with torch.no_grad():
            bridge_out = nemotron_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestNemotronHooks:
    def test_hooks_fire(self, nemotron_bridge, sample_tokens):
        d_model = nemotron_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.1.hook_out"]
        with torch.no_grad():
            nemotron_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in hooks]
            )
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
