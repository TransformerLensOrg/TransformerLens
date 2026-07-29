"""Integration tests for the Idefics3 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "ibm-granite/granite-docling-258M"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


class TestIdefics3BridgeCreation:
    def test_adapter_and_components(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.idefics3 import (
            Idefics3ArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, Idefics3ArchitectureAdapter)
        assert bridge.cfg.is_multimodal is True
        assert hasattr(bridge, "vision_encoder")
        assert hasattr(bridge, "vision_projector")

    def test_vision_tower_is_live(self, bridge):
        hf_model = bridge.original_model
        assert bridge.vision_encoder is hf_model.model.vision_model
        assert bridge.vision_projector is hf_model.model.connector


class TestIdefics3ForwardEquivalence:
    def test_text_forward_matches_hf(self, bridge, sample_tokens):
        hf_model = bridge.original_model
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestIdefics3Hooks:
    def test_text_hooks_fire(self, bridge, sample_tokens):
        d_model = bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out"]
        with torch.no_grad():
            bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
