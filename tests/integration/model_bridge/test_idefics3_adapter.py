"""Integration tests for the Idefics3 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "ibm-granite/granite-docling-258M"


@pytest.fixture(scope="module")
def idefics_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(idefics_bridge):
    torch.manual_seed(0)
    return torch.randint(0, idefics_bridge.cfg.d_vocab - 10, (1, 8))


class TestIdefics3BridgeCreation:
    def test_adapter_and_components(self, idefics_bridge):
        from transformer_lens.model_bridge.supported_architectures.idefics3 import (
            Idefics3ArchitectureAdapter,
        )

        assert isinstance(idefics_bridge.adapter, Idefics3ArchitectureAdapter)
        assert idefics_bridge.cfg.is_multimodal is True
        assert hasattr(idefics_bridge, "vision_encoder")
        assert hasattr(idefics_bridge, "vision_projector")

    def test_vision_tower_is_live(self, idefics_bridge):
        hf_model = idefics_bridge.original_model
        assert idefics_bridge.vision_encoder is hf_model.model.vision_model
        assert idefics_bridge.vision_projector is hf_model.model.connector


class TestIdefics3ForwardEquivalence:
    def test_text_forward_matches_hf(self, idefics_bridge, sample_tokens):
        hf_model = idefics_bridge.original_model
        with torch.no_grad():
            bridge_out = idefics_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestIdefics3Hooks:
    def test_text_hooks_fire(self, idefics_bridge, sample_tokens):
        d_model = idefics_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out"]
        with torch.no_grad():
            idefics_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
