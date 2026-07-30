"""Integration tests for the Qwen3.5-MoE architecture adapters."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6"


@pytest.fixture(scope="module")
def moe_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(moe_bridge):
    torch.manual_seed(0)
    return torch.randint(0, moe_bridge.cfg.d_vocab - 10, (1, 8))


class TestQwen3_5MoeBridgeCreation:
    def test_routes_to_multimodal_moe_adapter(self, moe_bridge):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
            Qwen3_5MoeMultimodalArchitectureAdapter,
        )

        assert isinstance(moe_bridge.adapter, Qwen3_5MoeMultimodalArchitectureAdapter)
        assert moe_bridge.cfg.is_multimodal is True

    def test_has_vision_and_language_components(self, moe_bridge):
        assert hasattr(moe_bridge, "vision_encoder")
        assert hasattr(moe_bridge, "vision_projector")
        assert hasattr(moe_bridge, "embed")
        assert hasattr(moe_bridge, "unembed")

    def test_hybrid_layer_types_respected(self, moe_bridge):
        """Layer 0 is GatedDeltaNet (linear_attn), layer 1 full attention in the tiny checkpoint."""
        layer_types = moe_bridge.original_model.config.text_config.layer_types
        assert layer_types == ["linear_attention", "full_attention"]
        hook_keys = set(moe_bridge.hook_dict.keys())
        # startswith: the vision tower registers its own vision_encoder.blocks.*.attn hooks
        assert any(k.startswith("blocks.0.linear_attn") for k in hook_keys)
        assert not any(k.startswith("blocks.0.attn.") for k in hook_keys)
        assert any(k.startswith("blocks.1.attn.") for k in hook_keys)
        assert not any(k.startswith("blocks.1.linear_attn") for k in hook_keys)


class TestQwen3_5MoeForwardEquivalence:
    def test_forward_matches_hf(self, moe_bridge, sample_tokens):
        hf_model = moe_bridge.original_model
        with torch.no_grad():
            bridge_out = moe_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestQwen3_5MoeHooks:
    def test_moe_mlp_and_hybrid_attention_hooks_fire(self, moe_bridge, sample_tokens):
        d_model = moe_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "blocks.0.mlp.hook_out",
            "blocks.0.linear_attn.hook_out",
            "blocks.1.attn.hook_out",
        ]
        with torch.no_grad():
            moe_bridge.run_with_hooks(
                sample_tokens,
                use_cache=False,
                fwd_hooks=[(name, grab) for name in hooks],
            )
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"

    def test_moe_block_is_delegated_hf_module(self, moe_bridge):
        from transformer_lens.model_bridge.generalized_components import MoEBridge

        hf_model = moe_bridge.original_model
        assert isinstance(moe_bridge.blocks[0].mlp, MoEBridge)
        assert moe_bridge.blocks[0].mlp is hf_model.model.language_model.layers[0].mlp
