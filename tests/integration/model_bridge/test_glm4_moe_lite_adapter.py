"""Integration tests for the GLM-4.7-Flash (Glm4MoeLite) architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/glm-4.7-flash"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


class TestGlm4MoeLiteBridgeCreation:
    def test_adapter_selected(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.glm4_moe_lite import (
            Glm4MoeLiteArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, Glm4MoeLiteArchitectureAdapter)

    def test_dense_sparse_layer_mix(self, bridge):
        """The tiny checkpoint declares layer 0 dense, layer 1 sparse; the mlp
        template binds each accordingly (#1645)."""
        hf_model = bridge.original_model
        assert hf_model.config.mlp_layer_types == ["dense", "sparse"]
        dense_mlp = bridge.blocks[0].mlp
        sparse_mlp = bridge.blocks[1].mlp
        # Dense binding: gated-MLP neuron hooks via the dense_* projections.
        assert not hasattr(dense_mlp.original_component, "experts")
        assert dense_mlp._bound_dense is True
        assert dense_mlp.hook_aliases["hook_pre"] == "dense_gate.hook_out"
        assert not hasattr(dense_mlp, "hook_router_scores")
        # Sparse binding: router + shared experts survive setup on the BRIDGE
        # (a silently-popped optional router would otherwise stay green here).
        assert hasattr(sparse_mlp.original_component, "experts")
        assert {"gate", "shared_experts"} <= set(sparse_mlp.submodules)
        assert hasattr(sparse_mlp, "hook_router_scores")


class TestGlm4MoeLiteForwardEquivalence:
    def test_forward_matches_hf(self, bridge, sample_tokens):
        hf_model = bridge.original_model
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestGlm4MoeLiteMLAHooks:
    def test_mla_projection_hooks_fire(self, bridge, sample_tokens):
        hf_config = bridge.original_model.config
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "blocks.0.attn.kv_a_proj_with_mqa.hook_out",
            "blocks.0.attn.q_a_proj.hook_out",
            "blocks.0.attn.hook_out",
        ]
        with torch.no_grad():
            bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        assert captured["blocks.0.attn.q_a_proj.hook_out"] == (1, seq, hf_config.q_lora_rank)
        assert captured["blocks.0.attn.kv_a_proj_with_mqa.hook_out"] == (
            1,
            seq,
            hf_config.kv_lora_rank + hf_config.qk_rope_head_dim,
        )
        assert captured["blocks.0.attn.hook_out"] == (1, seq, bridge.cfg.d_model)

    def test_shared_expert_hook_fires_on_sparse_layer(self, bridge, sample_tokens):
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            bridge.run_with_hooks(
                sample_tokens,
                fwd_hooks=[("blocks.1.mlp.shared_experts.hook_out", grab)],
            )
        seq = sample_tokens.shape[1]
        assert captured["blocks.1.mlp.shared_experts.hook_out"] == (
            1,
            seq,
            bridge.cfg.d_model,
        )


class TestGlm4MoeLiteHFDelegation:
    def test_components_are_shared_wrappers(self, bridge):
        hf_model = bridge.original_model
        assert (
            bridge.blocks[0].attn.kv_a_proj_with_mqa
            is hf_model.model.layers[0].self_attn.kv_a_proj_with_mqa
        )
        assert bridge.blocks[1].mlp is hf_model.model.layers[1].mlp
