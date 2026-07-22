"""Unit tests for the Qwen3.5-MoE architecture adapters.

Covers the MoE-specific deltas from the dense Qwen3.5 adapters; the shared
GatedDeltaNet machinery is exercised by test_qwen3_5_adapter.py.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES

try:
    from transformers import Qwen3_5MoeForCausalLM as _Qwen3_5MoeForCausalLM
    from transformers import Qwen3_5MoeTextConfig

    _QWEN3_5_MOE_AVAILABLE = True
except ImportError:
    _QWEN3_5_MOE_AVAILABLE = False


def _make_bridge_cfg(**overrides):
    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )

    defaults = dict(
        d_model=1024,
        d_head=256,
        n_heads=8,
        n_layers=24,
        n_ctx=2048,
        d_vocab=248320,
        n_key_value_heads=2,
        architecture="Qwen3_5MoeForCausalLM",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


class TestQwen3_5MoeArchitectureDetection:
    def test_model_type_qwen3_5_moe_routes_to_text_only_architecture(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="qwen3_5_moe", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Qwen3_5MoeForCausalLM"

    def test_model_type_qwen3_5_moe_text_routes_to_text_only_architecture(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="qwen3_5_moe_text", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Qwen3_5MoeForCausalLM"

    def test_conditional_generation_routes_to_multimodal_adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
            Qwen3_5MoeArchitectureAdapter,
            Qwen3_5MoeMultimodalArchitectureAdapter,
        )

        assert "Qwen3_5MoeForConditionalGeneration" in SUPPORTED_ARCHITECTURES
        assert "Qwen3_5MoeForConditionalGeneration" in HF_SUPPORTED_ARCHITECTURES
        assert (
            SUPPORTED_ARCHITECTURES["Qwen3_5MoeForConditionalGeneration"]
            is Qwen3_5MoeMultimodalArchitectureAdapter
        )
        assert SUPPORTED_ARCHITECTURES["Qwen3_5MoeForCausalLM"] is Qwen3_5MoeArchitectureAdapter

    def test_conditional_generation_loads_via_image_text_to_text(self):
        from transformers import AutoModelForImageTextToText

        from transformer_lens.model_bridge.sources.transformers import (
            get_hf_model_class_for_architecture,
        )

        model_class = get_hf_model_class_for_architecture("Qwen3_5MoeForConditionalGeneration")
        assert model_class is AutoModelForImageTextToText


class TestQwen3_5MoeLoadingGuards:
    def test_prepare_loading_swaps_top_level_config_for_text_config(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
            Qwen3_5MoeArchitectureAdapter,
        )

        adapter = Qwen3_5MoeArchitectureAdapter(_make_bridge_cfg())
        text_config = SimpleNamespace(model_type="qwen3_5_moe_text")
        full_config = SimpleNamespace(model_type="qwen3_5_moe", text_config=text_config)
        model_kwargs = {"config": full_config}

        adapter.prepare_loading("Qwen/Qwen3.5-35B-A3B", model_kwargs)

        assert model_kwargs["config"] is text_config

    def test_prepare_model_accepts_text_only_model(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
            Qwen3_5MoeArchitectureAdapter,
        )

        adapter = Qwen3_5MoeArchitectureAdapter(_make_bridge_cfg())
        hf_model = SimpleNamespace(config=SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"]))

        adapter.prepare_model(hf_model)

    def test_prepare_model_rejects_conditional_generation_model(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
            Qwen3_5MoeArchitectureAdapter,
        )

        adapter = Qwen3_5MoeArchitectureAdapter(_make_bridge_cfg())
        hf_model = SimpleNamespace(
            config=SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"])
        )

        with pytest.raises(ValueError, match="text-only"):
            adapter.prepare_model(hf_model)

    def test_prepare_model_rejects_unswapped_top_level_config(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
            Qwen3_5MoeArchitectureAdapter,
        )

        adapter = Qwen3_5MoeArchitectureAdapter(_make_bridge_cfg())
        hf_model = SimpleNamespace(
            config=SimpleNamespace(
                architectures=["Qwen3_5MoeForCausalLM"],
                text_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
            )
        )

        with pytest.raises(ValueError, match="text-only"):
            adapter.prepare_model(hf_model)


class TestQwen3_5MoeComponentMapping:
    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
            Qwen3_5MoeArchitectureAdapter,
        )

        return Qwen3_5MoeArchitectureAdapter(_make_bridge_cfg())

    def test_component_mapping_keys(self, adapter):
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_block_submodules_keys_include_hybrid_attention(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert set(submodules.keys()) == {"ln1", "ln2", "mlp", "attn", "linear_attn"}

    def test_mlp_is_moe_not_dense(self, adapter):
        """The single structural delta vs the dense Qwen3.5 adapter."""
        from transformer_lens.model_bridge.generalized_components import (
            GatedMLPBridge,
            MoEBridge,
            MoERouterBridge,
        )

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert not isinstance(mlp, GatedMLPBridge)
        assert set(mlp.submodules) == {"gate", "experts", "shared_expert", "shared_expert_gate"}
        assert isinstance(mlp.submodules["gate"], MoERouterBridge)

    def test_gated_q_proj_flag_set(self, adapter):
        assert getattr(adapter.cfg, "gated_q_proj", False) is True


class TestQwen3_5MoeMultimodalComponentMapping:
    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
            Qwen3_5MoeMultimodalArchitectureAdapter,
        )

        cfg = _make_bridge_cfg(architecture="Qwen3_5MoeForConditionalGeneration")
        return Qwen3_5MoeMultimodalArchitectureAdapter(cfg)

    def test_has_vision_components(self, adapter):
        assert "vision_encoder" in adapter.component_mapping
        assert "vision_projector" in adapter.component_mapping

    def test_language_model_paths_are_nested(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.language_model.embed_tokens"
        assert adapter.component_mapping["blocks"].name == "model.language_model.layers"

    def test_mlp_is_moe(self, adapter):
        from transformer_lens.model_bridge.generalized_components import MoEBridge

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)

    def test_is_multimodal_flag(self, adapter):
        assert adapter.cfg.is_multimodal is True


def _make_tiny_hf_model():
    """Tiny Qwen3_5MoeForCausalLM: 4 layers, full-attn at 3 (interval=4), 4 experts top-2."""
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    model = _Qwen3_5MoeForCausalLM(cfg)
    model.eval()
    return model


def _make_tiny_bridge():
    from unittest.mock import MagicMock

    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
        Qwen3_5MoeArchitectureAdapter,
    )

    hf_model = _make_tiny_hf_model()
    bridge_cfg = _make_bridge_cfg(
        d_model=64,
        d_head=16,
        n_heads=4,
        n_layers=4,
        d_vocab=512,
    )
    adapter = Qwen3_5MoeArchitectureAdapter(bridge_cfg)
    return TransformerBridge(hf_model, adapter, tokenizer=MagicMock()), hf_model


@pytest.mark.skipif(
    not _QWEN3_5_MOE_AVAILABLE,
    reason="Qwen3_5MoeTextConfig / Qwen3_5MoeForCausalLM not available in installed transformers",
)
class TestQwen3_5MoeIntegration:
    @pytest.fixture(scope="class")
    def bridge_and_model(self):
        return _make_tiny_bridge()

    @pytest.fixture(scope="class")
    def bridge(self, bridge_and_model):
        return bridge_and_model[0]

    @pytest.fixture(scope="class")
    def hf_model(self, bridge_and_model):
        return bridge_and_model[1]

    def test_forward_pass_consistency(self, bridge, hf_model):
        import torch

        tokens = torch.randint(0, 512, (1, 6))
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits
            bridge_logits = bridge(tokens)
        assert hf_logits.shape == bridge_logits.shape
        assert torch.allclose(
            hf_logits, bridge_logits, atol=1e-4
        ), f"Logit mismatch: max diff = {(hf_logits - bridge_logits).abs().max().item():.6f}"

    def test_moe_and_hybrid_attention_hooks_fire(self, bridge):
        """MoE MLP, linear-attention (layer 0), and full-attention (layer 3) hooks."""
        import torch

        hook_names = [
            "blocks.0.mlp.hook_out",
            "blocks.0.linear_attn.hook_out",
            "blocks.3.attn.hook_out",
        ]
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        tokens = torch.randint(0, 512, (1, 6))
        with torch.no_grad():
            bridge.run_with_hooks(
                tokens,
                use_cache=False,
                fwd_hooks=[(name, grab) for name in hook_names],
            )
        for name in hook_names:
            assert captured.get(name) == (1, 6, 64), f"{name}: {captured.get(name)}"

    def test_full_attention_layers_lack_linear_attn_hooks(self, bridge):
        hook_keys = set(bridge.hook_dict.keys())
        assert not any("blocks.3.linear_attn" in k for k in hook_keys)
        assert not any("blocks.0.attn." in k for k in hook_keys)
