"""Unit tests for the OpenAIGPTArchitectureAdapter (GPT-1).

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    JointQKVAttentionBridge,
    MLPBridge,
    NormalizationBridge,
)
from transformer_lens.model_bridge.supported_architectures.openai_gpt import (
    OpenAIGPTArchitectureAdapter,
    _OpenAIGPTJointQKVAttentionBridge,
)


def _make_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        architecture="OpenAIGPTLMHeadModel",
    )


@pytest.fixture(scope="class")
def adapter() -> OpenAIGPTArchitectureAdapter:
    return OpenAIGPTArchitectureAdapter(_make_cfg())


class TestOpenAIGPTAdapterConfig:
    def test_post_ln_processing_guards(self, adapter):
        assert adapter.cfg.normalization_type == "LN"
        assert adapter.cfg.positional_embedding_type == "standard"
        assert adapter.supports_fold_ln is False
        assert adapter.supports_center_writing_weights is False


class TestOpenAIGPTComponentMapping:
    def test_no_final_norm(self, adapter):
        """GPT-1 has no ln_f — the last block's ln_2 is the final norm."""
        assert "ln_final" not in adapter.component_mapping
        assert set(adapter.component_mapping) == {"embed", "pos_embed", "blocks", "unembed"}

    def test_gpt1_specific_embedding_paths(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "transformer.tokens_embed"
        assert mapping["pos_embed"].name == "transformer.positions_embed"

    def test_block_uses_gpt2_style_internals(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        attn = submodules["attn"]
        assert isinstance(attn, JointQKVAttentionBridge)
        assert attn.submodules["qkv"].name == "c_attn"
        assert attn.submodules["o"].name == "c_proj"
        mlp = submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.submodules["in"].name == "c_fc"
        assert mlp.submodules["out"].name == "c_proj"
        assert isinstance(submodules["ln1"], NormalizationBridge)

    def test_attention_returns_list_for_legacy_block(self, adapter):
        """GPT-1's Block does `[h] + attn_outputs[1:]` — tuples crash it."""
        import torch

        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, _OpenAIGPTJointQKVAttentionBridge)
        out = attn._process_output((torch.zeros(1, 2, 4), None))
        assert isinstance(out, list)


class TestOpenAIGPTRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["OpenAIGPTLMHeadModel"] is OpenAIGPTArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="openai-gpt", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "OpenAIGPTLMHeadModel"


class TestOpenAIGPTPostNormResidMid:
    """ln_2 sees the post-MLP sum, so hook_resid_mid must point at mlp.hook_in
    (the true attn->MLP mid-point n = ln_1(x+a))."""

    def test_hook_resid_mid_points_at_mlp_hook_in(self, adapter) -> None:
        block = adapter.component_mapping["blocks"]
        assert block.hook_aliases["hook_resid_mid"] == "mlp.hook_in"
