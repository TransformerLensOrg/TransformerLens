"""Normalization-rule mount classification read from adapter ``component_mapping``s for
Gemma 3n, Gemma 4 and StableLM, whose q/k/v or sandwich norms are placeholders."""

from typing import Any

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.model_bridge._relevance_rules import _RelevanceRuleCapable
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.gemma3n import (
    Gemma3nArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gemma4 import (
    Gemma4ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.stablelm import (
    StableLmArchitectureAdapter,
)


def _block_submodules(adapter: Any) -> dict:
    return dict(adapter.component_mapping["blocks"].submodules)


def _attn_submodules(block_submodules: dict, attn_key: str) -> dict:
    return dict(block_submodules[attn_key].submodules)


class TestGemma3nPlaceholders:
    def _adapter(self) -> Gemma3nArchitectureAdapter:
        cfg = make_bridge_cfg("Gemma3nForConditionalGeneration", n_key_value_heads=2, d_head=8)
        return Gemma3nArchitectureAdapter(cfg)

    def test_no_ln1_or_ln2_key_at_block_level(self):
        block_submodules = _block_submodules(self._adapter())
        assert "ln1" not in block_submodules
        assert "ln2" not in block_submodules

    def test_qkv_norm_placeholders_are_plain_and_rule_incapable(self):
        block_submodules = _block_submodules(self._adapter())
        attn_submodules = _attn_submodules(block_submodules, "self_attn")
        for key in ("q_norm", "k_norm", "v_norm"):
            component = attn_submodules[key]
            assert type(component) is GeneralizedComponent
            assert not isinstance(component, NormalizationBridge)
            assert not isinstance(component, _RelevanceRuleCapable)


class TestGemma4Placeholders:
    def _adapter(self) -> Gemma4ArchitectureAdapter:
        from types import SimpleNamespace

        cfg = make_bridge_cfg("Gemma4ForConditionalGeneration", n_key_value_heads=1, d_head=8)
        cfg.vision_config = SimpleNamespace(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=4
        )
        cfg.vision_soft_tokens_per_image = 4
        return Gemma4ArchitectureAdapter(cfg)

    def test_ln1_ln2_are_plain_sandwich_placeholders_not_normalization_bridge(self):
        block_submodules = _block_submodules(self._adapter())
        for key in ("ln1", "ln2"):
            component = block_submodules[key]
            assert type(component) is GeneralizedComponent
            assert not isinstance(component, NormalizationBridge)
            assert not isinstance(component, _RelevanceRuleCapable)

    def test_qkv_norm_placeholders_are_not_keyed_ln1_or_ln2(self):
        block_submodules = _block_submodules(self._adapter())
        attn_submodules = _attn_submodules(block_submodules, "attn")
        for key in ("q_norm", "k_norm", "v_norm"):
            assert key not in ("ln1", "ln2")
            component = attn_submodules[key]
            assert type(component) is GeneralizedComponent
            assert not isinstance(component, _RelevanceRuleCapable)


class TestStableLmPlaceholders:
    def _adapter(self) -> StableLmArchitectureAdapter:
        cfg = make_bridge_cfg("StableLmForCausalLM", n_key_value_heads=2, d_head=8)
        return StableLmArchitectureAdapter(cfg)

    def test_ln1_and_ln2_are_genuine_rule_targets(self):
        """The positive case: unlike Gemma 3n/4, StableLM's block norms are rule-capable."""
        block_submodules = _block_submodules(self._adapter())
        for key in ("ln1", "ln2"):
            component = block_submodules[key]
            assert isinstance(component, NormalizationBridge)
            assert isinstance(component, _RelevanceRuleCapable)
            assert component._relevance_rule_kinds == ("normalization",)

    def test_per_head_norms_are_keyed_q_norm_k_norm_not_ln1_ln2(self):
        """The canonical-mount check reads the mount key (q_norm), not the HF name (q_layernorm)."""
        block_submodules = _block_submodules(self._adapter())
        attn_submodules = _attn_submodules(block_submodules, "attn")
        assert attn_submodules["q_norm"].name == "q_layernorm"
        assert attn_submodules["k_norm"].name == "k_layernorm"
        for key in ("q_norm", "k_norm"):
            component = attn_submodules[key]
            assert key not in ("ln1", "ln2")
            assert type(component) is GeneralizedComponent
            assert not isinstance(component, _RelevanceRuleCapable)
