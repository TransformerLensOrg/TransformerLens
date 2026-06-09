"""Unit tests for SGLang overlays — capture-spec shape and gating."""
from __future__ import annotations

from types import SimpleNamespace

from transformer_lens.model_bridge.sources.sglang.overlays import (
    DEFAULT_SGLANG_OVERLAY,
    DecoderOnlyOverlay,
    DeepseekOverlay,
    get_overlay,
)


def _hf_config(n_layers: int = 4, hidden: int = 64) -> SimpleNamespace:
    return SimpleNamespace(hidden_size=hidden, num_hidden_layers=n_layers)


class TestDecoderOnlyOverlay:
    """Llama-family captures: 2 module-level + 3 per-layer hooks."""

    def test_spec_shape(self):
        specs = DecoderOnlyOverlay().capture_specs(_hf_config(n_layers=2, hidden=16))
        # 2 module-level (embed, ln_final) + 3 per-layer * 2 layers = 8 entries.
        assert len(specs) == 2 + 3 * 2
        assert specs["embed.hook_out"] == ("model.embed_tokens", 16)
        assert specs["ln_final.hook_normalized"] == ("model.norm", 16)
        assert specs["blocks.0.hook_out"] == ("model.layers.0", 16)
        assert specs["blocks.0.attn.hook_out"] == ("model.layers.0.self_attn", 16)
        assert specs["blocks.0.mlp.hook_out"] == ("model.layers.0.mlp", 16)

    def test_nonfiring_includes_pattern_and_unembed(self):
        gated = DecoderOnlyOverlay().nonfiring_hooks()
        assert "blocks.{i}.attn.hook_pattern" in gated
        assert "blocks.{i}.attn.hook_attn_scores" in gated
        assert "unembed.hook_out" in gated


class TestDeepseekOverlay:
    """DeepSeek (MLA) shares the dotted paths but adds MLA-specific gated hooks."""

    def test_same_capture_paths_as_decoder_only(self):
        # The per-layer dotted paths are identical; MLA's internals (q_a_proj /
        # kv_a_proj / latent decompression) live below the self_attn boundary.
        decoder = DecoderOnlyOverlay().capture_specs(_hf_config(2, 16))
        deepseek = DeepseekOverlay().capture_specs(_hf_config(2, 16))
        assert decoder == deepseek

    def test_nonfiring_adds_mla_and_moe_entries(self):
        gated = DeepseekOverlay().nonfiring_hooks()
        assert "blocks.{i}.attn.hook_latent_kv" in gated
        assert "blocks.{i}.mlp.hook_expert_out" in gated


class TestRegistry:
    def test_decoder_only_is_default(self):
        assert get_overlay("LlamaForCausalLM") is DEFAULT_SGLANG_OVERLAY
        assert get_overlay("Qwen3ForCausalLM") is DEFAULT_SGLANG_OVERLAY
        assert get_overlay("MistralForCausalLM") is DEFAULT_SGLANG_OVERLAY

    def test_deepseek_routes_to_mla_overlay(self):
        assert isinstance(get_overlay("DeepseekV2ForCausalLM"), DeepseekOverlay)
        assert isinstance(get_overlay("DeepseekV3ForCausalLM"), DeepseekOverlay)
        assert isinstance(get_overlay("DeepseekV4ForCausalLM"), DeepseekOverlay)
