"""Tests for convert_hf_model_config covering every architecture branch.

These tests serve as regression tests so that refactoring the giant if/else
in convert_hf_model_config can be done with confidence.

For hardcoded branches (Llama, Gemma), no mocking is needed.
For architecture-based branches, we mock AutoConfig.from_pretrained.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from transformer_lens.loading_from_pretrained import convert_hf_model_config

# ============================================================================
# Helpers
# ============================================================================


def _mock_auto_config(hf_config):
    """Return a context manager that mocks AutoConfig.from_pretrained to return hf_config."""
    return mock.patch(
        "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
        return_value=hf_config,
    )


def _assert_cfg_matches(cfg_dict, expected):
    """Assert all keys in expected are present in cfg_dict with matching values."""
    for key, value in expected.items():
        assert key in cfg_dict, f"Missing key: {key}"
        assert (
            cfg_dict[key] == value
        ), f"Mismatch for {key}: expected {value!r}, got {cfg_dict[key]!r}"


# ============================================================================
# Llama hardcoded configs
# ============================================================================

LLAMA_COMMON = {
    "act_fn": "silu",
    "normalization_type": "RMS",
    "positional_embedding_type": "rotary",
    "rotary_adjacent_pairs": False,
    "final_rms": True,
    "gated_mlp": True,
    "original_architecture": "LlamaForCausalLM",
}


class TestLlamaHardcodedConfigs:
    """Test all hardcoded Llama model config branches."""

    def test_llama_7b(self):
        cfg = convert_hf_model_config("llama-7b-hf")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 11008,
                "n_layers": 32,
                "n_ctx": 2048,
                "eps": 1e-6,
                "d_vocab": 32000,
                "rotary_dim": 128,
            },
        )

    def test_llama_2_7b(self):
        cfg = convert_hf_model_config("meta-llama/Llama-2-7b-hf")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 11008,
                "n_layers": 32,
                "n_ctx": 4096,
                "eps": 1e-5,
                "d_vocab": 32000,
                "rotary_dim": 128,
            },
        )

    def test_codellama_7b(self):
        cfg = convert_hf_model_config("codellama/CodeLlama-7b-hf")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 11008,
                "n_layers": 32,
                "n_ctx": 4096,
                "eps": 1e-5,
                "d_vocab": 32016,
                "rotary_dim": 128,
                "rotary_base": 1000000,
            },
        )

    def test_codellama_python_vocab(self):
        cfg = convert_hf_model_config("codellama/CodeLlama-7b-Python-hf")
        assert cfg["d_vocab"] == 32000

    def test_llama_13b(self):
        cfg = convert_hf_model_config("llama-13b-hf")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 5120,
                "d_head": 128,
                "n_heads": 40,
                "d_mlp": 13824,
                "n_layers": 40,
                "n_ctx": 2048,
                "eps": 1e-6,
                "d_vocab": 32000,
                "rotary_dim": 128,
            },
        )

    def test_llama_2_13b(self):
        cfg = convert_hf_model_config("meta-llama/Llama-2-13b-hf")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 5120,
                "d_head": 128,
                "n_heads": 40,
                "d_mlp": 13824,
                "n_layers": 40,
                "n_ctx": 4096,
                "eps": 1e-5,
                "d_vocab": 32000,
                "rotary_dim": 128,
            },
        )

    def test_llama_30b(self):
        cfg = convert_hf_model_config("llama-30b-hf")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 6656,
                "d_head": 128,
                "n_heads": 52,
                "d_mlp": 17920,
                "n_layers": 60,
                "n_ctx": 2048,
                "eps": 1e-6,
                "d_vocab": 32000,
                "rotary_dim": 128,
            },
        )

    def test_llama_65b(self):
        cfg = convert_hf_model_config("llama-65b-hf")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 8192,
                "d_head": 128,
                "n_heads": 64,
                "d_mlp": 22016,
                "n_layers": 80,
                "n_ctx": 2048,
                "eps": 1e-6,
                "d_vocab": 32000,
                "rotary_dim": 128,
            },
        )

    def test_llama_2_70b(self):
        cfg = convert_hf_model_config("meta-llama/Llama-2-70b-chat-hf")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 8192,
                "d_head": 128,
                "n_heads": 64,
                "d_mlp": 28672,
                "n_layers": 80,
                "n_ctx": 4096,
                "eps": 1e-5,
                "d_vocab": 32000,
                "n_key_value_heads": 8,
                "rotary_dim": 128,
            },
        )

    def test_meta_llama_3_8b(self):
        cfg = convert_hf_model_config("meta-llama/Meta-Llama-3-8B")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 14336,
                "n_layers": 32,
                "n_ctx": 8192,
                "eps": 1e-5,
                "d_vocab": 128256,
                "n_key_value_heads": 8,
                "rotary_dim": 128,
                "rotary_base": 500000.0,
            },
        )

    def test_meta_llama_3_70b(self):
        cfg = convert_hf_model_config("meta-llama/Meta-Llama-3-70B")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 8192,
                "d_head": 128,
                "n_heads": 64,
                "d_mlp": 28672,
                "n_layers": 80,
                "n_ctx": 8192,
                "eps": 1e-5,
                "d_vocab": 128256,
                "n_key_value_heads": 8,
                "rotary_dim": 128,
                "rotary_base": 500000.0,
            },
        )

    def test_llama_3_2_1b(self):
        cfg = convert_hf_model_config("meta-llama/Llama-3.2-1B")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 2048,
                "d_head": 64,
                "n_heads": 32,
                "d_mlp": 8192,
                "n_layers": 16,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 128256,
                "n_key_value_heads": 8,
                "rotary_dim": 64,
                "rotary_base": 500000.0,
                "use_NTK_by_parts_rope": True,
                "NTK_by_parts_low_freq_factor": 1.0,
                "NTK_by_parts_high_freq_factor": 4.0,
                "NTK_by_parts_factor": 32.0,
                "NTK_original_ctx_len": 8192,
            },
        )

    def test_llama_3_2_3b(self):
        cfg = convert_hf_model_config("meta-llama/Llama-3.2-3B")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 3072,
                "d_head": 128,
                "n_heads": 24,
                "d_mlp": 8192,
                "n_layers": 28,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 128256,
                "n_key_value_heads": 8,
                "rotary_dim": 128,
                "rotary_base": 500000.0,
                "use_NTK_by_parts_rope": True,
                "NTK_by_parts_low_freq_factor": 1.0,
                "NTK_by_parts_high_freq_factor": 4.0,
                "NTK_by_parts_factor": 32.0,
                "NTK_original_ctx_len": 8192,
            },
        )

    def test_llama_3_3_70b(self):
        cfg = convert_hf_model_config("meta-llama/Llama-3.3-70B-Instruct")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 8192,
                "d_head": 128,
                "n_heads": 64,
                "d_mlp": 28672,
                "n_layers": 80,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 128256,
                "n_key_value_heads": 8,
                "rotary_dim": 128,
                "rotary_base": 500000.0,
                "use_NTK_by_parts_rope": True,
                "NTK_by_parts_low_freq_factor": 1.0,
                "NTK_by_parts_high_freq_factor": 4.0,
                "NTK_by_parts_factor": 8.0,
                "NTK_original_ctx_len": 8192,
            },
        )

    def test_llama_3_1_8b(self):
        cfg = convert_hf_model_config("meta-llama/Llama-3.1-8B")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 14336,
                "n_layers": 32,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 128256,
                "n_key_value_heads": 8,
                "rotary_dim": 128,
                "rotary_base": 500000.0,
                "use_NTK_by_parts_rope": True,
                "NTK_by_parts_low_freq_factor": 1.0,
                "NTK_by_parts_high_freq_factor": 4.0,
                "NTK_by_parts_factor": 8.0,
                "NTK_original_ctx_len": 8192,
            },
        )

    def test_llama_3_1_70b(self):
        cfg = convert_hf_model_config("meta-llama/Llama-3.1-70B")
        _assert_cfg_matches(
            cfg,
            {
                **LLAMA_COMMON,
                "d_model": 8192,
                "d_head": 128,
                "n_heads": 64,
                "d_mlp": 28672,
                "n_layers": 80,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 128256,
                "n_key_value_heads": 8,
                "rotary_dim": 128,
                "rotary_base": 500000.0,
                "use_NTK_by_parts_rope": True,
                "NTK_by_parts_low_freq_factor": 1.0,
                "NTK_by_parts_high_freq_factor": 4.0,
                "NTK_by_parts_factor": 8.0,
                "NTK_original_ctx_len": 8192,
            },
        )


# ============================================================================
# Gemma hardcoded configs
# ============================================================================


class TestGemmaHardcodedConfigs:
    """Test hardcoded Gemma 1 and Gemma 2 config branches.

    Gemma 3 configs are covered by test_gemma3_config.py.
    """

    def test_gemma_2b(self):
        cfg = convert_hf_model_config("google/gemma-2b")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 2048,
                "d_head": 256,
                "n_heads": 8,
                "d_mlp": 16384,
                "n_layers": 18,
                "n_ctx": 8192,
                "eps": 1e-06,
                "d_vocab": 256000,
                "act_fn": "gelu_new",
                "normalization_type": "RMS",
                "rotary_base": 10000,
                "rotary_dim": 256,
                "positional_embedding_type": "rotary",
                "use_attn_scale": True,
                "n_key_value_heads": 1,
                "gated_mlp": True,
                "final_rms": True,
                # NOTE: "google/gemma-2b" contains "gemma-2" so the architecture
                # detection assigns "Gemma2ForCausalLM" even though this is a Gemma 1 model.
                # The config values are still correct (matched by model name prefix).
                "original_architecture": "Gemma2ForCausalLM",
            },
        )

    def test_gemma_7b(self):
        cfg = convert_hf_model_config("google/gemma-7b")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 3072,
                "d_head": 256,
                "n_heads": 16,
                "d_mlp": 24576,
                "n_layers": 28,
                "n_ctx": 8192,
                "eps": 1e-06,
                "d_vocab": 256000,
                "act_fn": "gelu_new",
                "normalization_type": "RMS",
                "rotary_base": 10000.0,
                "rotary_dim": 256,
                "positional_embedding_type": "rotary",
                "use_attn_scale": True,
                "n_key_value_heads": 16,
                "gated_mlp": True,
                "final_rms": True,
                "original_architecture": "GemmaForCausalLM",
            },
        )

    def test_gemma_2_2b(self):
        cfg = convert_hf_model_config("google/gemma-2-2b")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 2304,
                "d_head": 256,
                "n_heads": 8,
                "d_mlp": 9216,
                "n_layers": 26,
                "n_ctx": 8192,
                "eps": 1e-06,
                "d_vocab": 256000,
                "act_fn": "gelu_pytorch_tanh",
                "normalization_type": "RMS",
                "rotary_base": 10000.0,
                "positional_embedding_type": "rotary",
                "use_attn_scale": True,
                "n_key_value_heads": 4,
                "window_size": 4096,
                "use_local_attn": True,
                "attn_types": ["global", "local"] * 21,
                "attn_scores_soft_cap": 50.0,
                "output_logits_soft_cap": 30.0,
                "gated_mlp": True,
                "final_rms": True,
                "use_normalization_before_and_after": True,
                "original_architecture": "Gemma2ForCausalLM",
            },
        )

    def test_gemma_2_9b(self):
        cfg = convert_hf_model_config("google/gemma-2-9b")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 3584,
                "d_head": 256,
                "n_heads": 16,
                "d_mlp": 14336,
                "n_layers": 42,
                "n_ctx": 8192,
                "eps": 1e-06,
                "d_vocab": 256000,
                "act_fn": "gelu_pytorch_tanh",
                "normalization_type": "RMS",
                "rotary_base": 10000.0,
                "positional_embedding_type": "rotary",
                "use_attn_scale": True,
                "n_key_value_heads": 8,
                "window_size": 4096,
                "use_local_attn": True,
                "attn_types": ["global", "local"] * 21,
                "attn_scores_soft_cap": 50.0,
                "output_logits_soft_cap": 30.0,
                "gated_mlp": True,
                "final_rms": True,
                "use_normalization_before_and_after": True,
                "original_architecture": "Gemma2ForCausalLM",
            },
        )

    def test_gemma_2_27b(self):
        cfg = convert_hf_model_config("google/gemma-2-27b")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 4608,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 36864,
                "n_layers": 46,
                "n_ctx": 8192,
                "eps": 1e-06,
                "d_vocab": 256000,
                "act_fn": "gelu_pytorch_tanh",
                "normalization_type": "RMS",
                "rotary_base": 10000.0,
                "positional_embedding_type": "rotary",
                "use_attn_scale": True,
                "attn_scale": 12.0,
                "n_key_value_heads": 16,
                "window_size": 4096,
                "use_local_attn": True,
                "attn_types": ["global", "local"] * 23,
                "attn_scores_soft_cap": 50.0,
                "output_logits_soft_cap": 30.0,
                "gated_mlp": True,
                "final_rms": True,
                "use_normalization_before_and_after": True,
                "original_architecture": "Gemma2ForCausalLM",
            },
        )


# ============================================================================
# Architecture-based configs (require mocked AutoConfig)
# ============================================================================


class TestGPTNeoConfig:
    def test_gpt_neo(self):
        hf_config = SimpleNamespace(
            architectures=["GPTNeoForCausalLM"],
            hidden_size=768,
            num_heads=12,
            num_layers=12,
            max_position_embeddings=2048,
            layer_norm_epsilon=1e-5,
            vocab_size=50257,
            attention_layers=["global", "local"] * 6,
            activation_function="gelu_new",
            window_size=256,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("EleutherAI/gpt-neo-125M")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 768,
                "d_head": 64,
                "n_heads": 12,
                "d_mlp": 3072,
                "n_layers": 12,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 50257,
                "attn_types": ["global", "local"] * 6,
                "act_fn": "gelu_new",
                "use_attn_scale": False,
                "use_local_attn": True,
                "window_size": 256,
                "scale_attn_by_inverse_layer_idx": False,
                "normalization_type": "LN",
                "original_architecture": "GPTNeoForCausalLM",
            },
        )


class TestGPT2Config:
    def test_gpt2(self):
        hf_config = SimpleNamespace(
            architectures=["GPT2LMHeadModel"],
            n_embd=768,
            n_head=12,
            n_layer=12,
            n_ctx=1024,
            layer_norm_epsilon=1e-5,
            vocab_size=50257,
            activation_function="gelu_new",
            scale_attn_by_inverse_layer_idx=False,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("gpt2")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 768,
                "d_head": 64,
                "n_heads": 12,
                "d_mlp": 3072,
                "n_layers": 12,
                "n_ctx": 1024,
                "eps": 1e-5,
                "d_vocab": 50257,
                "act_fn": "gelu_new",
                "use_attn_scale": True,
                "use_local_attn": False,
                "scale_attn_by_inverse_layer_idx": False,
                "normalization_type": "LN",
                "original_architecture": "GPT2LMHeadModel",
            },
        )


class TestOPTConfig:
    def test_opt(self):
        hf_config = SimpleNamespace(
            architectures=["OPTForCausalLM"],
            hidden_size=768,
            num_attention_heads=12,
            ffn_dim=3072,
            num_hidden_layers=12,
            max_position_embeddings=2048,
            vocab_size=50272,
            activation_function="relu",
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("facebook/opt-125m")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 768,
                "d_head": 64,
                "n_heads": 12,
                "d_mlp": 3072,
                "n_layers": 12,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 50272,
                "act_fn": "relu",
                "use_attn_scale": True,
                "use_local_attn": False,
                "scale_attn_by_inverse_layer_idx": False,
                "normalization_type": "LN",
                "original_architecture": "OPTForCausalLM",
            },
        )


class TestGPTJConfig:
    def test_gptj(self):
        hf_config = SimpleNamespace(
            architectures=["GPTJForCausalLM"],
            n_embd=4096,
            n_head=16,
            n_layer=28,
            n_positions=2048,
            vocab_size=50400,
            activation_function="gelu_new",
            rotary_dim=64,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("EleutherAI/gpt-j-6B")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 4096,
                "d_head": 256,
                "n_heads": 16,
                "d_mlp": 16384,
                "n_layers": 28,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 50400,
                "act_fn": "gelu_new",
                "use_attn_scale": True,
                "use_local_attn": False,
                "scale_attn_by_inverse_layer_idx": False,
                "parallel_attn_mlp": True,
                "positional_embedding_type": "rotary",
                "rotary_dim": 64,
                "rotary_adjacent_pairs": True,
                "normalization_type": "LN",
                "original_architecture": "GPTJForCausalLM",
            },
        )


class TestGPTNeoXConfig:
    def test_gpt_neox(self):
        hf_config = SimpleNamespace(
            architectures=["GPTNeoXForCausalLM"],
            hidden_size=6144,
            num_attention_heads=64,
            intermediate_size=24576,
            num_hidden_layers=44,
            max_position_embeddings=2048,
            layer_norm_eps=1e-5,
            vocab_size=50432,
            hidden_act="gelu",
            rotary_pct=0.25,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("EleutherAI/gpt-neox-20b")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 6144,
                "d_head": 96,
                "n_heads": 64,
                "d_mlp": 24576,
                "n_layers": 44,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 50432,
                "act_fn": "gelu",
                "use_attn_scale": True,
                "use_local_attn": False,
                "scale_attn_by_inverse_layer_idx": False,
                "parallel_attn_mlp": True,
                "positional_embedding_type": "rotary",
                "rotary_adjacent_pairs": False,
                "rotary_dim": 24,  # round(0.25 * 96)
                "normalization_type": "LN",
                "original_architecture": "GPTNeoXForCausalLM",
            },
        )


class TestBloomConfig:
    def test_bloom(self):
        hf_config = SimpleNamespace(
            architectures=["BloomForCausalLM"],
            hidden_size=1024,
            n_head=16,
            n_layer=24,
            vocab_size=250880,
            layer_norm_epsilon=1e-5,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("bigscience/bloom-560m")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 1024,
                "d_head": 64,
                "n_heads": 16,
                "d_mlp": 4096,
                "n_layers": 24,
                "n_ctx": 2048,
                "d_vocab": 250880,
                "act_fn": "gelu_fast",
                "eps": 1e-5,
                "normalization_type": "LN",
                "post_embedding_ln": True,
                "positional_embedding_type": "alibi",
                "default_prepend_bos": False,
                "original_architecture": "BloomForCausalLM",
            },
        )


class TestMistralConfig:
    def test_mistral(self):
        hf_config = SimpleNamespace(
            architectures=["MistralForCausalLM"],
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=14336,
            num_hidden_layers=32,
            vocab_size=32000,
            hidden_act="silu",
            sliding_window=4096,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            num_key_value_heads=8,
            head_dim=128,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("mistralai/Mistral-7B-v0.1")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 14336,
                "n_layers": 32,
                "n_ctx": 2048,  # Capped
                "d_vocab": 32000,
                "act_fn": "silu",
                "window_size": 4096,
                "eps": 1e-5,
                "rotary_base": 10000.0,
                "n_key_value_heads": 8,
                "use_local_attn": True,
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "gated_mlp": True,
                "original_architecture": "MistralForCausalLM",
            },
        )


class TestMixtralConfig:
    def test_mixtral(self):
        hf_config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=14336,
            num_hidden_layers=32,
            max_position_embeddings=32768,
            vocab_size=32000,
            hidden_act="silu",
            sliding_window=None,
            rms_norm_eps=1e-5,
            rope_theta=1000000.0,
            num_key_value_heads=8,
            num_local_experts=8,
            num_experts_per_tok=2,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("mistralai/Mixtral-8x7B-v0.1")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 14336,
                "n_layers": 32,
                "n_ctx": 32768,
                "d_vocab": 32000,
                "act_fn": "silu",
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "rotary_base": 1000000.0,
                "window_size": None,
                "attn_types": ["global"] * 32,
                "eps": 1e-5,
                "n_key_value_heads": 8,
                "gated_mlp": True,
                "use_local_attn": False,
                "rotary_dim": 128,
                "num_experts": 8,
                "experts_per_token": 2,
                "original_architecture": "MixtralForCausalLM",
            },
        )


class TestSantacoderConfig:
    def test_santacoder(self):
        hf_config = SimpleNamespace(
            architectures=["GPT2LMHeadCustomModel"],
            n_embd=2048,
            n_head=16,
            n_layer=24,
            n_positions=2048,
            layer_norm_epsilon=1e-5,
            vocab_size=49280,
            activation_function="gelu",
            scale_attn_by_inverse_layer_idx=False,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("bigcode/santacoder")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 2048,
                "d_head": 128,
                "n_heads": 16,
                "d_mlp": 8192,
                "n_layers": 24,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 49280,
                "act_fn": "gelu",
                "use_attn_scale": True,
                "use_local_attn": False,
                "trust_remote_code": True,
                "scale_attn_by_inverse_layer_idx": False,
                "normalization_type": "LN",
                "original_architecture": "GPT2LMHeadCustomModel",
            },
        )


class TestGenericLlamaConfig:
    """Test the generic LlamaForCausalLM handler (for models not matching hardcoded names)."""

    def test_generic_llama_via_hf_config(self):
        """Test a hypothetical Llama model that doesn't match any hardcoded name.

        We mock AutoConfig and use a model name containing 'llama' so it gets
        architecture='LlamaForCausalLM' from the name check, but we need a name
        that doesn't match any hardcoded startswith/in checks.

        Since ALL names containing 'llama' get architecture from the name check
        (line 843) and skip AutoConfig, we can't easily test the generic handler
        in isolation without a model that goes through the else branch.
        Instead, we test a Yi model which uses LlamaForCausalLM architecture.
        """
        hf_config = SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=11008,
            num_hidden_layers=32,
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            vocab_size=64000,
            hidden_act="silu",
            num_key_value_heads=4,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("01-ai/Yi-6B")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "d_mlp": 11008,
                "n_layers": 32,
                "n_ctx": 4096,
                "eps": 1e-6,
                "d_vocab": 64000,
                "act_fn": "silu",
                "n_key_value_heads": 4,
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "rotary_adjacent_pairs": False,
                "rotary_dim": 128,
                "final_rms": True,
                "gated_mlp": True,
                "original_architecture": "LlamaForCausalLM",
            },
        )

    def test_generic_llama_mha_no_gqa(self):
        """When num_key_value_heads == num_attention_heads, n_key_value_heads should be None."""
        hf_config = SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=11008,
            num_hidden_layers=32,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            hidden_act="silu",
            num_key_value_heads=32,  # Same as num_attention_heads
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("01-ai/Yi-6B")
        assert cfg["n_key_value_heads"] is None


class TestQwenConfig:
    def test_qwen(self):
        hf_config = SimpleNamespace(
            architectures=["QWenLMHeadModel"],
            hidden_size=2048,
            num_attention_heads=16,
            intermediate_size=11008,
            num_hidden_layers=24,
            layer_norm_epsilon=1e-6,
            vocab_size=151936,
            scale_attn_weights=True,
            initializer_range=0.02,
            kv_channels=128,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("Qwen/Qwen-1_8B")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 2048,
                "d_head": 128,
                "n_heads": 16,
                "d_mlp": 5504,  # intermediate_size // 2
                "n_layers": 24,
                "n_ctx": 2048,  # Capped
                "eps": 1e-6,
                "d_vocab": 151936,
                "act_fn": "silu",
                "use_attn_scale": True,
                "initializer_range": 0.02,
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "rotary_dim": 128,
                "rotary_adjacent_pairs": False,
                "tokenizer_prepends_bos": True,
                "trust_remote_code": True,
                "final_rms": True,
                "gated_mlp": True,
                "default_prepend_bos": False,
                "original_architecture": "QWenLMHeadModel",
            },
        )


class TestQwen2Config:
    def test_qwen2(self):
        hf_config = SimpleNamespace(
            architectures=["Qwen2ForCausalLM"],
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            intermediate_size=4864,
            num_hidden_layers=24,
            rms_norm_eps=1e-6,
            vocab_size=151936,
            hidden_act="silu",
            initializer_range=0.02,
            rope_theta=1000000.0,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("Qwen/Qwen2-0.5B")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 896,
                "d_head": 64,
                "n_heads": 14,
                "n_key_value_heads": 2,
                "d_mlp": 4864,
                "n_layers": 24,
                "n_ctx": 2048,  # Capped
                "eps": 1e-6,
                "d_vocab": 151936,
                "act_fn": "silu",
                "use_attn_scale": True,
                "initializer_range": 0.02,
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "rotary_base": 1000000,
                "rotary_adjacent_pairs": False,
                "rotary_dim": 64,
                "tokenizer_prepends_bos": True,
                "final_rms": True,
                "gated_mlp": True,
                "default_prepend_bos": False,
                "original_architecture": "Qwen2ForCausalLM",
            },
        )


class TestQwen3Config:
    def test_qwen3(self):
        hf_config = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,
            intermediate_size=3072,
            num_hidden_layers=28,
            rms_norm_eps=1e-6,
            vocab_size=151936,
            hidden_act="silu",
            initializer_range=0.02,
            rope_theta=1000000.0,
            head_dim=64,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("Qwen/Qwen3-0.6B")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 1024,
                "d_head": 64,
                "n_heads": 16,
                "n_key_value_heads": 8,
                "d_mlp": 3072,
                "n_layers": 28,
                "n_ctx": 2048,
                "eps": 1e-6,
                "d_vocab": 151936,
                "act_fn": "silu",
                "use_attn_scale": True,
                "initializer_range": 0.02,
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "rotary_base": 1000000,
                "rotary_adjacent_pairs": False,
                "rotary_dim": 64,
                "tokenizer_prepends_bos": True,
                "final_rms": True,
                "gated_mlp": True,
                "default_prepend_bos": False,
                "use_qk_norm": True,
                "trust_remote_code": True,
                "original_architecture": "Qwen3ForCausalLM",
            },
        )

    def test_qwen3_mha_no_gqa(self):
        """When num_key_value_heads == num_attention_heads, n_key_value_heads should be None."""
        hf_config = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=3072,
            num_hidden_layers=28,
            rms_norm_eps=1e-6,
            vocab_size=151936,
            hidden_act="silu",
            initializer_range=0.02,
            rope_theta=1000000.0,
            head_dim=64,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("Qwen/Qwen3-0.6B")
        assert cfg["n_key_value_heads"] is None


class TestPhiConfig:
    def test_phi(self):
        hf_config = SimpleNamespace(
            architectures=["PhiForCausalLM"],
            hidden_size=2048,
            num_attention_heads=32,
            intermediate_size=8192,
            num_hidden_layers=24,
            max_position_embeddings=2048,
            layer_norm_eps=1e-5,
            vocab_size=51200,
            hidden_act="gelu_new",
            initializer_range=0.02,
            rope_theta=10000.0,
            partial_rotary_factor=0.4,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("microsoft/phi-1")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 2048,
                "d_head": 64,
                "n_heads": 32,
                "d_mlp": 8192,
                "n_layers": 24,
                "n_ctx": 2048,
                "eps": 1e-5,
                "d_vocab": 51200,
                "act_fn": "gelu_new",
                "initializer_range": 0.02,
                "normalization_type": "LN",
                "positional_embedding_type": "rotary",
                "trust_remote_code": True,
                "rotary_base": 10000.0,
                "use_attn_scale": True,
                "parallel_attn_mlp": True,
                "rotary_dim": 26,  # round(0.4 * 64)
                "original_architecture": "PhiForCausalLM",
            },
        )


class TestPhi3Config:
    def test_phi3(self):
        hf_config = SimpleNamespace(
            architectures=["Phi3ForCausalLM"],
            hidden_size=3072,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=8192,
            num_hidden_layers=32,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            vocab_size=32064,
            hidden_act="silu",
            initializer_range=0.02,
            rope_theta=10000.0,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("microsoft/Phi-3-mini-4k-instruct")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 3072,
                "d_head": 96,
                "n_heads": 32,
                "n_key_value_heads": None,  # 32 == 32, so None
                "d_mlp": 8192,
                "n_layers": 32,
                "n_ctx": 4096,
                "eps": 1e-5,
                "d_vocab": 32064,
                "act_fn": "silu",
                "initializer_range": 0.02,
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "trust_remote_code": True,
                "rotary_base": 10000.0,
                "use_attn_scale": True,
                "gated_mlp": True,
                "parallel_attn_mlp": False,
                "rotary_dim": 96,
                "original_architecture": "Phi3ForCausalLM",
            },
        )


class TestBertConfig:
    def test_bert(self):
        hf_config = SimpleNamespace(
            architectures=["BertForMaskedLM"],
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            num_hidden_layers=12,
            max_position_embeddings=512,
            layer_norm_eps=1e-12,
            vocab_size=28996,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("google-bert/bert-base-cased")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 768,
                "d_head": 64,
                "n_heads": 12,
                "d_mlp": 3072,
                "n_layers": 12,
                "n_ctx": 512,
                "eps": 1e-12,
                "d_vocab": 28996,
                "act_fn": "gelu",
                "attention_dir": "bidirectional",
                "original_architecture": "BertForMaskedLM",
            },
        )


class TestHubertConfig:
    def test_hubert(self):
        hf_config = SimpleNamespace(
            architectures=["HubertModel"],
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            num_hidden_layers=12,
            layer_norm_eps=1e-5,
            hidden_act="gelu",
            max_position_embeddings=8192,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("facebook/hubert-base-ls960")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 768,
                "d_head": 64,
                "n_heads": 12,
                "d_mlp": 3072,
                "n_layers": 12,
                "n_ctx": 8192,
                "eps": 1e-5,
                "act_fn": "gelu",
                "attention_dir": "bidirectional",
                "d_vocab": -1,
                "original_architecture": "HubertModel",
            },
        )


class TestWav2Vec2Config:
    def test_wav2vec2_base(self):
        hf_config = SimpleNamespace(
            architectures=["Wav2Vec2Model"],
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            num_hidden_layers=12,
            layer_norm_eps=1e-5,
            hidden_act="gelu",
            max_position_embeddings=8192,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("facebook/wav2vec2-base")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 768,
                "d_head": 64,
                "n_heads": 12,
                "d_mlp": 3072,
                "n_layers": 12,
                "n_ctx": 8192,
                "eps": 1e-5,
                "act_fn": "gelu",
                "attention_dir": "bidirectional",
                "d_vocab": -1,
                "original_architecture": "Wav2Vec2Model",
            },
        )


class TestT5Config:
    def test_t5(self):
        hf_config = SimpleNamespace(
            architectures=["T5ForConditionalGeneration"],
            d_model=512,
            d_kv=64,
            num_heads=8,
            d_ff=2048,
            vocab_size=32128,
            num_layers=6,
            max_length=512,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="relu",
            relative_attention_max_distance=128,
            relative_attention_num_buckets=32,
            decoder_start_token_id=0,
            tie_word_embeddings=True,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("google-t5/t5-small")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 512,
                "d_head": 64,
                "n_heads": 8,
                "d_mlp": 2048,
                "d_vocab": 32128,
                "n_layers": 6,
                "n_ctx": 512,
                "eps": 1e-6,
                "act_fn": "relu",
                "positional_embedding_type": "relative_positional_bias",
                "relative_attention_max_distance": 128,
                "relative_attention_num_buckets": 32,
                "decoder_start_token_id": 0,
                "attention_dir": "bidirectional",
                "use_attn_scale": False,
                "tie_word_embeddings": True,
                "original_architecture": "T5ForConditionalGeneration",
            },
        )


class TestApertusConfig:
    def test_apertus_basic(self):
        hf_config = SimpleNamespace(
            architectures=["ApertusForCausalLM"],
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            num_hidden_layers=32,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            hidden_act="xielu",
            rope_theta=500000.0,
            qk_norm=True,
            rope_scaling=None,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("swiss-ai/Apertus-8B-2509")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 4096,
                "d_head": 128,
                "n_heads": 32,
                "n_key_value_heads": 8,
                "d_mlp": 14336,
                "n_layers": 32,
                "n_ctx": 8192,
                "eps": 1e-5,
                "d_vocab": 32000,
                "act_fn": "xielu",
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "rotary_dim": 128,
                "rotary_base": 500000.0,
                "gated_mlp": False,
                "final_rms": True,
                "use_qk_norm": True,
                "original_architecture": "ApertusForCausalLM",
            },
        )

    def test_apertus_with_llama3_rope_scaling(self):
        hf_config = SimpleNamespace(
            architectures=["ApertusForCausalLM"],
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            num_hidden_layers=32,
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            hidden_act="xielu",
            rope_theta=500000.0,
            qk_norm=True,
            rope_scaling={
                "type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("swiss-ai/Apertus-8B-2509")
        _assert_cfg_matches(
            cfg,
            {
                "use_NTK_by_parts_rope": True,
                "NTK_by_parts_factor": 8.0,
                "NTK_by_parts_low_freq_factor": 1.0,
                "NTK_by_parts_high_freq_factor": 4.0,
                "NTK_original_ctx_len": 8192,
            },
        )


class TestGptOssConfig:
    def test_gpt_oss(self):
        hf_config = SimpleNamespace(
            architectures=["GptOssForCausalLM"],
            hidden_size=6144,
            head_dim=128,
            num_attention_heads=48,
            intermediate_size=16384,
            num_hidden_layers=44,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            vocab_size=100352,
            hidden_act="silu",
            rope_theta=500000.0,
            num_key_value_heads=8,
            num_local_experts=16,
            num_experts_per_tok=4,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("openai/gpt-oss-20b")
        _assert_cfg_matches(
            cfg,
            {
                "d_model": 6144,
                "d_head": 128,
                "n_heads": 48,
                "d_mlp": 16384,
                "n_layers": 44,
                "n_ctx": 4096,
                "d_vocab": 100352,
                "act_fn": "silu",
                "normalization_type": "RMS",
                "positional_embedding_type": "rotary",
                "rotary_base": 500000.0,
                "eps": 1e-5,
                "n_key_value_heads": 8,
                "gated_mlp": True,
                "final_rms": True,
                "use_local_attn": False,
                "rotary_dim": 128,
                "num_experts": 16,
                "experts_per_token": 4,
                "original_architecture": "GptOssForCausalLM",
            },
        )


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_unsupported_architecture_raises(self):
        hf_config = SimpleNamespace(
            architectures=["SomeNewArchitecture"],
        )
        with _mock_auto_config(hf_config):
            with pytest.raises(NotImplementedError, match="SomeNewArchitecture"):
                convert_hf_model_config("gpt2")

    def test_tokenizer_name_set(self):
        """All configs should have tokenizer_name set to the official model name."""
        cfg = convert_hf_model_config("llama-7b-hf")
        assert cfg["tokenizer_name"] == "llama-7b-hf"

    def test_trust_remote_code_kwarg(self):
        cfg = convert_hf_model_config("llama-7b-hf", trust_remote_code=True)
        assert cfg["trust_remote_code"] is True

    def test_tiny_stories_n_ctx_override(self):
        hf_config = SimpleNamespace(
            architectures=["GPT2LMHeadModel"],
            n_embd=64,
            n_head=4,
            n_layer=2,
            n_ctx=2048,
            layer_norm_epsilon=1e-5,
            vocab_size=50257,
            activation_function="gelu_new",
            scale_attn_by_inverse_layer_idx=False,
        )
        with _mock_auto_config(hf_config):
            cfg = convert_hf_model_config("roneneldan/TinyStories-1M")
        assert cfg["n_ctx"] == 512
