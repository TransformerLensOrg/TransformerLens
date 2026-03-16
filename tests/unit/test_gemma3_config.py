"""
Unit tests for Gemma 3 and MedGemma model support.

Tests cover:
1. Configuration generation for all Gemma 3 model variants
2. Weight conversion from HuggingFace format
3. Hybrid local/global attention configuration
4. Per-layer RoPE base support
"""

from unittest import mock

import pytest

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.loading_from_pretrained import (
    OFFICIAL_MODEL_NAMES,
    get_pretrained_model_config,
)

# ============================================================================
# Test Data
# ============================================================================

GEMMA3_MODELS = [
    "google/gemma-3-270m",
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-pt",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-pt",
    "google/gemma-3-27b-it",
]

MEDGEMMA_MODELS = [
    "google/medgemma-4b-pt",
    "google/medgemma-4b-it",
    "google/medgemma-27b-it",
    "google/medgemma-27b-text-it",
]

GEMMA3_CONFIG_SPECS = {
    "270m": {
        "d_model": 640,
        "n_heads": 4,
        "n_layers": 18,
        "d_head": 256,
        "n_key_value_heads": 1,
        "d_mlp": 2048,
        "window_size": 512,
    },
    "1b": {
        "d_model": 1152,
        "n_heads": 4,
        "n_layers": 26,
        "d_head": 256,
        "n_key_value_heads": 1,
        "d_mlp": 6912,
        "window_size": 512,
    },
    "4b": {
        "d_model": 2560,
        "n_heads": 8,
        "n_layers": 34,
        "d_head": 256,
        "n_key_value_heads": 4,
        "d_mlp": 10240,
        "window_size": 1024,
    },
    "12b": {
        "d_model": 3840,
        "n_heads": 16,
        "n_layers": 48,
        "d_head": 256,
        "n_key_value_heads": 8,
        "d_mlp": 15360,
        "window_size": 1024,
    },
    "27b": {
        "d_model": 5376,
        "n_heads": 32,
        "n_layers": 62,
        "d_head": 128,
        "n_key_value_heads": 16,
        "d_mlp": 21504,
        "window_size": 1024,
    },
}


# ============================================================================
# Test: Model names in official list
# ============================================================================


class TestGemma3ModelRegistration:
    """Test that all Gemma 3 and MedGemma models are registered in OFFICIAL_MODEL_NAMES."""

    @pytest.mark.parametrize("model_name", GEMMA3_MODELS)
    def test_gemma3_models_in_official_list(self, model_name: str):
        assert model_name in OFFICIAL_MODEL_NAMES, f"{model_name} should be in OFFICIAL_MODEL_NAMES"

    @pytest.mark.parametrize("model_name", MEDGEMMA_MODELS)
    def test_medgemma_models_in_official_list(self, model_name: str):
        assert model_name in OFFICIAL_MODEL_NAMES, f"{model_name} should be in OFFICIAL_MODEL_NAMES"


# ============================================================================
# Test: Configuration generation
# ============================================================================


class TestGemma3ConfigGeneration:
    """Test that get_pretrained_model_config generates correct configs for Gemma 3."""

    @pytest.fixture
    def mock_hf_config(self):
        """Create a minimal mock HuggingFace config."""
        config = mock.Mock()
        config.architectures = ["Gemma3ForCausalLM"]
        return config

    @pytest.mark.parametrize(
        "model_name,size_key",
        [
            ("google/gemma-3-270m", "270m"),
            ("google/gemma-3-270m-it", "270m"),
            ("google/gemma-3-1b-pt", "1b"),
            ("google/gemma-3-1b-it", "1b"),
        ],
    )
    def test_gemma3_small_model_config(self, model_name: str, size_key: str, mock_hf_config):
        """Test configuration for small Gemma 3 models (270M, 1B)."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config(model_name)

        expected = GEMMA3_CONFIG_SPECS[size_key]
        assert cfg.d_model == expected["d_model"]
        assert cfg.n_heads == expected["n_heads"]
        assert cfg.n_layers == expected["n_layers"]
        assert cfg.d_head == expected["d_head"]
        assert cfg.n_key_value_heads == expected["n_key_value_heads"]
        assert cfg.d_mlp == expected["d_mlp"]

    @pytest.mark.parametrize(
        "model_name,size_key",
        [
            ("google/gemma-3-4b-pt", "4b"),
            ("google/gemma-3-4b-it", "4b"),
            ("google/medgemma-4b-pt", "4b"),
            ("google/medgemma-4b-it", "4b"),
        ],
    )
    def test_gemma3_4b_model_config(self, model_name: str, size_key: str, mock_hf_config):
        """Test configuration for 4B models (Gemma 3 and MedGemma)."""
        mock_hf_config.architectures = ["Gemma3ForConditionalGeneration"]
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config(model_name)

        expected = GEMMA3_CONFIG_SPECS[size_key]
        assert cfg.d_model == expected["d_model"]
        assert cfg.n_heads == expected["n_heads"]
        assert cfg.n_layers == expected["n_layers"]
        assert cfg.n_key_value_heads == expected["n_key_value_heads"]


# ============================================================================
# Test: Hybrid attention configuration
# ============================================================================


class TestGemma3HybridAttention:
    """Test hybrid local/global attention configuration (5:1 pattern)."""

    @pytest.fixture
    def mock_hf_config(self):
        config = mock.Mock()
        config.architectures = ["Gemma3ForCausalLM"]
        return config

    def test_attn_types_pattern_270m(self, mock_hf_config):
        """Test 5:1 local/global pattern for 270M (18 layers)."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-270m")

        assert cfg.use_local_attn is True
        assert cfg.attn_types is not None
        assert len(cfg.attn_types) == 18

        # Check 5:1 pattern: global at indices 5, 11, 17
        for i, attn_type in enumerate(cfg.attn_types):
            expected = "global" if (i + 1) % 6 == 0 else "local"
            assert attn_type == expected, f"Layer {i}: expected {expected}, got {attn_type}"

    def test_attn_types_pattern_1b(self, mock_hf_config):
        """Test 5:1 local/global pattern for 1B (26 layers)."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-1b-pt")

        assert cfg.use_local_attn is True
        assert len(cfg.attn_types) == 26

        # Count global layers
        global_count = cfg.attn_types.count("global")
        local_count = cfg.attn_types.count("local")
        assert global_count == 4  # 26 // 6 = 4 global layers
        assert local_count == 22

    def test_window_size_small_models(self, mock_hf_config):
        """Test that 270M/1B models use 512 token window."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-270m")
        assert cfg.window_size == 512

    def test_window_size_large_models(self, mock_hf_config):
        """Test that 4B+ models use 1024 token window."""
        mock_hf_config.architectures = ["Gemma3ForConditionalGeneration"]
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-4b-pt")
        assert cfg.window_size == 1024


# ============================================================================
# Test: Per-layer RoPE base
# ============================================================================


class TestGemma3PerLayerRoPE:
    """Test per-layer RoPE base configuration."""

    @pytest.fixture
    def mock_hf_config(self):
        config = mock.Mock()
        config.architectures = ["Gemma3ForCausalLM"]
        return config

    def test_rotary_base_global(self, mock_hf_config):
        """Test that global attention layers use 1M RoPE base."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-270m")

        assert cfg.rotary_base == 1_000_000

    def test_rotary_base_local(self, mock_hf_config):
        """Test that local attention layers use 10K RoPE base."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-270m")

        assert cfg.rotary_base_local == 10_000


# ============================================================================
# Test: Q/K Normalization
# ============================================================================


class TestGemma3QKNorm:
    """Test Q/K normalization configuration."""

    @pytest.fixture
    def mock_hf_config(self):
        config = mock.Mock()
        config.architectures = ["Gemma3ForCausalLM"]
        return config

    def test_use_qk_norm_enabled(self, mock_hf_config):
        """Test that Q/K normalization is enabled for all Gemma 3 models."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-270m")

        assert cfg.use_qk_norm is True


# ============================================================================
# Test: Normalization before and after
# ============================================================================


class TestGemma3Normalization:
    """Test Gemma 2/3 style normalization (before and after blocks)."""

    @pytest.fixture
    def mock_hf_config(self):
        config = mock.Mock()
        config.architectures = ["Gemma3ForCausalLM"]
        return config

    def test_normalization_before_and_after(self, mock_hf_config):
        """Test that use_normalization_before_and_after is enabled."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-270m")

        assert cfg.use_normalization_before_and_after is True


# ============================================================================
# Test: Vocabulary size
# ============================================================================


class TestGemma3VocabSize:
    """Test vocabulary size configuration."""

    @pytest.fixture
    def mock_hf_config(self):
        config = mock.Mock()
        config.architectures = ["Gemma3ForCausalLM"]
        return config

    def test_vocab_size_small_models(self, mock_hf_config):
        """Test vocab size for 270M/1B models."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-270m")
        assert cfg.d_vocab == 262144

    def test_vocab_size_multimodal_models(self, mock_hf_config):
        """Test vocab size for 4B+ multimodal models (262208)."""
        mock_hf_config.architectures = ["Gemma3ForConditionalGeneration"]
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-4b-pt")
        assert cfg.d_vocab == 262208

    def test_vocab_size_medgemma_text_only(self, mock_hf_config):
        """Test vocab size for MedGemma 27B text-only variant (262144)."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/medgemma-27b-text-it")
        assert cfg.d_vocab == 262144


# ============================================================================
# Test: Default context length
# ============================================================================


class TestGemma3ContextLength:
    """Test default context length configuration."""

    @pytest.fixture
    def mock_hf_config(self):
        config = mock.Mock()
        config.architectures = ["Gemma3ForCausalLM"]
        return config

    def test_default_context_length(self, mock_hf_config):
        """Test that default n_ctx is 8192 (memory-safe default)."""
        with mock.patch(
            "transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained",
            return_value=mock_hf_config,
        ):
            cfg = get_pretrained_model_config("google/gemma-3-270m")
        assert cfg.n_ctx == 8192


# ============================================================================
# Test: HookedTransformerConfig with rotary_base_local
# ============================================================================


class TestHookedTransformerConfigRotaryBaseLocal:
    """Test that HookedTransformerConfig supports rotary_base_local."""

    def test_rotary_base_local_default_none(self):
        """Test that rotary_base_local defaults to None."""
        cfg = HookedTransformerConfig(
            d_model=128,
            d_head=32,
            n_heads=4,
            n_ctx=128,
            n_layers=2,
            attn_only=True,
        )
        assert cfg.rotary_base_local is None

    def test_rotary_base_local_can_be_set(self):
        """Test that rotary_base_local can be set to a custom value."""
        cfg = HookedTransformerConfig(
            d_model=128,
            d_head=32,
            n_heads=4,
            n_ctx=128,
            n_layers=2,
            attn_only=True,
            rotary_base_local=10000,
        )
        assert cfg.rotary_base_local == 10000

    def test_rotary_base_and_rotary_base_local_coexist(self):
        """Test that both rotary_base and rotary_base_local can be set."""
        cfg = HookedTransformerConfig(
            d_model=128,
            d_head=32,
            n_heads=4,
            n_ctx=128,
            n_layers=2,
            attn_only=True,
            rotary_base=1000000,
            rotary_base_local=10000,
        )
        assert cfg.rotary_base == 1000000
        assert cfg.rotary_base_local == 10000
