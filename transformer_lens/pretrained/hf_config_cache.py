"""Cached HF config objects for gated/problematic models.

This avoids needing an HF token for models whose architecture config
is known ahead of time (e.g. Meta Llama models).
"""

from __future__ import annotations

from typing import Optional, Union

from transformers import GemmaConfig, LlamaConfig

try:
    from transformers import Gemma2Config
except ImportError:  # transformers < 4.42
    Gemma2Config = None  # type: ignore[assignment,misc]

try:
    from transformers import Gemma3Config, Gemma3TextConfig
except ImportError:  # transformers < 4.50
    Gemma3Config = None  # type: ignore[assignment,misc]
    Gemma3TextConfig = None  # type: ignore[assignment,misc]

ConfigType = Union[GemmaConfig, Gemma2Config, Gemma3Config, Gemma3TextConfig, LlamaConfig]


_HF_CONFIG_CACHE: dict[str, ConfigType] = {
    # --- LLaMA 1 ---
    "llama-7b": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=4096,
        num_attention_heads=32,
        intermediate_size=11008,
        num_hidden_layers=32,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        vocab_size=32000,
    ),
    "llama-13b": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=5120,
        num_attention_heads=40,
        intermediate_size=13824,
        num_hidden_layers=40,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        vocab_size=32000,
    ),
    "llama-30b": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=6656,
        num_attention_heads=52,
        intermediate_size=17920,
        num_hidden_layers=60,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        vocab_size=32000,
    ),
    "llama-65b": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=8192,
        num_attention_heads=64,
        intermediate_size=22016,
        num_hidden_layers=80,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        vocab_size=32000,
    ),
    # --- LLaMA 2 ---
    "meta-llama/Llama-2-7b": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=4096,
        num_attention_heads=32,
        intermediate_size=11008,
        num_hidden_layers=32,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        vocab_size=32000,
    ),
    "meta-llama/Llama-2-13b": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=5120,
        num_attention_heads=40,
        intermediate_size=13824,
        num_hidden_layers=40,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        vocab_size=32000,
    ),
    "meta-llama/Llama-2-70b": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        num_hidden_layers=80,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        vocab_size=32000,
    ),
    # --- CodeLlama ---
    "codellama/CodeLlama-7b": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=4096,
        num_attention_heads=32,
        intermediate_size=11008,
        num_hidden_layers=32,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        vocab_size=32016,
        rope_theta=1000000,
    ),
    "codellama/CodeLlama-7b-Python": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=4096,
        num_attention_heads=32,
        intermediate_size=11008,
        num_hidden_layers=32,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        vocab_size=32000,
        rope_theta=1000000,
    ),
    # --- Meta-Llama-3 ---
    "meta-llama/Meta-Llama-3-8B": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        num_hidden_layers=32,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        vocab_size=128256,
        rope_theta=500000.0,
    ),
    "meta-llama/Meta-Llama-3-70B": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        num_hidden_layers=80,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        vocab_size=128256,
        rope_theta=500000.0,
    ),
    # --- Llama-3.1 ---
    "meta-llama/Llama-3.1-8B": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        num_hidden_layers=32,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        vocab_size=128256,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    ),
    "meta-llama/Llama-3.1-70B": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        num_hidden_layers=80,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        vocab_size=128256,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    ),
    # --- Llama-3.2 ---
    "meta-llama/Llama-3.2-1B": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=8192,
        num_hidden_layers=16,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        vocab_size=128256,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    ),
    "meta-llama/Llama-3.2-3B": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=3072,
        num_attention_heads=24,
        num_key_value_heads=8,
        intermediate_size=8192,
        num_hidden_layers=28,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        vocab_size=128256,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    ),
    # --- Llama-3.3 ---
    "meta-llama/Llama-3.3-70B": LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        num_hidden_layers=80,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        vocab_size=128256,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    ),
    # =========================================================================
    # Gemma 1
    # =========================================================================
    "google/gemma-2b": GemmaConfig(
        architectures=["GemmaForCausalLM"],
        hidden_size=2048,
        num_attention_heads=8,
        intermediate_size=16384,
        num_hidden_layers=18,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        vocab_size=256000,
        num_key_value_heads=1,
        head_dim=256,
        hidden_act="gelu_new",
    ),
    "google/gemma-7b": GemmaConfig(
        architectures=["GemmaForCausalLM"],
        hidden_size=3072,
        num_attention_heads=16,
        intermediate_size=24576,
        num_hidden_layers=28,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        vocab_size=256000,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act="gelu_new",
    ),
}

# =========================================================================
# Gemma 2 (requires transformers >= 4.50)
# =========================================================================
if Gemma2Config is not None:
    _HF_CONFIG_CACHE.update(
        {
            "google/gemma-2-2b": Gemma2Config(
                architectures=["Gemma2ForCausalLM"],
                hidden_size=2304,
                num_attention_heads=8,
                intermediate_size=9216,
                num_hidden_layers=26,
                max_position_embeddings=8192,
                rms_norm_eps=1e-6,
                vocab_size=256000,
                num_key_value_heads=4,
                head_dim=256,
                query_pre_attn_scalar=256,
                layer_types=["full_attention", "sliding_attention"] * 13,
            ),
            "google/gemma-2-9b": Gemma2Config(
                architectures=["Gemma2ForCausalLM"],
                hidden_size=3584,
                num_attention_heads=16,
                intermediate_size=14336,
                num_hidden_layers=42,
                max_position_embeddings=8192,
                rms_norm_eps=1e-6,
                vocab_size=256000,
                num_key_value_heads=8,
                head_dim=256,
                query_pre_attn_scalar=256,
                layer_types=["full_attention", "sliding_attention"] * 21,
            ),
            "google/gemma-2-27b": Gemma2Config(
                architectures=["Gemma2ForCausalLM"],
                hidden_size=4608,
                num_attention_heads=32,
                intermediate_size=36864,
                num_hidden_layers=46,
                max_position_embeddings=8192,
                rms_norm_eps=1e-6,
                vocab_size=256000,
                num_key_value_heads=16,
                head_dim=128,
                query_pre_attn_scalar=144,
                layer_types=["full_attention", "sliding_attention"] * 23,
            ),
        }
    )

# =========================================================================
# Gemma 3 text-only (Gemma3TextConfig -> Gemma3ForCausalLM)
# =========================================================================
if Gemma3TextConfig is not None:
    _HF_CONFIG_CACHE.update(
        {
            "google/gemma-3-270m": Gemma3TextConfig(
                architectures=["Gemma3ForCausalLM"],
                hidden_size=640,
                num_attention_heads=4,
                intermediate_size=2048,
                num_hidden_layers=18,
                max_position_embeddings=32768,
                rms_norm_eps=1e-6,
                vocab_size=262144,
                num_key_value_heads=1,
                head_dim=256,
                sliding_window=512,
            ),
            "google/gemma-3-1b": Gemma3TextConfig(
                architectures=["Gemma3ForCausalLM"],
                hidden_size=1152,
                num_attention_heads=4,
                intermediate_size=6912,
                num_hidden_layers=26,
                max_position_embeddings=32768,
                rms_norm_eps=1e-6,
                vocab_size=262144,
                num_key_value_heads=1,
                head_dim=256,
                sliding_window=512,
            ),
            "google/medgemma-27b-text": Gemma3TextConfig(
                architectures=["Gemma3ForCausalLM"],
                hidden_size=5376,
                num_attention_heads=32,
                intermediate_size=21504,
                num_hidden_layers=62,
                max_position_embeddings=131072,
                rms_norm_eps=1e-6,
                vocab_size=262144,
                num_key_value_heads=16,
                head_dim=128,
                sliding_window=1024,
            ),
        }
    )

# =========================================================================
# Gemma 3 multimodal (Gemma3Config -> Gemma3ForConditionalGeneration)
# =========================================================================
if Gemma3Config is not None:
    _HF_CONFIG_CACHE.update(
        {
            "google/gemma-3-4b": Gemma3Config(
                architectures=["Gemma3ForConditionalGeneration"],
                text_config={
                    "hidden_size": 2560,
                    "num_attention_heads": 8,
                    "intermediate_size": 10240,
                    "num_hidden_layers": 34,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262208,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "sliding_window": 1024,
                },
            ),
            "google/gemma-3-12b": Gemma3Config(
                architectures=["Gemma3ForConditionalGeneration"],
                text_config={
                    "hidden_size": 3840,
                    "num_attention_heads": 16,
                    "intermediate_size": 15360,
                    "num_hidden_layers": 48,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262208,
                    "num_key_value_heads": 8,
                    "head_dim": 256,
                    "sliding_window": 1024,
                },
            ),
            "google/gemma-3-27b": Gemma3Config(
                architectures=["Gemma3ForConditionalGeneration"],
                text_config={
                    "hidden_size": 5376,
                    "num_attention_heads": 32,
                    "intermediate_size": 21504,
                    "num_hidden_layers": 62,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262208,
                    "num_key_value_heads": 16,
                    "head_dim": 128,
                    "sliding_window": 1024,
                },
            ),
            "google/medgemma-4b": Gemma3Config(
                architectures=["Gemma3ForConditionalGeneration"],
                text_config={
                    "hidden_size": 2560,
                    "num_attention_heads": 8,
                    "intermediate_size": 10240,
                    "num_hidden_layers": 34,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262208,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "sliding_window": 1024,
                },
            ),
            "google/medgemma-27b": Gemma3Config(
                architectures=["Gemma3ForConditionalGeneration"],
                text_config={
                    "hidden_size": 5376,
                    "num_attention_heads": 32,
                    "intermediate_size": 21504,
                    "num_hidden_layers": 62,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 262208,
                    "num_key_value_heads": 16,
                    "head_dim": 128,
                    "sliding_window": 1024,
                },
            ),
        }
    )


def get_hf_config(official_model_name: str) -> Optional[ConfigType]:
    """Look up a cached HF config for a known model.

    Matches by longest prefix: e.g. "meta-llama/Llama-3.1-8B-Instruct" matches
    the "meta-llama/Llama-3.1-8B" entry. Returns None if no cache entry matches.
    """
    for prefix in sorted(_HF_CONFIG_CACHE, key=len, reverse=True):
        if official_model_name.startswith(prefix):
            return _HF_CONFIG_CACHE[prefix]
    return None
