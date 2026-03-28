"""Cached HF config objects for gated/problematic models.

This avoids needing an HF token for models whose architecture config
is known ahead of time (e.g. Meta Llama models).
"""

from __future__ import annotations

from typing import Optional

from transformers import LlamaConfig

_HF_CONFIG_CACHE: dict[str, LlamaConfig] = {
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
}


def get_hf_config(official_model_name: str) -> Optional[LlamaConfig]:
    """Look up a cached HF config for a known model.

    Matches by longest prefix: e.g. "meta-llama/Llama-3.1-8B-Instruct" matches
    the "meta-llama/Llama-3.1-8B" entry. Returns None if no cache entry matches.
    """
    for prefix in sorted(_HF_CONFIG_CACHE, key=len, reverse=True):
        if official_model_name.startswith(prefix):
            return _HF_CONFIG_CACHE[prefix]
    return None
