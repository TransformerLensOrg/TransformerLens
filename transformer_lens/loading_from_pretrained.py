"""Loading Pretrained Models Utilities.

This module contains functions for loading pretrained models from the Hugging Face Hub.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    BertForPreTraining,
    T5ForConditionalGeneration,
)

import transformer_lens.utils as utils
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import (
    convert_bert_weights,
    convert_bloom_weights,
    convert_coder_weights,
    convert_gemma_weights,
    convert_gpt2_weights,
    convert_gptj_weights,
    convert_llama_weights,
    convert_mingpt_weights,
    convert_mistral_weights,
    convert_mixtral_weights,
    convert_neel_solu_old_weights,
    convert_neo_weights,
    convert_neox_weights,
    convert_olmo2_weights,
    convert_olmo_weights,
    convert_olmoe_weights,
    convert_opt_weights,
    convert_phi3_weights,
    convert_phi_weights,
    convert_qwen2_weights,
    convert_qwen3_weights,
    convert_qwen_weights,
    convert_t5_weights,
)
from transformer_lens.supported_models import MODEL_ALIASES, OFFICIAL_MODEL_NAMES
from transformer_lens.utilities.hf_utils import get_rotary_pct_from_config

NON_HF_HOSTED_MODEL_NAMES = [
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
]
"""Official model names for models not hosted on HuggingFace."""

NEED_REMOTE_CODE_MODELS = (
    "bigcode/santacoder",
    "Qwen/Qwen-",
    "Qwen/Qwen3-",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/phi-4",
)


def make_model_alias_map() -> dict[str, str]:
    """
    Converts OFFICIAL_MODEL_NAMES (the list of actual model names on
    HuggingFace) and MODEL_ALIASES (a dictionary mapping official model names to
    aliases) into a dictionary mapping all aliases to the official model name.
    """
    model_alias_map = {}
    for official_model_name in OFFICIAL_MODEL_NAMES:
        aliases = MODEL_ALIASES.get(official_model_name, [])
        for alias in aliases:
            model_alias_map[alias.lower()] = official_model_name
        model_alias_map[official_model_name.lower()] = official_model_name
    return model_alias_map


def get_official_model_name(model_name: str) -> str:
    """
    Returns the official model name for a given model name (or alias).
    """
    model_alias_map = make_model_alias_map()
    official_model_name = model_alias_map.get(model_name.lower())
    if official_model_name is None:
        raise ValueError(
            f"{model_name} not found. Valid official model names (excl aliases): {OFFICIAL_MODEL_NAMES}"
        )
    return official_model_name


def convert_hf_model_config(model_name: str, **kwargs: Any) -> dict[str, Any]:
    """
    Returns the model config for a HuggingFace model, converted to a dictionary
    in the HookedTransformerConfig format.

    Takes the official_model_name as an input.
    """
    # In case the user passed in an alias
    if (Path(model_name) / "config.json").exists():
        logging.info("Loading model config from local directory")
        official_model_name = model_name
    else:
        official_model_name = get_official_model_name(model_name)

    # Load HuggingFace model config
    if "llama" in official_model_name.lower():
        architecture = "LlamaForCausalLM"
    elif "gemma-3" in official_model_name.lower() or "medgemma" in official_model_name.lower():
        # Gemma 3: 270M and 1B are text-only (CausalLM), 4B+ are multimodal (ConditionalGeneration)
        # Exception: medgemma-27b-text-it is text-only
        if "270m" in official_model_name.lower() or "1b" in official_model_name.lower():
            architecture = "Gemma3ForCausalLM"
        elif "medgemma-27b-text" in official_model_name.lower():
            # medgemma-27b-text-it is text-only variant
            architecture = "Gemma3ForCausalLM"
        else:
            # 4B, 12B, 27B and medgemma are multimodal
            architecture = "Gemma3ForConditionalGeneration"
    elif "gemma-2" in official_model_name.lower():
        architecture = "Gemma2ForCausalLM"
    elif "gemma" in official_model_name.lower():
        architecture = "GemmaForCausalLM"
    else:
        huggingface_token = os.environ.get("HF_TOKEN", "")
        hf_config = AutoConfig.from_pretrained(
            official_model_name,
            token=huggingface_token if len(huggingface_token) > 0 else None,
            **kwargs,
        )
        architecture = hf_config.architectures[0]

    cfg_dict: dict[str, Any]
    if official_model_name.startswith(
        ("llama-7b", "meta-llama/Llama-2-7b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048 if official_model_name.startswith("llama-7b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-7b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif official_model_name.startswith("codellama"):  # same architecture CodeLlama and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32016,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 1000000,
        }
        if "python" in official_model_name.lower():
            # The vocab size of python version of CodeLlama-7b is 32000
            cfg_dict["d_vocab"] = 32000
    elif official_model_name.startswith(
        ("llama-13b", "meta-llama/Llama-2-13b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048 if official_model_name.startswith("llama-13b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-13b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 5120 // 40,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-30b" in official_model_name:
        cfg_dict = {
            "d_model": 6656,
            "d_head": 6656 // 52,
            "n_heads": 52,
            "d_mlp": 17920,
            "n_layers": 60,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 6656 // 52,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-65b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 8192 // 64,
            "n_heads": 64,
            "d_mlp": 22016,
            "n_layers": 80,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 8192 // 64,
            "rotary_adjacent_pairs": False,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Llama-2-70b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Meta-Llama-3-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
        }
    elif "Meta-Llama-3-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
        }
    elif "Llama-3.2-1B" in official_model_name:
        cfg_dict = {
            "d_model": 2048,
            "d_head": 64,
            "n_heads": 32,
            "d_mlp": 8192,
            "n_layers": 16,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 64,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 32.0,
            "NTK_original_ctx_len": 8192,
        }
    elif "Llama-3.2-3B" in official_model_name:
        cfg_dict = {
            "d_model": 3072,
            "d_head": 128,
            "n_heads": 24,
            "d_mlp": 8192,
            "n_layers": 28,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 32.0,
            "NTK_original_ctx_len": 8192,
        }
    elif "Llama-3.3-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
            "NTK_original_ctx_len": 8192,
        }
    elif "Llama-3.1-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
            "NTK_original_ctx_len": 8192,
        }
    elif "Llama-3.1-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
            "NTK_original_ctx_len": 8192,
        }
    elif architecture == "GPTNeoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "attn_types": hf_config.attention_layers,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": False,
            "use_local_attn": True,
            "window_size": hf_config.window_size,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTJForCausalLM":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.rotary_dim,
            "rotary_adjacent_pairs": True,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
        }
        rotary_pct = get_rotary_pct_from_config(hf_config)
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    elif architecture == "BertForMaskedLM":
        # All supported Bert architectures have the same config,
        # so we can use the BertForMaskedLM config for all of them
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu",
            "attention_dir": "bidirectional",
        }
    elif architecture == "MistralForCausalLM":
        use_local_attn = True if hf_config.sliding_window else False
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": (
                hf_config.head_dim
                if hasattr(hf_config, "head_dim")
                and hf_config.head_dim is not None
                and hf_config.head_dim > 0
                else hf_config.hidden_size // hf_config.num_attention_heads
            ),
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "window_size": hf_config.sliding_window,  # None if no sliding window was used
            "attn_types": ["local"] * hf_config.num_hidden_layers if use_local_attn else None,
            "eps": hf_config.rms_norm_eps,
            "rotary_base": hf_config.rope_theta,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "use_local_attn": use_local_attn,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "gated_mlp": True,
        }
    elif architecture == "MixtralForCausalLM":
        cfg_dict = {
            "dtype": torch.bfloat16,
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,  # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "window_size": hf_config.sliding_window,  # This is None, as no sliding window was used
            "attn_types": ["global"] * 32,
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "gated_mlp": True,
            "use_local_attn": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "num_experts": hf_config.num_local_experts,
            "experts_per_token": hf_config.num_experts_per_tok,
        }
    elif architecture == "BloomForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": 2048,  # Capped due to HF Tokenizer Constraints
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu_fast",
            "eps": hf_config.layer_norm_epsilon,
            "normalization_type": "LN",
            "post_embedding_ln": True,
            "positional_embedding_type": "alibi",
            "default_prepend_bos": False,
        }
    elif architecture == "GPT2LMHeadCustomModel":
        # santacoder
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "trust_remote_code": "santacoder"
            in official_model_name,  # Only santacoder needs trust_remote_code
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "LlamaForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            # This is done because the current implementation of GQA will use Grouped-Query Attention if
            # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
            # the same as hf_config.num_attention_heads, in which case GQA should not be used.
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "QWenLMHeadModel":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size // 2,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "silu",
            "use_attn_scale": hf_config.scale_attn_weights,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.kv_channels,
            "rotary_adjacent_pairs": False,
            "tokenizer_prepends_bos": True,
            "trust_remote_code": True,
            "final_rms": True,
            "gated_mlp": True,
            "default_prepend_bos": False,
        }
    elif architecture == "Qwen2ForCausalLM":
        # Note that Qwen1.5 models have architecture type Qwen2ForCausalLM.
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": int(hf_config.rope_theta),
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
            "default_prepend_bos": False,
        }
    elif architecture == "Qwen3ForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.head_dim
            if hasattr(hf_config, "head_dim")
            and hf_config.head_dim is not None
            and hf_config.head_dim > 0
            else hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": int(hf_config.rope_theta),
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.head_dim
            if hasattr(hf_config, "head_dim") and hf_config.head_dim > 0
            else hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
            "default_prepend_bos": False,
            "use_qk_norm": True,
            "trust_remote_code": True,
        }
    elif architecture == "PhiForCausalLM":
        # Architecture for microsoft/phi models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "LN",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "parallel_attn_mlp": True,
        }
        partial_rotary_factor = hf_config.partial_rotary_factor
        cfg_dict["rotary_dim"] = round(partial_rotary_factor * cfg_dict["d_head"])
    elif architecture == "Phi3ForCausalLM":
        # Architecture for microsoft/phi3 models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "gated_mlp": True,
            "parallel_attn_mlp": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        }

    elif official_model_name.startswith("google/gemma-2b"):
        # Architecture for Gemma 2b and Gemma 2b Instruct models
        cfg_dict = {
            "d_model": 2048,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 16384,
            "n_layers": 18,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 1,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-7b"):
        # Architecture for Gemma 7b and Gemma 7b Instruct models
        cfg_dict = {
            "d_model": 3072,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 24576,
            "n_layers": 28,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 16,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-2-2b"):
        # Architecture for Gemma-2 2b and Gemma-2 2b Instruct models
        cfg_dict = {
            "d_model": 2304,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 9216,
            "n_layers": 26,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 4,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-9b"):
        # Architecture for Gemma-2 9b and Gemma-2 9b Instruct models
        cfg_dict = {
            "d_model": 3584,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 14336,
            "n_layers": 42,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 8,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-27b"):
        # Architecture for Gemma-2 27b and Gemma-2 27b Instruct models
        cfg_dict = {
            "d_model": 4608,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 36864,
            "n_layers": 46,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "attn_scale": 12.0,
            "n_key_value_heads": 16,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 23,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-3-270m"):
        # Architecture for Gemma-3 270m and Gemma-3 270m Instruct models
        cfg_dict = {
            "d_model": 640,
            "d_head": 256,
            "n_heads": 4,
            "d_mlp": 2048,
            "n_layers": 18,
            "n_ctx": 8192,  # Safe default (model supports up to 32K). Override: cfg_kwargs={"n_ctx": 32768}
            "eps": 1e-06,
            "d_vocab": 262144,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 1000000,  # Global attention layers
            "rotary_base_local": 10000,  # Local attention layers (per Gemma 3 paper)
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 1,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
            "use_qk_norm": True,
            "window_size": 512,
            "use_local_attn": True,
            "attn_types": [
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
            ],
        }
    elif official_model_name.startswith("google/gemma-3-1b"):
        # Architecture for Gemma-3 1b-pt and Gemma-3 1b-it models
        cfg_dict = {
            "d_model": 1152,
            "d_head": 256,
            "n_heads": 4,
            "d_mlp": 6912,
            "n_layers": 26,
            "n_ctx": 8192,  # Safe default (model supports up to 32K). Override: cfg_kwargs={"n_ctx": 32768}
            "eps": 1e-06,
            "d_vocab": 262144,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 1000000,  # Global attention layers
            "rotary_base_local": 10000,  # Local attention layers (per Gemma 3 paper)
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 1,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
            "use_qk_norm": True,
            "window_size": 512,
            "use_local_attn": True,
            "attn_types": [
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
            ],
        }
    elif official_model_name.startswith("google/gemma-3-4b") or official_model_name.startswith(
        "google/medgemma-4b"
    ):
        # Architecture for Gemma-3 4b and MedGemma 4b models (multimodal, text-only extraction)
        cfg_dict = {
            "d_model": 2560,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 10240,
            "n_layers": 34,
            "n_ctx": 8192,  # Safe default (model supports up to 128K). Override: cfg_kwargs={"n_ctx": 131072}
            "eps": 1e-06,
            "d_vocab": 262208,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 1000000,  # Global attention layers
            "rotary_base_local": 10000,  # Local attention layers (per Gemma 3 paper)
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 4,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
            "use_qk_norm": True,
            "window_size": 1024,
            "use_local_attn": True,
            "attn_types": [
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
            ],
        }
    elif official_model_name.startswith("google/gemma-3-12b"):
        # Architecture for Gemma-3 12b models (multimodal, text-only extraction)
        cfg_dict = {
            "d_model": 3840,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 15360,
            "n_layers": 48,
            "n_ctx": 8192,  # Safe default (model supports up to 128K). Override: cfg_kwargs={"n_ctx": 131072}
            "eps": 1e-06,
            "d_vocab": 262208,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 1000000,  # Global attention layers
            "rotary_base_local": 10000,  # Local attention layers (per Gemma 3 paper)
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 8,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
            "use_qk_norm": True,
            "window_size": 1024,
            "use_local_attn": True,
            "attn_types": [
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
            ],
        }
    elif official_model_name.startswith("google/gemma-3-27b") or official_model_name.startswith(
        "google/medgemma-27b"
    ):
        # Architecture for Gemma-3 27b and MedGemma 27b models (multimodal/text-only extraction)
        # Note: medgemma-27b-text-it uses Gemma3ForCausalLM (text-only), others use Gemma3ForConditionalGeneration
        cfg_dict = {
            "d_model": 5376,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 21504,
            "n_layers": 62,
            "n_ctx": 8192,  # Safe default (model supports up to 128K). Override: cfg_kwargs={"n_ctx": 131072}
            "eps": 1e-06,
            "d_vocab": (
                262144 if official_model_name == "google/medgemma-27b-text-it" else 262208
            ),  # text-only variant uses 262144
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 1000000,  # Global attention layers
            "rotary_base_local": 10000,  # Local attention layers (per Gemma 3 paper)
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 16,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
            "use_qk_norm": True,
            "window_size": 1024,
            "use_local_attn": True,
            "attn_types": [
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
                "local",
                "local",
                "local",
                "global",
                "local",
                "local",
            ],
        }
    elif official_model_name.startswith("google/gemma-2b"):
        # Architecture for Gemma 2b and Gemma 2b Instruct models
        cfg_dict = {
            "d_model": 2048,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 16384,
            "n_layers": 18,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 1,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-7b"):
        # Architecture for Gemma 7b and Gemma 7b Instruct models
        cfg_dict = {
            "d_model": 3072,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 24576,
            "n_layers": 28,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 16,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-2-2b"):
        # Architecture for Gemma-2 2b and Gemma-2 2b Instruct models
        cfg_dict = {
            "d_model": 2304,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 9216,
            "n_layers": 26,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 4,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-9b"):
        # Architecture for Gemma-2 9b and Gemma-2 9b Instruct models
        cfg_dict = {
            "d_model": 3584,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 14336,
            "n_layers": 42,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 8,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-27b"):
        # Architecture for Gemma-2 27b and Gemma-2 27b Instruct models
        cfg_dict = {
            "d_model": 4608,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 36864,
            "n_layers": 46,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "attn_scale": 12.0,
            "n_key_value_heads": 16,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 23,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("allenai/OLMo-1B") and official_model_name.endswith("hf"):
        cfg_dict = {
            "d_model": 2048,
            "d_head": 128,
            "n_heads": 16,
            "d_mlp": 8192,
            "n_layers": 16,
            "n_ctx": 2048,
            "eps": 1e-05,
            "d_vocab": 50304,
            "act_fn": "silu",
            "initializer_range": 0.02,
            "normalization_type": "LN",
            "rotary_base": 10000.0,
            "attn_types": ["global"] * 16,
            "positional_embedding_type": "rotary",
            "gated_mlp": True,
        }
    elif official_model_name.startswith("allenai/OLMo-7B") and official_model_name.endswith("hf"):
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048,
            "eps": 1e-05,
            "d_vocab": 50304,
            "act_fn": "silu",
            "initializer_range": 0.02,
            "normalization_type": "LN",
            "rotary_base": 10000.0,
            "attn_types": ["global"] * 32,
            "positional_embedding_type": "rotary",
            "gated_mlp": True,
        }
    elif official_model_name.startswith("allenai/OLMo-2-0425-1B"):
        cfg_dict = {
            "d_model": 2048,
            "d_head": 128,
            "n_heads": 16,
            "d_mlp": 8192,
            "n_layers": 16,
            "n_ctx": 4096,
            "eps": 1e-06,
            "d_vocab": 100352,
            "act_fn": "silu",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 500000.0,
            "attn_types": ["global"] * 16,
            "positional_embedding_type": "rotary",
            "gated_mlp": True,
        }
    elif official_model_name.startswith("allenai/OLMo-2-1124-7B"):
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 4096,
            "eps": 1e-06,
            "d_vocab": 100352,
            "act_fn": "silu",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 500000.0,
            "attn_types": ["global"] * 32,
            "positional_embedding_type": "rotary",
            "gated_mlp": True,
        }
    elif architecture == "OlmoeForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "num_experts": hf_config.num_experts,
            "experts_per_token": hf_config.num_experts_per_tok,
            "norm_topk_prob": hf_config.norm_topk_prob,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "rotary_base": getattr(
                hf_config,
                "rope_theta",
                hf_config.rope_parameters.get("rope_theta", 10000.0),
            ),
            "tie_word_embeddings": hf_config.tie_word_embeddings,
            "initializer_range": hf_config.initializer_range,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "gated_mlp": True,
            "normalization_type": "RMS",
        }
    elif architecture == "T5ForConditionalGeneration":
        cfg_dict = {
            "d_model": hf_config.d_model,
            "d_head": hf_config.d_kv,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.d_ff,
            "d_vocab": hf_config.vocab_size,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_length,
            "eps": hf_config.layer_norm_epsilon,
            "act_fn": hf_config.feed_forward_proj,
            "positional_embedding_type": "relative_positional_bias",
            "relative_attention_max_distance": hf_config.relative_attention_max_distance,
            "relative_attention_num_buckets": hf_config.relative_attention_num_buckets,
            "decoder_start_token_id": hf_config.decoder_start_token_id,
            "attention_dir": "bidirectional",
            "use_attn_scale": False,
            "tie_word_embeddings": hf_config.tie_word_embeddings,
        }
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict


def convert_neel_model_config(official_model_name: str, **kwargs: Any) -> dict[str, Any]:
    """
    Loads the config for a model trained by me (NeelNanda), converted to a dictionary
    in the HookedTransformerConfig format.

    AutoConfig is not supported, because these models are in the HookedTransformer format, so we directly download and load the json.
    """
    official_model_name = get_official_model_name(official_model_name)
    cfg_json: dict = utils.download_file_from_hf(official_model_name, "config.json", **kwargs)
    cfg_arch = cfg_json.get(
        "architecture", "neel" if "_old" not in official_model_name else "neel-solu-old"
    )
    cfg_dict = {
        "d_model": cfg_json["d_model"],
        "n_layers": cfg_json["n_layers"],
        "d_mlp": cfg_json["d_mlp"],
        "d_head": cfg_json["d_head"],
        "n_heads": cfg_json["n_heads"],
        "n_ctx": cfg_json["n_ctx"],
        "d_vocab": cfg_json["d_vocab"],
        "tokenizer_name": cfg_json.get("tokenizer_name", None),
        "act_fn": cfg_json["act_fn"],
        "attn_only": cfg_json["attn_only"],
        "final_rms": cfg_json.get("final_rms", False),
        "original_architecture": cfg_arch,
    }
    if "normalization" in cfg_json:
        cfg_dict["normalization_type"] = cfg_json["normalization"]
    else:
        cfg_dict["normalization_type"] = cfg_json["normalization_type"]
    if "shortformer_pos" in cfg_json:
        cfg_dict["positional_embedding_type"] = (
            "shortformer" if cfg_json["shortformer_pos"] else "standard"
        )
    else:
        cfg_dict["positional_embedding_type"] = "standard"
    return cfg_dict


def get_pretrained_model_config(
    model_name: str,
    hf_cfg: dict[str, Any] | None = None,
    checkpoint_index: int | None = None,
    checkpoint_value: int | None = None,
    fold_ln: bool = False,
    device: str | torch.device | None = None,
    n_devices: int = 1,
    default_prepend_bos: bool | None = None,
    dtype: torch.dtype = torch.float32,
    first_n_layers: int | None = None,
    **kwargs: Any,
) -> HookedTransformerConfig:
    """Returns the pretrained model config as an HookedTransformerConfig object.

    There are two types of pretrained models: HuggingFace models (where
    AutoModel and AutoConfig work), and models trained by me (NeelNanda) which
    aren't as integrated with HuggingFace infrastructure.

    Args:
        model_name: The name of the model. This can be either the official
            HuggingFace model name, or the name of a model trained by me
            (NeelNanda).
        hf_cfg (dict, optional): Config of a loaded pretrained HF model,
            converted to a dictionary.
        checkpoint_index (int, optional): If loading from a
            checkpoint, the index of the checkpoint to load. Defaults to None.
        checkpoint_value (int, optional): If loading from a checkpoint, the
        value of
            the checkpoint to load, ie the step or token number (each model has
            checkpoints labelled with exactly one of these). Defaults to None.
        fold_ln (bool, optional): Whether to fold the layer norm into the
            subsequent linear layers (see HookedTransformer.fold_layer_norm for
            details). Defaults to False.
        device (str, optional): The device to load the model onto. By
            default will load to CUDA if available, else CPU.
        n_devices (int, optional): The number of devices to split the model across. Defaults to 1.
        default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
            methods of HookedTransformer process input text to tokenize (only when input is a string).
            Resolution order for default_prepend_bos:
            1. If user passes value explicitly, use that value
            2. Model-specific default from cfg_dict if it exists (e.g. for bloom models it's False)
            3. Global default (True)

            Even for models not explicitly trained with the BOS token, heads often use the
            first position as a resting position and accordingly lose information from the first token,
            so this empirically seems to give better results. Note that you can also locally override the default behavior
            by passing in prepend_bos=True/False when you call a method that processes the input string.
        dtype (torch.dtype, optional): The dtype to load the TransformerLens model in.
        kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
            Also given to other HuggingFace functions when compatible.

    """
    if Path(model_name).exists():
        # If the model_name is a path, it's a local model
        cfg_dict = convert_hf_model_config(model_name, **kwargs)
        official_model_name = model_name
    else:
        official_model_name = get_official_model_name(model_name)
    if (
        official_model_name.startswith("NeelNanda")
        or official_model_name.startswith("ArthurConmy")
        or official_model_name.startswith("Baidicoot")
    ):
        cfg_dict = convert_neel_model_config(official_model_name, **kwargs)
    else:
        if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
            "trust_remote_code", False
        ):
            logging.warning(
                f"Loading model {official_model_name} requires setting trust_remote_code=True"
            )
            kwargs["trust_remote_code"] = True
        cfg_dict = convert_hf_model_config(official_model_name, **kwargs)
    # Processing common to both model types
    # Remove any prefix, saying the organization who made a model.
    cfg_dict["model_name"] = official_model_name.split("/")[-1]
    # Don't need to initialize weights, we're loading from pretrained
    cfg_dict["init_weights"] = False

    if (
        "positional_embedding_type" in cfg_dict
        and cfg_dict["positional_embedding_type"] == "shortformer"
        and fold_ln
    ):
        logging.warning(
            "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_ln=False instead."
        )
        fold_ln = False

    # OLMo 2 uses post-norm (norm after attention/MLP, not before), so folding
    # the norm weights into adjacent linear layers is not mathematically valid.
    if cfg_dict.get("original_architecture") == "Olmo2ForCausalLM" and fold_ln:
        logging.warning(
            "fold_ln=True is incompatible with OLMo 2's post-norm architecture. "
            "Setting fold_ln=False."
        )
        fold_ln = False

    if device is not None:
        cfg_dict["device"] = device

    cfg_dict["dtype"] = dtype

    if fold_ln:
        if cfg_dict["normalization_type"] in ["LN", "LNPre"]:
            cfg_dict["normalization_type"] = "LNPre"
        elif cfg_dict["normalization_type"] in ["RMS", "RMSPre"]:
            cfg_dict["normalization_type"] = "RMSPre"
        else:
            logging.warning("Cannot fold in layer norm, normalization_type is not LN.")

    if checkpoint_index is not None or checkpoint_value is not None:
        checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(
            official_model_name,
            **kwargs,
        )
        cfg_dict["from_checkpoint"] = True
        cfg_dict["checkpoint_label_type"] = checkpoint_label_type
        if checkpoint_index is not None:
            cfg_dict["checkpoint_index"] = checkpoint_index
            cfg_dict["checkpoint_value"] = checkpoint_labels[checkpoint_index]
        elif checkpoint_value is not None:
            assert (
                checkpoint_value in checkpoint_labels
            ), f"Checkpoint value {checkpoint_value} is not in list of available checkpoints"
            cfg_dict["checkpoint_value"] = checkpoint_value
            cfg_dict["checkpoint_index"] = checkpoint_labels.index(checkpoint_value)
    else:
        cfg_dict["from_checkpoint"] = False

    cfg_dict["device"] = device
    cfg_dict["n_devices"] = n_devices

    if default_prepend_bos is not None:
        # User explicitly set prepend_bos behavior, override config/default value
        cfg_dict["default_prepend_bos"] = default_prepend_bos
    elif "default_prepend_bos" not in cfg_dict:
        # No config value or user override, set default value (True)
        cfg_dict["default_prepend_bos"] = True

    if hf_cfg is not None:
        cfg_dict["load_in_4bit"] = hf_cfg.get("quantization_config", {}).get("load_in_4bit", False)
        cfg_dict["d_vocab"] = hf_cfg.get("vocab_size", cfg_dict["d_vocab"])
        if cfg_dict["original_architecture"] == "Qwen2ForCausalLM":
            cfg_dict["rotary_base"] = hf_cfg.get("rope_theta", cfg_dict["rotary_base"])
    if first_n_layers is not None:
        cfg_dict["n_layers"] = first_n_layers

    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    return cfg


def get_num_params_of_pretrained(model_name: str) -> int:
    """
    Returns the number of parameters of a pretrained model, used to filter to only run code for sufficiently small models.
    """
    cfg = get_pretrained_model_config(model_name)
    if cfg.n_params is None:
        raise ValueError(f"n_params not calculated for model {model_name}")
    return cfg.n_params


# %% Load checkpointed model state dicts
# The steps for which there are checkpoints in the stanford crfm models
STANFORD_CRFM_CHECKPOINTS = (
    list(range(0, 100, 10))
    + list(range(100, 2000, 50))
    + list(range(2000, 20000, 100))
    + list(range(20000, 400000 + 1, 1000))
)

# Linearly spaced checkpoints for Pythia models, taken every 1000 steps.
# Batch size 2,097,152 tokens, so checkpoints every 2.1B tokens
PYTHIA_CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
    range(1000, 143000 + 1, 1000)
)
# Pythia V1 has log-spaced early checkpoints (see line above), but V0 doesn't
PYTHIA_V0_CHECKPOINTS = list(range(1000, 143000 + 1, 1000))


def get_checkpoint_labels(model_name: str, **kwargs: Any) -> tuple[list[int], str]:
    """Returns the checkpoint labels for a given model, and the label_type
    (step or token). Raises an error for models that are not checkpointed."""
    official_model_name = get_official_model_name(model_name)
    if official_model_name.startswith("stanford-crfm/"):
        return STANFORD_CRFM_CHECKPOINTS, "step"
    elif official_model_name.startswith("EleutherAI/pythia"):
        if "v0" in official_model_name:
            return PYTHIA_V0_CHECKPOINTS, "step"
        else:
            logging.warning(
                "Pythia models on HF were updated on 4/3/23! add '-v0' to model name to access the old models."
            )
            return PYTHIA_CHECKPOINTS, "step"
    elif official_model_name.startswith("NeelNanda/"):
        api = HfApi()
        files_list = api.list_repo_files(
            official_model_name,
            **utils.select_compatible_kwargs(kwargs, api.list_repo_files),
        )
        labels = []
        for file_name in files_list:
            match = re.match(r"checkpoints/.*_(\d*)\.pth", file_name)
            if match:
                labels.append(int(match.group(1)))
        if labels[-1] > 1e9:
            label_type = "token"
        else:
            label_type = "step"
        return labels, label_type
    else:
        raise ValueError(f"Model {official_model_name} is not checkpointed.")


# %% Loading state dicts
def get_pretrained_state_dict(
    official_model_name: str,
    cfg: HookedTransformerConfig,
    hf_model: Any | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Loads in the model weights for a pretrained model, and processes them to
    have the HookedTransformer parameter names and shapes. Supports checkpointed
    models (and expects the checkpoint info to be stored in the config object)

    hf_model: Optionally, a HuggingFace model object. If provided, we will use
        these weights rather than reloading the model.
    dtype: The dtype to load the HuggingFace model in.
    kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
        Also given to other HuggingFace functions when compatible.
    """
    if "torch_dtype" in kwargs:
        dtype = kwargs["torch_dtype"]
        del kwargs["torch_dtype"]
    if Path(official_model_name).exists():
        official_model_name = str(Path(official_model_name).resolve())
        logging.info(f"Loading model from local path {official_model_name}")
    else:
        official_model_name = get_official_model_name(official_model_name)
    if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get(
        "trust_remote_code", False
    ):
        logging.warning(
            f"Loading model {official_model_name} state dict requires setting trust_remote_code=True"
        )
        kwargs["trust_remote_code"] = True
    if (
        official_model_name.startswith("NeelNanda")
        or official_model_name.startswith("ArthurConmy")
        or official_model_name.startswith("Baidicoot")
    ):
        api = HfApi()
        repo_files = api.list_repo_files(
            official_model_name,
            **utils.select_compatible_kwargs(kwargs, api.list_repo_files),
        )
        if cfg.from_checkpoint:
            file_name = list(
                filter(lambda x: x.endswith(f"{cfg.checkpoint_value}.pth"), repo_files)
            )[0]
        else:
            file_name = list(filter(lambda x: x.endswith("final.pth"), repo_files))[0]
        state_dict = utils.download_file_from_hf(official_model_name, file_name, **kwargs)

        # Convert to dtype
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

        if cfg.original_architecture == "neel-solu-old":
            state_dict = convert_neel_solu_old_weights(state_dict, cfg)
        elif cfg.original_architecture == "mingpt":
            state_dict = convert_mingpt_weights(state_dict, cfg)
        return state_dict
    else:
        if cfg.from_checkpoint:
            huggingface_token = os.environ.get("HF_TOKEN", "")
            if official_model_name.startswith("stanford-crfm"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"checkpoint-{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token if len(huggingface_token) > 0 else None,
                    **kwargs,
                )
            elif official_model_name.startswith("EleutherAI/pythia"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"step{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
            else:
                raise ValueError(f"Checkpoints for model {official_model_name} are not supported")
        elif hf_model is None:
            huggingface_token = os.environ.get("HF_TOKEN", "")
            if official_model_name in NON_HF_HOSTED_MODEL_NAMES:
                raise NotImplementedError("Model not hosted on HuggingFace, must pass in hf_model")
            elif "bert" in official_model_name:
                hf_model = BertForPreTraining.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token if len(huggingface_token) > 0 else None,
                    **kwargs,
                )
            elif "t5" in official_model_name:
                hf_model = T5ForConditionalGeneration.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token if len(huggingface_token) > 0 else None,
                    **kwargs,
                )
            elif cfg.original_architecture == "Gemma3ForConditionalGeneration":
                # Multimodal Gemma 3 models - use AutoModel
                hf_model = AutoModel.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token if len(huggingface_token) > 0 else None,
                    **kwargs,
                )
            else:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    torch_dtype=dtype,
                    token=huggingface_token if len(huggingface_token) > 0 else None,
                    **kwargs,
                )

            # Load model weights, and fold in layer norm weights
            if hf_model is not None:
                for param in hf_model.parameters():
                    param.requires_grad = False

        if cfg.original_architecture == "GPT2LMHeadModel":
            state_dict = convert_gpt2_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTNeoForCausalLM":
            state_dict = convert_neo_weights(hf_model, cfg)
        elif cfg.original_architecture == "OPTForCausalLM":
            state_dict = convert_opt_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTJForCausalLM":
            state_dict = convert_gptj_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPTNeoXForCausalLM":
            state_dict = convert_neox_weights(hf_model, cfg)
        elif cfg.original_architecture == "LlamaForCausalLM":
            state_dict = convert_llama_weights(hf_model, cfg)
        elif cfg.original_architecture == "BertForMaskedLM":
            state_dict = convert_bert_weights(hf_model, cfg)
        elif cfg.original_architecture == "T5ForConditionalGeneration":
            state_dict = convert_t5_weights(hf_model, cfg)
        elif cfg.original_architecture == "MistralForCausalLM":
            state_dict = convert_mistral_weights(hf_model, cfg)
        elif cfg.original_architecture == "MixtralForCausalLM":
            state_dict = convert_mixtral_weights(hf_model, cfg)
        elif cfg.original_architecture == "BloomForCausalLM":
            state_dict = convert_bloom_weights(hf_model, cfg)
        elif cfg.original_architecture == "GPT2LMHeadCustomModel":
            state_dict = convert_coder_weights(hf_model, cfg)
        elif cfg.original_architecture == "QWenLMHeadModel":
            state_dict = convert_qwen_weights(hf_model, cfg)
        elif cfg.original_architecture == "Qwen2ForCausalLM":
            state_dict = convert_qwen2_weights(hf_model, cfg)
        elif cfg.original_architecture == "Qwen3ForCausalLM":
            state_dict = convert_qwen3_weights(hf_model, cfg)
        elif cfg.original_architecture == "PhiForCausalLM":
            state_dict = convert_phi_weights(hf_model, cfg)
        elif cfg.original_architecture == "Phi3ForCausalLM":
            state_dict = convert_phi3_weights(hf_model, cfg)
        elif cfg.original_architecture == "GemmaForCausalLM":
            state_dict = convert_gemma_weights(hf_model, cfg)
        elif cfg.original_architecture == "Gemma2ForCausalLM":
            state_dict = convert_gemma_weights(hf_model, cfg)
        elif cfg.original_architecture == "Gemma3ForCausalLM":
            state_dict = convert_gemma_weights(hf_model, cfg)
        elif cfg.original_architecture == "Gemma3ForConditionalGeneration":
            state_dict = convert_gemma_weights(hf_model, cfg)
        elif cfg.original_architecture == "OlmoForCausalLM":
            state_dict = convert_olmo_weights(hf_model, cfg)
        elif cfg.original_architecture == "Olmo2ForCausalLM":
            state_dict = convert_olmo2_weights(hf_model, cfg)
        elif cfg.original_architecture == "OlmoeForCausalLM":
            state_dict = convert_olmoe_weights(hf_model, cfg)
        else:
            raise ValueError(
                f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
            )

        return state_dict


def fill_missing_keys(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

    This function is assumed to be run before weights are initialized.

    Args:
        model: The model to fill missing keys for
        state_dict: State dict from a pretrained model

    Returns:
        dict: State dict with missing keys filled in
    """
    # Get the default state dict
    default_state_dict = model.state_dict()
    # Get the keys that are missing from the pretrained model
    missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
    # Fill in the missing keys with the default initialization
    for key in missing_keys:
        if "hf_model" in key:
            # Skip keys that are from the HuggingFace model, if loading from HF.
            continue
        if "W_" in key:
            logging.warning(
                "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                    key
                )
            )
        state_dict[key] = default_state_dict[key]
    return state_dict


@dataclasses.dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


# Returns the configuration parameters of the model as a basic Config dataclass
def get_basic_config(model_name: str, **kwargs: Any) -> Config:
    """Returns the configuration parameters of the model as a basic Config dataclass."""
    return Config(
        **{
            k: v
            for k, v in get_pretrained_model_config(model_name, **kwargs).to_dict().items()
            if k
            in [
                "d_model",
                "debug",
                "layer_norm_eps",
                "d_vocab",
                "init_range",
                "n_ctx",
                "d_head",
                "d_mlp",
                "n_heads",
                "n_layers",
            ]
        }
    )
