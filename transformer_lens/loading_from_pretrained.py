# %%
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import einops
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import transformer_lens.utils as utils
from typing import Optional, Dict
import logging
from huggingface_hub import HfApi
import re

# %% The model names used to access the models on the HuggingFace Hub.
OFFICIAL_MODEL_NAMES = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    "facebook/opt-66b",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neox-20b",
    "stanford-crfm/alias-gpt2-small-x21",
    "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/caprica-gpt2-small-x81",
    "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/expanse-gpt2-small-x777",
    "stanford-crfm/arwen-gpt2-medium-x21",
    "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-crfm/celebrimbor-gpt2-medium-x81",
    "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-crfm/eowyn-gpt2-medium-x777",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "EleutherAI/pythia-70m-v0",
    "EleutherAI/pythia-160m-v0",
    "EleutherAI/pythia-410m-v0",
    "EleutherAI/pythia-1b-v0",
    "EleutherAI/pythia-1.4b-v0",
    "EleutherAI/pythia-2.8b-v0",
    "EleutherAI/pythia-6.9b-v0",
    "EleutherAI/pythia-12b-v0",
    "EleutherAI/pythia-70m-deduped-v0",
    "EleutherAI/pythia-160m-deduped-v0",
    "EleutherAI/pythia-410m-deduped-v0",
    "EleutherAI/pythia-1b-deduped-v0",
    "EleutherAI/pythia-1.4b-deduped-v0",
    "EleutherAI/pythia-2.8b-deduped-v0",
    "EleutherAI/pythia-6.9b-deduped-v0",
    "EleutherAI/pythia-12b-deduped-v0",
    "NeelNanda/SoLU_1L_v9_old",
    "NeelNanda/SoLU_2L_v10_old",
    "NeelNanda/SoLU_4L_v11_old",
    "NeelNanda/SoLU_6L_v13_old",
    "NeelNanda/SoLU_8L_v21_old",
    "NeelNanda/SoLU_10L_v22_old",
    "NeelNanda/SoLU_12L_v23_old",
    "NeelNanda/SoLU_1L512W_C4_Code",
    "NeelNanda/SoLU_2L512W_C4_Code",
    "NeelNanda/SoLU_3L512W_C4_Code",
    "NeelNanda/SoLU_4L512W_C4_Code",
    "NeelNanda/SoLU_6L768W_C4_Code",
    "NeelNanda/SoLU_8L1024W_C4_Code",
    "NeelNanda/SoLU_10L1280W_C4_Code",
    "NeelNanda/SoLU_12L1536W_C4_Code",
    "NeelNanda/GELU_1L512W_C4_Code",
    "NeelNanda/GELU_2L512W_C4_Code",
    "NeelNanda/GELU_3L512W_C4_Code",
    "NeelNanda/GELU_4L512W_C4_Code",
    "NeelNanda/Attn_Only_1L512W_C4_Code",
    "NeelNanda/Attn_Only_2L512W_C4_Code",
    "NeelNanda/Attn_Only_3L512W_C4_Code",
    "NeelNanda/Attn_Only_4L512W_C4_Code",
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr",
    "NeelNanda/SoLU_1L512W_Wiki_Finetune",
    "NeelNanda/SoLU_4L512W_Wiki_Finetune",
    "ArthurConmy/redwood_attn_2l",
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf", 
    "llama-65b-hf",
]

# Model Aliases:
MODEL_ALIASES = {
    "NeelNanda/SoLU_1L_v9_old": ["solu-1l-pile", "solu-1l-old"],
    "NeelNanda/SoLU_2L_v10_old": ["solu-2l-pile", "solu-2l-old"],
    "NeelNanda/SoLU_4L_v11_old": ["solu-4l-pile", "solu-4l-old"],
    "NeelNanda/SoLU_6L_v13_old": ["solu-6l-pile", "solu-6l-old"],
    "NeelNanda/SoLU_8L_v21_old": ["solu-8l-pile", "solu-8l-old"],
    "NeelNanda/SoLU_10L_v22_old": ["solu-10l-pile", "solu-10l-old"],
    "NeelNanda/SoLU_12L_v23_old": ["solu-12l-pile", "solu-12l-old"],
    "NeelNanda/SoLU_1L512W_C4_Code": ["solu-1l", "solu-1l-new", "solu-1l-c4-code"],
    "NeelNanda/SoLU_2L512W_C4_Code": ["solu-2l", "solu-2l-new", "solu-2l-c4-code"],
    "NeelNanda/SoLU_3L512W_C4_Code": ["solu-3l", "solu-3l-new", "solu-3l-c4-code"],
    "NeelNanda/SoLU_4L512W_C4_Code": ["solu-4l", "solu-4l-new", "solu-4l-c4-code"],
    "NeelNanda/GELU_1L512W_C4_Code": ["gelu-1l", "gelu-1l-new", "gelu-1l-c4-code"],
    "NeelNanda/GELU_2L512W_C4_Code": ["gelu-2l", "gelu-2l-new", "gelu-2l-c4-code"],
    "NeelNanda/GELU_3L512W_C4_Code": ["gelu-3l", "gelu-3l-new", "gelu-3l-c4-code"],
    "NeelNanda/GELU_4L512W_C4_Code": ["gelu-4l", "gelu-4l-new", "gelu-4l-c4-code"],
    "NeelNanda/Attn_Only_1L512W_C4_Code": [
        "attn-only-1l",
        "attn-only-1l-new",
        "attn-only-1l-c4-code",
    ],
    "NeelNanda/Attn_Only_2L512W_C4_Code": [
        "attn-only-2l",
        "attn-only-2l-new",
        "attn-only-2l-c4-code",
    ],
    "NeelNanda/Attn_Only_3L512W_C4_Code": [
        "attn-only-3l",
        "attn-only-3l-new",
        "attn-only-3l-c4-code",
    ],
    "NeelNanda/Attn_Only_4L512W_C4_Code": [
        "attn-only-4l",
        "attn-only-4l-new",
        "attn-only-4l-c4-code",
    ],
    "NeelNanda/SoLU_6L768W_C4_Code": ["solu-6l", "solu-6l-new", "solu-6l-c4-code"],
    "NeelNanda/SoLU_8L1024W_C4_Code": ["solu-8l", "solu-8l-new", "solu-8l-c4-code"],
    "NeelNanda/SoLU_10L1280W_C4_Code": ["solu-10l", "solu-10l-new", "solu-10l-c4-code"],
    "NeelNanda/SoLU_12L1536W_C4_Code": ["solu-12l", "solu-12l-new", "solu-12l-c4-code"],
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr": [
        "attn-only-2l-demo",
        "attn-only-2l-shortformer-6b-big-lr",
        "attn-only-2l-induction-demo",
        "attn-only-demo",
    ],
    "NeelNanda/SoLU_1L512W_Wiki_Finetune": [
        "solu-1l-wiki",
        "solu-1l-wiki-finetune",
        "solu-1l-finetune",
    ],
    "NeelNanda/SoLU_4L512W_Wiki_Finetune": [
        "solu-4l-wiki",
        "solu-4l-wiki-finetune",
        "solu-4l-finetune",
    ],

    "EleutherAI/pythia-70m": [
        "pythia-70m", 
        "pythia",
        "EleutherAI/pythia-19m",
        "pythia-19m", # EleutherAI renamed this model 
    ],
    "EleutherAI/pythia-160m": [
        "pythia-160m",
        "EleutherAI/pythia-125m",
        "pythia-125m", # EleutherAI renamed this model"        
    ],
    "EleutherAI/pythia-410m": [
        "pythia-410m",
        "EleutherAI/pythia-350m",
        "pythia-350m", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1b": [
        "pythia-1b",
        "EleutherAI/pythia-800m",
        "pythia-800m", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1.4b": [
        "pythia-1.4b",
        "EleutherAI/pythia-1.3b",
        "pythia-1.3b", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-2.8b": [
        "pythia-2.8b",
        "EleutherAI/pythia-2.7b",
        "pythia-2.7b", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-6.9b": [
        "pythia-6.9b",
        "EleutherAI/pythia-6.7b",
        "pythia-6.7b", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-12b": [
        "pythia-12b",
        "EleutherAI/pythia-13b",
        "pythia-13b", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-70m-deduped": [
        "pythia-70m-deduped",
        "EleutherAI/pythia-19m-deduped", # EleutherAI renamed this model 
        "pythia-19m-deduped",
    ],
    "EleutherAI/pythia-160m-deduped": [
        "pythia-160m-deduped",
        "EleutherAI/pythia-125m-deduped", # EleutherAI renamed this model
        "pythia-125m-deduped",
    ],
    "EleutherAI/pythia-410m-deduped": [
        "pythia-410m-deduped",
        "EleutherAI/pythia-350m-deduped", # EleutherAI renamed this model
        "pythia-350m-deduped",
    ],
    "EleutherAI/pythia-1b-deduped": [
        "pythia-1b-deduped",
        "EleutherAI/pythia-800m-deduped", # EleutherAI renamed this model
        "pythia-800m-deduped",
    ],
    "EleutherAI/pythia-1.4b-deduped": [
        "pythia-1.4b-deduped",
        "EleutherAI/pythia-1.3b-deduped", # EleutherAI renamed this model
        "pythia-1.3b-deduped",
    ],
    "EleutherAI/pythia-2.8b-deduped": [
        "pythia-2.8b-deduped",
        "EleutherAI/pythia-2.7b-deduped", # EleutherAI renamed this model
        "pythia-2.7b-deduped",
    ],
    "EleutherAI/pythia-6.9b-deduped": [
        "pythia-6.9b-deduped",
        "EleutherAI/pythia-6.7b-deduped", # EleutherAI renamed this model
        "pythia-6.7b-deduped",
    ],
    "EleutherAI/pythia-12b-deduped": [
        "pythia-12b-deduped",
        "EleutherAI/pythia-13b-deduped", # EleutherAI renamed this model
        "pythia-13b-deduped",
    ],

    "EleutherAI/pythia-70m-v0": [
        "pythia-70m-v0", 
        "pythia-v0",
        "EleutherAI/pythia-19m-v0",
        "pythia-19m-v0", # EleutherAI renamed this model 
    ],
    "EleutherAI/pythia-160m-v0": [
        "pythia-160m-v0",
        "EleutherAI/pythia-125m-v0",
        "pythia-125m-v0", # EleutherAI renamed this model"        
    ],
    "EleutherAI/pythia-410m-v0": [
        "pythia-410m-v0",
        "EleutherAI/pythia-350m-v0",
        "pythia-350m-v0", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1b-v0": [
        "pythia-1b-v0",
        "EleutherAI/pythia-800m-v0",
        "pythia-800m-v0", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1.4b-v0": [
        "pythia-1.4b-v0",
        "EleutherAI/pythia-1.3b-v0",
        "pythia-1.3b-v0", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-2.8b-v0": [
        "pythia-2.8b-v0",
        "EleutherAI/pythia-2.7b-v0",
        "pythia-2.7b-v0", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-6.9b-v0": [
        "pythia-6.9b-v0",
        "EleutherAI/pythia-6.7b-v0",
        "pythia-6.7b-v0", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-12b-v0": [
        "pythia-12b-v0",
        "EleutherAI/pythia-13b-v0",
        "pythia-13b-v0", # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-70m-deduped-v0": [
        "pythia-70m-deduped-v0",
        "EleutherAI/pythia-19m-deduped-v0", # EleutherAI renamed this model 
        "pythia-19m-deduped-v0",
    ],
    "EleutherAI/pythia-160m-deduped-v0": [
        "pythia-160m-deduped-v0",
        "EleutherAI/pythia-125m-deduped-v0", # EleutherAI renamed this model
        "pythia-125m-deduped-v0",
    ],
    "EleutherAI/pythia-410m-deduped-v0": [
        "pythia-410m-deduped-v0",
        "EleutherAI/pythia-350m-deduped-v0", # EleutherAI renamed this model
        "pythia-350m-deduped-v0",
    ],
    "EleutherAI/pythia-1b-deduped-v0": [
        "pythia-1b-deduped-v0",
        "EleutherAI/pythia-800m-deduped-v0", # EleutherAI renamed this model
        "pythia-800m-deduped-v0",
    ],
    "EleutherAI/pythia-1.4b-deduped-v0": [
        "pythia-1.4b-deduped-v0",
        "EleutherAI/pythia-1.3b-deduped-v0", # EleutherAI renamed this model
        "pythia-1.3b-deduped-v0",
    ],
    "EleutherAI/pythia-2.8b-deduped-v0": [
        "pythia-2.8b-deduped-v0",
        "EleutherAI/pythia-2.7b-deduped-v0", # EleutherAI renamed this model
        "pythia-2.7b-deduped-v0",
    ],
    "EleutherAI/pythia-6.9b-deduped-v0": [
        "pythia-6.9b-deduped-v0",
        "EleutherAI/pythia-6.7b-deduped-v0", # EleutherAI renamed this model
        "pythia-6.7b-deduped-v0",
    ],
    "EleutherAI/pythia-12b-deduped-v0": [
        "pythia-12b-deduped-v0",
        "EleutherAI/pythia-13b-deduped-v0", # EleutherAI renamed this model
        "pythia-13b-deduped-v0",
    ],
    
    "gpt2": ["gpt2-small"],
    "distilgpt2": ["distillgpt2", "distill-gpt2", "distil-gpt2", "gpt2-xs"],
    "facebook/opt-125m": ["opt-125m", "opt-small", "opt"],
    "facebook/opt-1.3b": ["opt-1.3b", "opt-medium"],
    "facebook/opt-2.7b": ["opt-2.7b", "opt-large"],
    "facebook/opt-6.7b": ["opt-6.7b", "opt-xl"],
    "facebook/opt-13b": ["opt-13b", "opt-xxl"],
    "facebook/opt-30b": ["opt-30b", "opt-xxxl"],
    "facebook/opt-66b": ["opt-66b", "opt-xxxxl"],
    "EleutherAI/gpt-neo-125M": ["gpt-neo-125M", "gpt-neo-small", "neo-small", "neo"],
    "EleutherAI/gpt-neo-1.3B": ["gpt-neo-1.3B", "gpt-neo-medium", "neo-medium"],
    "EleutherAI/gpt-neo-2.7B": ["gpt-neo-2.7B", "gpt-neo-large", "neo-large"],
    "EleutherAI/gpt-j-6B": ["gpt-j-6B", "gpt-j", "gptj"],
    "EleutherAI/gpt-neox-20b": ["gpt-neox-20b", "gpt-neox", "neox"],
    "stanford-crfm/alias-gpt2-small-x21": [
        "stanford-gpt2-small-a",
        "alias-gpt2-small-x21",
        "gpt2-mistral-small-a",
        "gpt2-stanford-small-a",
    ],
    "stanford-crfm/battlestar-gpt2-small-x49": [
        "stanford-gpt2-small-b",
        "battlestar-gpt2-small-x49",
        "gpt2-mistral-small-b",
        "gpt2-mistral-small-b",
    ],
    "stanford-crfm/caprica-gpt2-small-x81": [
        "stanford-gpt2-small-c",
        "caprica-gpt2-small-x81",
        "gpt2-mistral-small-c",
        "gpt2-stanford-small-c",
    ],
    "stanford-crfm/darkmatter-gpt2-small-x343": [
        "stanford-gpt2-small-d",
        "darkmatter-gpt2-small-x343",
        "gpt2-mistral-small-d",
        "gpt2-mistral-small-d",
    ],
    "stanford-crfm/expanse-gpt2-small-x777": [
        "stanford-gpt2-small-e",
        "expanse-gpt2-small-x777",
        "gpt2-mistral-small-e",
        "gpt2-mistral-small-e",
    ],
    "stanford-crfm/arwen-gpt2-medium-x21": [
        "stanford-gpt2-medium-a",
        "arwen-gpt2-medium-x21",
        "gpt2-medium-small-a",
        "gpt2-stanford-medium-a",
    ],
    "stanford-crfm/beren-gpt2-medium-x49": [
        "stanford-gpt2-medium-b",
        "beren-gpt2-medium-x49",
        "gpt2-medium-small-b",
        "gpt2-stanford-medium-b",
    ],
    "stanford-crfm/celebrimbor-gpt2-medium-x81": [
        "stanford-gpt2-medium-c",
        "celebrimbor-gpt2-medium-x81",
        "gpt2-medium-small-c",
        "gpt2-medium-small-c",
    ],
    "stanford-crfm/durin-gpt2-medium-x343": [
        "stanford-gpt2-medium-d",
        "durin-gpt2-medium-x343",
        "gpt2-medium-small-d",
        "gpt2-stanford-medium-d",
    ],
    "stanford-crfm/eowyn-gpt2-medium-x777": [
        "stanford-gpt2-medium-e",
        "eowyn-gpt2-medium-x777",
        "gpt2-medium-small-e",
        "gpt2-stanford-medium-e",
    ],
    "ArthurConmy/redwood_attn_2l": ["redwood_attn_2l"],
    "llama-7b-hf": ["llama-7b"], 
    "llama-13b-hf": ["llama-13b"],
    "llama-30b-hf": ["llama-30b"],
    "llama-65b-hf": ["llama-65b"],
}

# Sets a default model alias, by convention the first one in the model alias table, else the official name if it has no aliases
DEFAULT_MODEL_ALIASES = [MODEL_ALIASES[name][0] if name in MODEL_ALIASES else name for name in OFFICIAL_MODEL_NAMES]

def make_model_alias_map():
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


def get_official_model_name(model_name: str):
    """
    Returns the official model name for a given model name (or alias).
    """
    model_alias_map = make_model_alias_map()
    official_model_name = model_alias_map.get(model_name.lower(), None)
    if official_model_name is None:
        raise ValueError(
            f"{model_name} not found. Valid official model names (excl aliases): {OFFICIAL_MODEL_NAMES}"
        )
    return official_model_name


def convert_hf_model_config(official_model_name: str):
    """
    Returns the model config for a HuggingFace model, converted to a dictionary
    in the HookedTransformerConfig format.

    Takes the official_model_name as an input.
    """
    # In case the user passed in an alias
    official_model_name = get_official_model_name(official_model_name)
    # Load HuggingFace model config
    if 'llama' not in official_model_name:
        hf_config = AutoConfig.from_pretrained(official_model_name)
        architecture = hf_config.architectures[0]
    else: 
        architecture = "LLaMAForCausalLM"
    if 'llama-7b' in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048,
            "eps": 1e-6, 
            "d_vocab": 32000,
            "act_fn": "silu", 
            "normalization_type": "RMS", 
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32, 
            "final_rms": True, 
            "gated_mlp": True, 
        }
    elif 'llama-13b' in official_model_name:
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048,
            "eps": 1e-6, 
            "d_vocab": 32000,
            "act_fn": "silu", 
            "normalization_type": "RMS", 
            "positional_embedding_type": "rotary",
            "rotary_dim": 5120 // 40, 
            "final_rms": True, 
            "gated_mlp": True, 
        }
    elif 'llama-30b' in official_model_name:
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
            "rotary_dim": 6656 // 52, 
            "final_rms": True, 
            "gated_mlp": True, 
        }
    elif 'llama-65b' in official_model_name:
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
            "final_rms": True, 
            "gated_mlp": True, 
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
            "normalization_type": "LN", 
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    return cfg_dict


def convert_neel_model_config(official_model_name: str):
    """
    Loads the config for a model trained by me (NeelNanda), converted to a dictionary
    in the HookedTransformerConfig format.

    AutoConfig is not supported, because these models are in the HookedTransformer format, so we directly download and load the json.
    """
    official_model_name = get_official_model_name(official_model_name)
    cfg_json: dict = utils.download_file_from_hf(official_model_name, "config.json")
    cfg_dict = {
        "d_model": cfg_json["d_model"],
        "n_layers": cfg_json["n_layers"],
        "d_mlp": cfg_json["d_mlp"],
        "d_head": cfg_json["d_head"],
        "n_heads": cfg_json["n_heads"],
        "n_ctx": cfg_json["n_ctx"],
        "d_vocab": cfg_json["d_vocab"],
        "tokenizer_name": cfg_json["tokenizer_name"],
        "act_fn": cfg_json["act_fn"],
        "attn_only": cfg_json["attn_only"],
        "final_rms": cfg_json.get("final_rms", False),
        "original_architecture": (
            "neel" if "_old" not in official_model_name else "neel-solu-old"
        ),
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
    checkpoint_index: Optional[int] = None,
    checkpoint_value: Optional[int] = None,
    fold_ln: bool = False,
    device: Optional[str] = None,
    n_devices: int = 1,
):
    """Returns the pretrained model config as an HookedTransformerConfig object.

    There are two types of pretrained models: HuggingFace models (where
    AutoModel and AutoConfig work), and models trained by me (NeelNanda) which
    aren't as integrated with HuggingFace infrastructure.

    Args:
        model_name: The name of the model. This can be either the official
            HuggingFace model name, or the name of a model trained by me
            (NeelNanda).
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
        n_devices (int): The number of devices to split the model across. Defaults to 1.

    """
    official_model_name = get_official_model_name(model_name)
    if official_model_name.startswith("NeelNanda") or official_model_name.startswith("ArthurConmy"):
        cfg_dict = convert_neel_model_config(official_model_name)
    else:
        cfg_dict = convert_hf_model_config(official_model_name)
    # Processing common to both model types
    # Remove any prefix, saying the organization who made a model.
    cfg_dict["model_name"] = official_model_name.split("/")[-1]
    # Don't need to initialize weights, we're loading from pretrained
    cfg_dict["init_weights"] = False

    if device is not None:
        cfg_dict["device"] = device
    if fold_ln:
        if cfg_dict["normalization_type"] in ["LN", "LNPre"]:
            cfg_dict["normalization_type"] = "LNPre"
        else:
            logging.warning("Cannot fold in layer norm, normalization_type is not LN.")
            pass

    if checkpoint_index is not None or checkpoint_value is not None:
        checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(
            official_model_name
        )
        cfg_dict["from_checkpoint"] = True
        cfg_dict["checkpoint_label_type"] = checkpoint_label_type
        if checkpoint_index is not None:
            cfg_dict["checkpoint_index"] = checkpoint_index
            cfg_dict["checkpoint_value"] = checkpoint_labels[checkpoint_index]
        elif checkpoint_value is not None:
            assert checkpoint_value in checkpoint_labels, f"Checkpoint value {checkpoint_value} is not in list of available checkpoints"
            cfg_dict["checkpoint_value"] = checkpoint_value
            cfg_dict["checkpoint_index"] = checkpoint_labels.index(checkpoint_value)
    else:
        cfg_dict["from_checkpoint"] = False

    cfg_dict["device"] = device
    cfg_dict["n_devices"] = n_devices

    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    return cfg


def get_num_params_of_pretrained(model_name):
    """
    Returns the number of parameters of a pretrained model, used to filter to only run code for sufficiently small models.
    """
    cfg = get_pretrained_model_config(model_name)
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
PYTHIA_CHECKPOINTS = (
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    + list(range(1000, 143000+1, 1000))
)
# Pythia V1 has log-spaced early checkpoints (see line above), but V0 doesn't
PYTHIA_V0_CHECKPOINTS = list(range(1000, 143000+1, 1000))


def get_checkpoint_labels(model_name: str):
    """Returns the checkpoint labels for a given model, and the label_type
    (step or token). Raises an error for models that are not checkpointed."""
    official_model_name = get_official_model_name(model_name)
    if official_model_name.startswith("stanford-crfm/"):
        return STANFORD_CRFM_CHECKPOINTS, "step"
    elif official_model_name.startswith("EleutherAI/pythia"):
        if "v0" in official_model_name:
            return PYTHIA_V0_CHECKPOINTS, "step"
        else:
            logging.warning("Pythia models on HF were updated on 4/3/23! add '-v0' to model name to access the old models.")
            return PYTHIA_CHECKPOINTS, "step"
    elif official_model_name.startswith("NeelNanda/"):
        api = HfApi()
        files_list = api.list_repo_files(official_model_name)
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
    hf_model=None,
) -> Dict[str, torch.Tensor]:
    """
    Loads in the model weights for a pretrained model, and processes them to
    have the HookedTransformer parameter names and shapes. Supports checkpointed
    models (and expects the checkpoint info to be stored in the config object)

    hf_model: Optionally, a HuggingFace model object. If provided, we will use
    these weights rather than reloading the model.
    """
    official_model_name = get_official_model_name(official_model_name)
    if official_model_name.startswith("NeelNanda") or official_model_name.startswith("ArthurConmy"):
        api = HfApi()
        repo_files = api.list_repo_files(official_model_name)
        if cfg.from_checkpoint:
            file_name = list(
                filter(lambda x: x.endswith(f"{cfg.checkpoint_value}.pth"), repo_files)
            )[0]
        else:
            file_name = list(filter(lambda x: x.endswith("final.pth"), repo_files))[0]
        state_dict = utils.download_file_from_hf(official_model_name, file_name)
        if cfg.original_architecture == "neel-solu-old":
            state_dict = convert_neel_solu_old_weights(state_dict, cfg)
        return state_dict
    else:
        if cfg.from_checkpoint:
            if official_model_name.startswith("stanford-crfm"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name, revision=f"checkpoint-{cfg.checkpoint_value}"
                )
            elif official_model_name.startswith("EleutherAI/pythia"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name, revision=f"step{cfg.checkpoint_value}"
                )
            else:
                raise ValueError(f"Checkpoints for model {official_model_name} are not supported")
        elif hf_model is None:
            if "llama" in official_model_name:
                raise NotImplementedError("Must pass in hf_model for LLaMA models")
            else: 
                hf_model = AutoModelForCausalLM.from_pretrained(official_model_name)

            # Load model weights, and fold in layer norm weights
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
        elif cfg.original_architecture == "LLaMAForCausalLM": 
            state_dict = convert_llama_weights(hf_model, cfg)
        else:
            raise ValueError(
                f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
            )

        return state_dict


# %%
def convert_state_dict(
    state_dict: dict,
    cfg: HookedTransformerConfig,
):
    """Converts a state_dict from a HuggingFace model to a state_dict
    compatible with HookedTransformer."""
    official_model_name = get_official_model_name(official_model_name)

    if cfg["original_architecture"] == "gpt2":
        return convert_gpt2_weights(state_dict, cfg)
    elif cfg["original_architecture"] == "neo":
        return convert_neo_weights(state_dict, cfg)
    elif cfg["original_architecture"] == "gptj":
        return convert_gptj_weights(state_dict, cfg)
    elif cfg["original_architecture"] == "neox":
        return convert_neox_weights(state_dict, cfg)
    elif cfg["original_architecture"] == "opt":
        return convert_opt_weights(state_dict, cfg)
    elif cfg["original_architecture"] == "neel-solu-old":
        return convert_neel_solu_old_weights(state_dict, cfg)
    elif cfg["original_architecture"] == "neel":
        return state_dict
    else:
        raise ValueError(f"Unknown architecture {cfg['original_architecture']}")


# Convert state dicts
def convert_gpt2_weights(gpt2, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = gpt2.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = gpt2.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gpt2.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gpt2.transformer.h[l].ln_1.bias

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = gpt2.transformer.h[l].attn.c_attn.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = gpt2.transformer.h[l].attn.c_attn.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = gpt2.transformer.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = gpt2.transformer.h[l].attn.c_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = gpt2.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = gpt2.transformer.h[l].ln_2.bias

        W_in = gpt2.transformer.h[l].mlp.c_fc.weight
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = gpt2.transformer.h[l].mlp.c_fc.bias

        W_out = gpt2.transformer.h[l].mlp.c_proj.weight
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = gpt2.transformer.h[l].mlp.c_proj.bias
    state_dict[f"unembed.W_U"] = gpt2.lm_head.weight.T

    state_dict["ln_final.w"] = gpt2.transformer.ln_f.weight
    state_dict["ln_final.b"] = gpt2.transformer.ln_f.bias
    return state_dict


def convert_neo_weights(neo, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = neo.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = neo.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = neo.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = neo.transformer.h[l].ln_1.bias

        W_Q = neo.transformer.h[l].attn.attention.q_proj.weight
        W_K = neo.transformer.h[l].attn.attention.k_proj.weight
        W_V = neo.transformer.h[l].attn.attention.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head)

        W_O = neo.transformer.h[l].attn.attention.out_proj.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = neo.transformer.h[
            l
        ].attn.attention.out_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = neo.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = neo.transformer.h[l].ln_2.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = neo.transformer.h[l].mlp.c_fc.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neo.transformer.h[l].mlp.c_fc.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = neo.transformer.h[l].mlp.c_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neo.transformer.h[l].mlp.c_proj.bias
    state_dict["ln_final.w"] = neo.transformer.ln_f.weight
    state_dict["ln_final.b"] = neo.transformer.ln_f.bias

    state_dict["unembed.W_U"] = neo.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return state_dict


def convert_gptj_weights(gptj, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = gptj.transformer.wte.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gptj.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gptj.transformer.h[l].ln_1.bias

        W_Q = gptj.transformer.h[l].attn.q_proj.weight
        W_K = gptj.transformer.h[l].attn.k_proj.weight
        W_V = gptj.transformer.h[l].attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head)

        W_O = gptj.transformer.h[l].attn.out_proj.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model)

        # Layer Norm 1 and 2 are tied.
        state_dict[f"blocks.{l}.ln2.w"] = state_dict[f"blocks.{l}.ln1.w"]
        state_dict[f"blocks.{l}.ln2.b"] = state_dict[f"blocks.{l}.ln1.b"]

        state_dict[f"blocks.{l}.mlp.W_in"] = gptj.transformer.h[l].mlp.fc_in.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = gptj.transformer.h[l].mlp.fc_in.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = gptj.transformer.h[l].mlp.fc_out.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = gptj.transformer.h[l].mlp.fc_out.bias
    state_dict["ln_final.w"] = gptj.transformer.ln_f.weight
    state_dict["ln_final.b"] = gptj.transformer.ln_f.bias

    state_dict["unembed.W_U"] = gptj.lm_head.weight.T
    # Contains a bias, for some reason?
    state_dict["unembed.b_U"] = gptj.lm_head.bias
    return state_dict


def convert_neox_weights(neox, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = neox.gpt_neox.embed_in.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = neox.gpt_neox.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = neox.gpt_neox.layers[l].input_layernorm.bias

        # For some inexplicable reason, NeoX both uses the concatenated QKV
        # matmul of GPT-2 (afaict this has a neglible performance impact) AND
        # has the flattened axis in the DIFFERENT order of (head_index qkv
        # d_head) - this took me an hour to debug...
        W = neox.gpt_neox.layers[l].attention.query_key_value.weight
        W = einops.rearrange(W, "(i qkv h) m->qkv i m h", i=cfg.n_heads, qkv=3)

        # Fold in layer norm weights
        state_dict[f"blocks.{l}.attn.W_Q"] = W[0]
        state_dict[f"blocks.{l}.attn.W_K"] = W[1]
        state_dict[f"blocks.{l}.attn.W_V"] = W[2]

        qkv_bias = neox.gpt_neox.layers[l].attention.query_key_value.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(index qkv head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        # Fold in layer norm biases
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = neox.gpt_neox.layers[l].attention.dense.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = neox.gpt_neox.layers[
            l
        ].attention.dense.bias

        state_dict[f"blocks.{l}.ln2.w"] = neox.gpt_neox.layers[
            l
        ].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = neox.gpt_neox.layers[
            l
        ].post_attention_layernorm.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = neox.gpt_neox.layers[
            l
        ].mlp.dense_h_to_4h.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neox.gpt_neox.layers[
            l
        ].mlp.dense_h_to_4h.bias

        state_dict[f"blocks.{l}.mlp.W_out"] = neox.gpt_neox.layers[
            l
        ].mlp.dense_4h_to_h.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neox.gpt_neox.layers[
            l
        ].mlp.dense_4h_to_h.bias
    state_dict["ln_final.w"] = neox.gpt_neox.final_layer_norm.weight
    state_dict["ln_final.b"] = neox.gpt_neox.final_layer_norm.bias

    state_dict["unembed.W_U"] = neox.embed_out.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return state_dict

def convert_llama_weights(llama, cfg: HookedTransformerConfig): 
    state_dict = {}

    state_dict["embed.W_E"] = llama.model.embed_tokens.weight

    # llama has no biases anywhere and deals with everything else roughly like
    # GPTNeoX with different names
    
    for l in range(cfg.n_layers):

        state_dict[f"blocks.{l}.ln1.w"] = llama.model.layers[l].input_layernorm.weight

        W_Q = llama.model.layers[l].self_attn.q_proj.weight
        W_K = llama.model.layers[l].self_attn.k_proj.weight
        W_V = llama.model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head)

        W_O = llama.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model)

        state_dict[f"blocks.{l}.ln2.w"] = llama.model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = llama.model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = llama.model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp)

        state_dict[f"blocks.{l}.mlp.W_out"] = llama.model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model)
    
    state_dict["ln_final.w"] = llama.model.norm.weight

    state_dict["unembed.W_U"] = llama.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)

    return state_dict

def convert_opt_weights(opt, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = opt.model.decoder.embed_tokens.weight
    state_dict["pos_embed.W_pos"] = opt.model.decoder.embed_positions.weight[2:, :]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = opt.model.decoder.layers[
            l
        ].self_attn_layer_norm.weight
        state_dict[f"blocks.{l}.ln1.b"] = opt.model.decoder.layers[
            l
        ].self_attn_layer_norm.bias

        W_Q = opt.model.decoder.layers[l].self_attn.q_proj.weight
        W_K = opt.model.decoder.layers[l].self_attn.k_proj.weight
        W_V = opt.model.decoder.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(
            W_Q,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )
        W_K = einops.rearrange(
            W_K,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )
        W_V = einops.rearrange(
            W_V,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        q_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.q_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )
        k_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.k_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )
        v_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.v_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = q_bias
        state_dict[f"blocks.{l}.attn.b_K"] = k_bias
        state_dict[f"blocks.{l}.attn.b_V"] = v_bias

        W_O = opt.model.decoder.layers[l].self_attn.out_proj.weight
        W_O = einops.rearrange(
            W_O,
            "d_model (index d_head)->index d_head d_model",
            index=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = opt.model.decoder.layers[
            l
        ].self_attn.out_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = opt.model.decoder.layers[
            l
        ].final_layer_norm.weight
        state_dict[f"blocks.{l}.ln2.b"] = opt.model.decoder.layers[
            l
        ].final_layer_norm.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = opt.model.decoder.layers[l].fc1.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = opt.model.decoder.layers[l].fc2.weight.T

        state_dict[f"blocks.{l}.mlp.b_in"] = opt.model.decoder.layers[l].fc1.bias
        state_dict[f"blocks.{l}.mlp.b_out"] = opt.model.decoder.layers[l].fc2.bias
    state_dict[f"ln_final.w"] = opt.model.decoder.final_layer_norm.weight
    state_dict[f"ln_final.b"] = opt.model.decoder.final_layer_norm.bias
    state_dict["unembed.W_U"] = opt.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return state_dict


def convert_neel_solu_old_weights(state_dict: dict, cfg: HookedTransformerConfig):
    """
    Converts the weights of my old SoLU models to the HookedTransformer format.
    Takes as input a state dict, *not* a model object.

    There are a bunch of dumb bugs in the original code, sorry!

    Models 1L, 2L, 4L and 6L have left facing weights (ie, weights have shape
    [dim_out, dim_in]) while HookedTransformer does right facing (ie [dim_in,
    dim_out]).

    8L has *just* a left facing W_pos, the rest right facing.

    And some models were trained with
    """
    # Early models have left facing W_pos
    reverse_pos = cfg.n_layers <= 8

    # Models prior to 8L have left facing everything (8L has JUST left facing W_pos - sorry! Stupid bug)
    reverse_weights = cfg.n_layers <= 6

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("norm", "ln")
        if k.startswith("ln."):
            k = k.replace("ln.", "ln_final.")
        new_state_dict[k] = v

    if reverse_pos:
        new_state_dict["pos_embed.W_pos"] = new_state_dict["pos_embed.W_pos"].T
    if reverse_weights:
        for k, v in new_state_dict.items():
            if "W_" in k and "W_pos" not in k:
                new_state_dict[k] = v.transpose(-2, -1)
    return new_state_dict

# %%
