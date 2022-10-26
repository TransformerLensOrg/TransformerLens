# %%
from easy_transformer import EasyTransformerConfig
import einops
import torch
from collections import defaultdict
from transformers import AutoConfig
import easy_transformer.utils as utils

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
    "EleutherAI/pythia-19m",
    "EleutherAI/pythia-125m",
    "EleutherAI/pythia-350m",
    "EleutherAI/pythia-800m",
    "EleutherAI/pythia-1.3b",
    "EleutherAI/pythia-6.7b",
    "EleutherAI/pythia-13b",
    "EleutherAI/pythia-125m-deduped",
    "EleutherAI/pythia-800m-deduped",
    "EleutherAI/pythia-1.3b-deduped",
    "EleutherAI/pythia-6.7b-deduped",
    'NeelNanda/SoLU_1L_v9_old',
    'NeelNanda/SoLU_2L_v10_old',
    'NeelNanda/SoLU_4L_v11_old',
    'NeelNanda/SoLU_6L_v13_old',
    'NeelNanda/SoLU_8L_v21_old',
    'NeelNanda/SoLU_10L_v22_old',
    "NeelNanda/SoLU_1L512W_C4_Code",
    "NeelNanda/SoLU_2L512W_C4_Code",
    "NeelNanda/SoLU_3L512W_C4_Code",
    "NeelNanda/SoLU_4L512W_C4_Code",
    "NeelNanda/SoLU_8L1024W_C4_Code",
    "NeelNanda/GELU_1L512W_C4_Code",
    "NeelNanda/GELU_2L512W_C4_Code",
    "NeelNanda/GELU_3L512W_C4_Code",
    "NeelNanda/GELU_4L512W_C4_Code",
    "NeelNanda/Attn_Only_1L512W_C4_Code",
    "NeelNanda/Attn_Only_2L512W_C4_Code",
    "NeelNanda/Attn_Only_3L512W_C4_Code",
    # "NeelNanda/Attn_Only_4L512W_C4_Code",
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr",
]

# Model Aliases:
MODEL_ALIASES = {
    'NeelNanda/SoLU_1L_v9_old': ["solu-1l-old", "solu-1l-pile"],
    'NeelNanda/SoLU_2L_v10_old': ["solu-2l-old", "solu-2l-pile"],
    'NeelNanda/SoLU_4L_v11_old': ["solu-4l-old", "solu-4l-pile"],
    'NeelNanda/SoLU_6L_v13_old': ["solu-6l-old", "solu-6l-pile"],
    'NeelNanda/SoLU_8L_v21_old': ["solu-8l-old", "solu-8l-pile"],
    'NeelNanda/SoLU_10L_v22_old': ["solu-10l-old", "solu-10l-pile"],
    "NeelNanda/SoLU_1L512W_C4_Code": ["solu-1l", "solu-1l-new", "solu-1l-c4-code"],
    "NeelNanda/SoLU_2L512W_C4_Code": ["solu-2l", "solu-2l-new", "solu-2l-c4-code"],
    "NeelNanda/SoLU_3L512W_C4_Code": ["solu-3l", "solu-3l-new", "solu-3l-c4-code"],
    "NeelNanda/SoLU_4L512W_C4_Code": ["solu-4l", "solu-4l-new", "solu-4l-c4-code"],
    "NeelNanda/SoLU_8L1024W_C8_Code": ["solu-8l", "solu-8l-new", "solu-8l-c4-code"],
    "NeelNanda/GELU_1L512W_C4_Code": ["gelu-1l", "gelu-1l-new", "gelu-1l-c4-code"],
    "NeelNanda/GELU_2L512W_C4_Code": ["gelu-2l", "gelu-2l-new", "gelu-2l-c4-code"],
    "NeelNanda/GELU_3L512W_C4_Code": ["gelu-3l", "gelu-3l-new", "gelu-3l-c4-code"],
    "NeelNanda/GELU_4L512W_C4_Code": ["gelu-4l", "gelu-4l-new", "gelu-4l-c4-code"],
    "NeelNanda/Attn_Only_1L512W_C4_Code": ["attn-only-1l", "attn-only-1l-new", "attn-only-1l-c4-code"],
    "NeelNanda/Attn_Only_2L512W_C4_Code": ["attn-only-2l", "attn-only-2l-new", "attn-only-2l-c4-code"],
    "NeelNanda/Attn_Only_3L512W_C4_Code": ["attn-only-3l", "attn-only-3l-new", "attn-only-3l-c4-code"],
    # "NeelNanda/Attn_Only_4L512W_C4_Code": ["attn-only-4l", "attn-only-4l-new", "attn-only-4l-c4-code"],
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr": ["attn-only-2l-shortformer-6b-big-lr", "attn-only-2l-induction-demo", "attn-only-demo", "attn-only-2l-demo"],
    "EleutherAI/pythia-19m": ["pythia-19m", "pythia"],
    "EleutherAI/pythia-125m": ["pythia-125m",],
    "EleutherAI/pythia-350m": ["pythia-350m",],
    "EleutherAI/pythia-800m": ["pythia-800m",],
    "EleutherAI/pythia-1.3b": ["pythia-1.3b",],
    "EleutherAI/pythia-6.7b": ["pythia-6.7b",],
    "EleutherAI/pythia-13b": ["pythia-13b",],
    "EleutherAI/pythia-125m-deduped": ["pythia-125m-deduped",],
    "EleutherAI/pythia-800m-deduped": ["pythia-800m-deduped",],
    "EleutherAI/pythia-1.3b-deduped": ["pythia-1.3b-deduped",],
    "EleutherAI/pythia-6.7b-deduped": ["pythia-6.7b-deduped",],
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
    "EleutherAI/gpt-j-6B": ["gpt-j-6B", "gpt-j", "j"],
    "EleutherAI/gpt-neox-20b": ["gpt-neox-20b", "gpt-neox", "neox"],
    "stanford-crfm/alias-gpt2-small-x21": ["alias-gpt2-small-x21", "gpt2-mistral-small-a", "gpt2-stanford-small-a", "stanford-gpt2-small-a"],
    "stanford-crfm/battlestar-gpt2-small-x49": ["battlestar-gpt2-small-x49", "gpt2-mistral-small-b", "gpt2-mistral-small-b", "stanford-gpt2-small-b"],
    "stanford-crfm/caprica-gpt2-small-x81": ["caprica-gpt2-small-x81", "gpt2-mistral-small-c", "gpt2-stanford-small-c", "stanford-gpt2-small-c"],
    "stanford-crfm/darkmatter-gpt2-small-x343": ["darkmatter-gpt2-small-x343", "gpt2-mistral-small-d", "gpt2-mistral-small-d", "stanford-gpt2-small-d"],
    "stanford-crfm/expanse-gpt2-small-x777": ["expanse-gpt2-small-x777", "gpt2-mistral-small-e", "gpt2-mistral-small-e", "stanford-gpt2-small-e"],
    "stanford-crfm/arwen-gpt2-medium-x21": ["arwen-gpt2-medium-x21", "gpt2-medium-small-a", "gpt2-stanford-medium-a", "stanford-gpt2-medium-a"],
    "stanford-crfm/beren-gpt2-medium-x49": ["beren-gpt2-medium-x49", "gpt2-medium-small-b", "gpt2-stanford-medium-b", "stanford-gpt2-medium-b"],
    "stanford-crfm/celebrimbor-gpt2-medium-x81": ["celebrimbor-gpt2-medium-x81", "gpt2-medium-small-c", "gpt2-medium-small-c", "stanford-gpt2-medium-c"],
    "stanford-crfm/durin-gpt2-medium-x343": ["durin-gpt2-medium-x343", "gpt2-medium-small-d", "gpt2-stanford-medium-d", "stanford-gpt2-medium-d"],
    "stanford-crfm/eowyn-gpt2-medium-x777": ["eowyn-gpt2-medium-x777", "gpt2-medium-small-e", "gpt2-stanford-medium-e", "stanford-gpt2-medium-e"],
}

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
        raise ValueError(f"{model_name} not found. Valid model names: {OFFICIAL_MODEL_NAMES}")
    return official_model_name

def convert_hf_model_config(official_model_name: str):
    """
    Returns the model config for a HuggingFace model, converted to an
    EasyTransformerConfig object.

    Takes the official_model_name as an input
    """
    # In case the user passed in an alias
    official_model_name = get_official_model_name(official_model_name)
    # Load HuggingFace model config
    hf_config = AutoConfig.from_pretrained(official_model_name)
    architecture = hf_config.architectures[0]
    if architecture == "GPTNeoForCausalLM":
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
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    cfg_dict["original_architecture"] = architecture
    cfg = EasyTransformerConfig.from_dict(cfg_dict)
    return cfg


def convert_neel_model_config(official_model_name: str):
    """ 
    Loads the config for a model trained by me (NeelNanda), and converts it to
    an EasyTransformerConfig object.

    AutoConfig is not supported, because these models are in the EasyTransformer format, so we directly download and load the json.
    """
    official_model_name = get_official_model_name(official_model_name)
    cfg_json = utils.download_file_from_hf(official_model_name, "config.json")
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
        "original_architecture": "neel-solu" if "_old" not in official_model_name else "neel-solu-old",
    }
    if "normalization" in cfg_json:
        cfg_dict["normalization_type"] = cfg_json["normalization"]
    else:
        cfg_dict["normalization_type"] = cfg_json["normalization_type"]
    if "shortformer_pos" in cfg_json:
        cfg_dict["positional_embedding_type"] = "shortformer" if cfg_json["shortformer_pos"] else "standard"
    else:
        cfg_dict["positional_embedding_type"] = "standard"
    return EasyTransformerConfig.from_dict(cfg_dict)

def get_pretrained_model_config(model_name: str):
    """ Returns model config as an EasyTransformerConfig object"""
    official_model_name = get_official_model_name(model_name)
    if official_model_name.startswith("NeelNanda"):
        return convert_neel_model_config(official_model_name)
    else:
        return convert_hf_model_config(official_model_name)


# %% Load checkpointed model state dicts The steps for which there are
# checkpoints in the stanford crfm models
STANFORD_CRFM_CHECKPOINTS = (
    list(range(0, 100, 10))
    + list(range(100, 2000, 50))
    + list(range(2000, 20000, 100))
    + list(range(20000, 400000 + 1, 1000))
)

def get_model_checkpoint_list(official_model_name: str):
    if official_model_name.startswith("stanford-crfm/"):
        return STANFORD_CRFM_CHECKPOINTS
    else:
        raise NotImplementedError(f"Model {official_model_name} is not supported by this function yet.")

def get_model_checkpoint_state_dict(official_model_name: str):
    # if official_model_name.startswith("stanford-crfm/"): return
    #     torch.hub.load_state_dict_from_url(
    #         f"https://huggingface.co/{official_model_name}/resolve/main/checkpoint-{checkpoint}.bin",
    #         map_location="cpu", ) else:
    raise NotImplementedError(f"Model {official_model_name} is not supported by this function yet.")



# %% Loading state dicts

def get_model_state_dict(official_model_name: str):
    pass


# %%
def convert_state_dict(
    state_dict: dict, 
    official_model_name: str,
    ):
    pass
# Convert state dicts
def convert_gpt2_weights(gpt2, cfg: EasyTransformerConfig):
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

def convert_neo_weights(neo, cfg: EasyTransformerConfig):
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

def convert_gptj_weights(gptj, cfg: EasyTransformerConfig):
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

def convert_neox_weights(neox, cfg: EasyTransformerConfig):
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
        state_dict[f"blocks.{l}.attn.b_O"] = neox.gpt_neox.layers[l].attention.dense.bias
        
        state_dict[f"blocks.{l}.ln2.w"] = neox.gpt_neox.layers[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = neox.gpt_neox.layers[l].post_attention_layernorm.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = neox.gpt_neox.layers[l].mlp.dense_h_to_4h.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neox.gpt_neox.layers[l].mlp.dense_h_to_4h.bias
        
        state_dict[f"blocks.{l}.mlp.W_out"] = neox.gpt_neox.layers[l].mlp.dense_4h_to_h.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neox.gpt_neox.layers[l].mlp.dense_4h_to_h.bias
    state_dict["ln_final.w"] = neox.gpt_neox.final_layer_norm.weight
    state_dict["ln_final.b"] = neox.gpt_neox.final_layer_norm.bias
    
    state_dict["unembed.W_U"] = neox.embed_out.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return state_dict

def convert_opt_weights(opt, cfg: EasyTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = opt.model.decoder.embed_tokens.weight
    state_dict["pos_embed.W_pos"] = opt.model.decoder.embed_positions.weight[2:, :]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = opt.model.decoder.layers[l].self_attn_layer_norm.weight
        state_dict[f"blocks.{l}.ln1.b"] = opt.model.decoder.layers[l].self_attn_layer_norm.bias
        
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

        state_dict[f"blocks.{l}.ln2.w"] = opt.model.decoder.layers[l].final_layer_norm.weight
        state_dict[f"blocks.{l}.ln2.b"] = opt.model.decoder.layers[l].final_layer_norm.bias
        
        state_dict[f"blocks.{l}.mlp.W_in"] = opt.model.decoder.layers[l].fc1.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = opt.model.decoder.layers[l].fc2.weight.T
        
        state_dict[f"blocks.{l}.mlp.b_in"] = opt.model.decoder.layers[l].fc1.bias
        state_dict[f"blocks.{l}.mlp.b_out"] = opt.model.decoder.layers[l].fc2.bias
    state_dict[f"ln_final.w"] = opt.model.decoder.final_layer_norm.weight
    state_dict[f"ln_final.b"] = opt.model.decoder.final_layer_norm.bias
    state_dict["unembed.W_U"] = opt.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return state_dict