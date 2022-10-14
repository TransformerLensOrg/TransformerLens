from mimetypes import init
from typing import Callable, Union, List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import logging
import tqdm.auto as tqdm

from functools import *

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer import EasyTransformerConfig

from easy_transformer.caching import (
    EasyTransformerKeyValueCache,
    EasyTransformerKeyValueCacheEntry,
)

from easy_transformer.components import *
import easy_transformer.weight_conversion as weight_conversion
from easy_transformer.utils import lm_cross_entropy_loss, sample_logits


"""
TODO: Add Bloom, GPT-J and GPT-NeoX
EleutherAI/gpt-j-6B
EleutherAI/gpt-neox-20b
bloom-350m
bloom-760m
bloom-1b3
bloom-2b5
bloom-6b3
bloom (176B parameters)
https://huggingface.co/docs/transformers/model_doc/bloom
"""

# Full transformer
class EasyTransformer(HookedRootModule):
    """
    This class implements a full Transformer using the components in ./components.py, with
    HookPoints on every interesting activation. It inherits from HookedRootModule.

    It can have a pretrained Transformer's weights automatically loaded in via the EasyTransformer.from_pretrained class method. It can also be instantiated with randomly initialized weights via __init__ and being passed a dict or EasyTransformerConfig object. 
    """
    
    VALID_PRETRAINED_MODEL_NAMES = weight_conversion.VALID_PRETRAINED_MODEL_NAMES
    PRETRAINED_MODEL_NAMES_DICT = weight_conversion.PRETRAINED_MODEL_NAMES_DICT
    STANFORD_CRFM_CHECKPOINTS = weight_conversion.STANFORD_CRFM_CHECKPOINTS

    def __init__(
        self,
        cfg,
        tokenizer = None,
        move_to_device = True,
    ):
        """
        Model initialization. Note that if you want to load the model from pretrained weights, you should use the EasyTransformer.from_pretrained() class method instead of this one.

        cfg Union[EasyTransformerConfig, Dict]: The config to use for the
            model. 
        tokenizer (*optional): The tokenizer to use for the model. If not
            provided, it is inferred from cfg.tokenizer_name or initialized to None. 
            If None, then the model cannot be passed strings, and d_vocab must be explicitly set.
        move_to_device (bool): Whether to move the model to the device specified in cfg.
            device.
        """
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError("Please pass in a config dictionary or EasyTransformerConfig object. If you want to load a pretrained model, use EasyTransformer.from_pretrained() instead.")
        self.cfg = cfg
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if self.cfg.tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens directly. In this case, we don't need a tokenizer.
            self.tokenizer = None
        
        if not self.cfg.d_vocab:
            # If we have a tokenizer, vocab size can be inferred from it.
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1

        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.cfg, block_index)
                for block_index in range(self.cfg.n_layers)
            ]
        )
        
        if self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )
        self.unembed = Unembed(self.cfg)

        if self.cfg.init_weights:
            self.init_weights()

        if move_to_device:
            self.to(self.cfg.device)
        
        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    def forward(
        self, 
        input: Union[str, torch.Tensor], 
        return_type: Optional[str] = "logits", 
        prepend_bos: bool = True, 
        past_kv_cache: Optional[EasyTransformerKeyValueCache] = None
        ) -> Union[None, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Input is either a batch of tokens ([batch, pos]) or a text string, a string is automatically tokenized to a batch of a single element. The prepend_bos flag only applies when inputting a text string.

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), 'logits' (return logits), 'loss' (return cross-entropy loss), 'both' (return logits and loss)
        """
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input
        assert isinstance(tokens, torch.Tensor)
        # If we're doing caching, then we reuse keys and values from previous runs, as that's the only 
        # way that past activations will affect the final logits. The cache contains those so we don't 
        # need to recompute them. This is useful for generating text. As we have absolute positional 
        # encodings, to implement this we have a `pos_offset` variable, defaulting to zero, which says 
        # to offset which positional encodings are used (cached keys and values were calculated with 
        # their own positional encodings).
        if past_kv_cache is None:
            pos_offset = 0
        else:
            batch_size, ctx_length = tokens.shape
            cached_batch_size, cache_ctx_length, num_heads_in_cache, d_head_in_cache = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            assert num_heads_in_cache == self.cfg.n_heads
            assert d_head_in_cache == self.cfg.d_head
            # If we want to generate from the empty string, we'd pass in an empty cache, so we need to handle that case
            assert cache_ctx_length == 0 or ctx_length == 1, "Pass in one token at a time after loading cache"
            pos_offset = cache_ctx_length
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(
            self.pos_embed(tokens, pos_offset)
        )  # [batch, pos, d_model]
        if self.cfg.positional_embedding_type != "shortformer":
            residual = embed + pos_embed  # [batch, pos, d_model]
            shortformer_pos_embed = None
        else:
            # If we're using shortformer style attention, we don't add the positional embedding to the residual stream. See EasyTransformerConfig for details
            residual = embed
            shortformer_pos_embed = pos_embed
        for i, block in enumerate(self.blocks):
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(
                residual, 
                past_kv_cache_entry = past_kv_cache[i] if past_kv_cache is not None else None, # Cache is contains a list of EasyTransformerKeyValueCache objects, one for each block
                shortformer_pos_embed = shortformer_pos_embed
            )  # [batch, pos, d_model]
        if return_type is None:
            return None
        else:
            if self.cfg.normalization_type is not None:
                residual = self.ln_final(residual)  # [batch, pos, d_vocab]
            logits = self.unembed(residual)  # [batch, pos, d_vocab]
            if return_type == "logits":
                return logits
            else:
                loss = lm_cross_entropy_loss(logits, tokens)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return {"logits": logits, "loss": loss}
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None
                
    def set_tokenizer(self, tokenizer):
        """
        Sets the tokenizer to use for this model.
        tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer
        """
        assert isinstance(tokenizer, PreTrainedTokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def to_tokens(self, input, prepend_bos=True):
        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
        if prepend_bos:
            if isinstance(input, str):
                input = self.tokenizer.bos_token + input
            else:
                input = [self.tokenizer.bos_token + string for string in input]
        return self.tokenizer(input, return_tensors="pt", padding=True)["input_ids"].to(self.cfg.device)
    
    def to_str_tokens(
        self, 
        input: Union[str, torch.Tensor, list], 
        prepend_bos: bool = True
        ):
        """Method to map text, a list of text or tokens to a list of tokens as strings

        Args:
            input (Union[str, list, torch.Tensor]): The input - either a string or a tensor of tokens. If tokens, should be a tensor of shape [pos] or [1, pos]
            prepend_bos (bool, optional): Whether to prepend a BOS token. Only applies if input is a string. Defaults to True.

        Returns:
            str_tokens: List of individual tokens as strings
        """
        if isinstance(input, list):
            return list(map(lambda tokens: self.to_str_tokens(tokens, prepend_bos), input))
        elif isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)[0]
        elif isinstance(input, torch.Tensor):
            tokens = input
            tokens = tokens.squeeze() # Get rid of a trivial batch dimension
            assert tokens.dim() == 1, f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        else:
            raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
        str_tokens = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        return str_tokens

    @classmethod
    def from_pretrained(cls, 
                        model_name: str, 
                        fold_ln = True, 
                        center_writing_weights = True, 
                        center_unembed = True,
                        checkpoint = None,
                        hf_model = None,
                        device = None,
                        **kwargs):
        """Class method to load a pretrained model from HuggingFace and to automatically convert and load those weights into EasyTransformer format.
        
        See fold_layer_norm for more details on the folding and centering.

        Args:
            model_name (str): The model name - must be in VALID_MODEL_NAMES
            fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the 
                subsequent linear layer. This does not change the computation. Defaults to True.
            center_writing_weights (bool, optional): Whether to center weights writing to   
                the residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the computation. Defaults to True.
            center_unembed (bool, optional): Whether to center W_U (ie set mean to be zero). 
                Softmax is translation invariant so this doesn't affect log probs or loss, but does change logits. Defaults to True.
            keep_original_model (bool, optional): Whether to delete the model loaded from    HuggingFace (stored as model.hf_model). Defaults to False.
            device (str, optional): The device to load the model onto. By default will load to CUDA if available, else CPU
        """
        assert (
            (model_name in cls.VALID_PRETRAINED_MODEL_NAMES) or (model_name in cls.PRETRAINED_MODEL_NAMES_DICT)
        ), f"Invalid model name: {model_name}. Valid model names are: {cls.VALID_PRETRAINED_MODEL_NAMES}"

        # hf_model_name is the model's name on HuggingFace
        if model_name in cls.PRETRAINED_MODEL_NAMES_DICT:
            hf_model_name = cls.PRETRAINED_MODEL_NAMES_DICT[model_name]
        else:
            hf_model_name = model_name
        # The model family (eg "gpt2" or "neo")
        model_family = cls.get_model_family(hf_model_name)
        
        if hf_model is None:
            if checkpoint is not None:
                if "stanford" not in model_name:
                    logging.warning(
                        f"Loading checkpoints is not supported for the model {model_name}. Loading without checkpoints"
                    )
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        hf_model_name
                    )
                else:
                    assert (
                        checkpoint in cls.STANFORD_CRFM_CHECKPOINTS
                    ), f"Checkpoint {checkpoint} is not valid. Available checkpoints are {cls.STANFORD_CRFM_CHECKPOINTS}"
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        hf_model_name, revision=f"checkpoint-{checkpoint}"
                    )
            else:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name
                )

        cfg = cls.convert_hf_config(
            hf_model.config, model_family=model_family
        )
        if device is not None:
            cfg.device = device
        cfg.checkpoint = checkpoint
        cfg.model_family = model_family
        cfg.model_name = model_name

        cfg.normalization_type = "LNPre" if fold_ln else "LN"
        cfg.tokenizer_name = hf_model_name
        cfg.init_weights = False
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = cls(cfg, **kwargs)

        # Load model weights, and fold in layer norm weights
        if model_family == "gpt2":
            state_dict = weight_conversion.convert_gpt2_weights(hf_model, model.cfg)
        elif model_family == "mistral":
            # Stanford (Mistral) models have identical structure to GPT-2, but scale attention scores by 1/(layer_id+1) before softmax.
            state_dict = weight_conversion.convert_gpt2_weights(hf_model, model.cfg)
        elif model_family == "neo":
            state_dict = weight_conversion.convert_neo_weights(hf_model, model.cfg)
        elif model_family == "opt":
            state_dict = weight_conversion.convert_opt_weights(hf_model, model.cfg)
        else:
            raise ValueError(f"Loading weights from this model family is not currently supported: {model_family}, generated from model name {model_name}. Feel free to open an issue on GitHub to request this feature.")
        
        model.load_and_process_state_dict(state_dict, 
                        fold_ln=fold_ln, 
                        center_writing_weights=center_writing_weights, 
                        center_unembed=center_unembed,
                        move_dict_to_device=True)
        return model

    @classmethod
    def get_model_family(cls, model_name):
        if "stanford" in model_name:
            return "mistral"
        elif "gpt2" in model_name and "stanford" not in model_name:
            return "gpt2"
        elif "opt" in model_name:
            return "opt"
        elif model_name == "EleutherAI/gpt-neox-20b":
            return "neox"
        elif model_name == "EleutherAI/gpt-j-6B":
            return "gptj"
        elif model_name in ['EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B',]:
            # Important to exclude GPT-J and GPT-NeoX, they have different config.
            return "neo"
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    @classmethod
    def convert_hf_config(cls, hf_config, model_family):
        cfg_dict = {}
        if model_family == "neo":
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
        elif model_family == "gpt2":
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
                "scale_attn_by_inverse_layer_idx": False,
            }
        elif model_family == "mistral":
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
                "scale_attn_by_inverse_layer_idx": True,
            }
        elif model_family == "opt":
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
        else:
            raise NotImplementedError
        cfg = EasyTransformerConfig.from_dict(cfg_dict)
        return cfg

    def init_weights(self):
        """
        Initialize weights matrices with a normal of std=initializer_range (default=0.02). This roughly follows the GPT-2 paper's scheme (but with truncation, and not halving the std for W_pos).

        LayerNorm weights are already initialized to 1.0, and all biases are initialized to 0.0 (including LayerNorm), so this just initializes weight matrices. 
        
        Weight matrices are set to empty by default (to save space + compute, since they're the bulk of the parameters), so it is important to call this if you are not loading in pretrained weights! Note that this function assumes that weight names being with W_

        Set seed here to ensure determinism.

        This does NOT follow the PyTorch scheme, which as far as I can tell is super out of date but no one has gotten round to updating it?
        https://github.com/pytorch/pytorch/issues/18182
        
        PyTorch Transformers are especially bad - TransformerEncoder initializes all layers to the exact same weights?! https://github.com/pytorch/pytorch/issues/72253

        The best paper I've found on transformer initialization is the muP paper, but haven't integrated those ideas yet: https://arxiv.org/abs/2203.03466
        """
        
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        
        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=self.cfg.initializer_range)
    
    def load_and_process_state_dict(self, 
                                    state_dict: Dict[str, torch.Tensor], 
                                    fold_ln: bool=True, 
                                    center_writing_weights: bool = True, 
                                    center_unembed: bool = True,
                                    move_dict_to_device: bool = True):
        """Method to load a state dict into the model, and to apply processing to simplify it. The state dict is assumed to be in the EasyTransformer format.
        
        See fold_layer_norm for more details on the folding and centering.

        Args:
            state_dict (dict): The state dict of the model, in EasyTransformer format
            fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the   
                subsequent linear layer. This does not change the computation. Defaults to True.
            center_writing_weights (bool, optional): Whether to center weights writing to the 
                residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the computation. Defaults to True.
            center_unembed (bool, optional): Whether to center W_U (ie set mean to be zero). 
                Softmax is translation invariant so this doesn't affect log probs or loss, but does change logits. Defaults to True.
            move_dict_to_device (bool, optional): Whether to move the state dict to the device of the model. Defaults to True.
        """
        if move_dict_to_device:
            state_dict = {k: v.to(self.cfg.device) for k, v in state_dict.items()}
        state_dict = self.fill_missing_keys(state_dict)
        if fold_ln:
            state_dict = self.fold_layer_norm(state_dict)
        if center_writing_weights:
            state_dict = self.center_writing_weights(state_dict)
        if center_unembed:
            state_dict = self.center_unembed(state_dict)
        self.load_state_dict(state_dict)


    def fill_missing_keys(self, state_dict):
        """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

        This function is assumed to be run before weights are initialized.

        Args:
            state_dict (dict): State dict from a pretrained model

        Returns:
            dict: State dict with missing keys filled in
        """
        # Get the default state dict
        default_state_dict = self.state_dict()
        # Get the keys that are missing from the pretrained model
        missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
        # Fill in the missing keys with the default initialization
        for key in missing_keys:
            if 'hf_model' in key:
                # Skip keys that are from the HuggingFace model, if loading from HF.
                continue
            if 'W_' in key:
                logging.warning("Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(key))
            state_dict[key] = default_state_dict[key]
        return state_dict
    
    def fold_layer_norm(self, state_dict: Dict[str, torch.Tensor]):
        """Takes in a state dict from a pretrained model, formatted to be consistent with EasyTransformer but with LayerNorm weights and biases. Folds these into the neighbouring weights. See EasyTransformerConfig for more details

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model
        """
        for l in range(self.cfg.n_layers):
            # Fold ln1 into attention - it's important to fold biases first, 
            # since biases depend on weights but not vice versa
            # The various indexing is just to broadcast ln.b and ln.w along every axis other than d_model. Each weight matrix right multiplies.
            # To fold in the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need to sum along axis -2, which is the residual stream space axis.
            state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[f"blocks.{l}.attn.b_Q"] + (state_dict[f"blocks.{l}.attn.W_Q"] * state_dict[f"blocks.{l}.ln1.b"][None, :, None]).sum(-2)
            state_dict[f"blocks.{l}.attn.b_K"] = state_dict[f"blocks.{l}.attn.b_K"] + (state_dict[f"blocks.{l}.attn.W_K"] * state_dict[f"blocks.{l}.ln1.b"][None, :, None]).sum(-2)
            state_dict[f"blocks.{l}.attn.b_V"] = state_dict[f"blocks.{l}.attn.b_V"] + (state_dict[f"blocks.{l}.attn.W_V"] * state_dict[f"blocks.{l}.ln1.b"][None, :, None]).sum(-2)
            
            state_dict[f"blocks.{l}.attn.W_Q"] = state_dict[f"blocks.{l}.attn.W_Q"] * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            state_dict[f"blocks.{l}.attn.W_K"] = state_dict[f"blocks.{l}.attn.W_K"] * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            state_dict[f"blocks.{l}.attn.W_V"] = state_dict[f"blocks.{l}.attn.W_V"] * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            
            
            # Fold ln2 into MLP
            state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[f"blocks.{l}.mlp.b_in"] + (state_dict[f"blocks.{l}.mlp.W_in"] * state_dict[f"blocks.{l}.ln2.b"][:, None]).sum(-2)
            state_dict[f"blocks.{l}.mlp.W_in"] = state_dict[f"blocks.{l}.mlp.W_in"] * state_dict[f"blocks.{l}.ln2.w"][:, None]
            del state_dict[f"blocks.{l}.ln1.w"], state_dict[f"blocks.{l}.ln1.b"], state_dict[f"blocks.{l}.ln2.w"], state_dict[f"blocks.{l}.ln2.b"]
        # Fold ln_final into Unembed
        # We assume there is no existing bias in the unembed layer
        state_dict[f"unembed.b_U"] = state_dict[f"unembed.b_U"] + (state_dict[f"unembed.W_U"] * state_dict[f"ln_final.b"][:, None]).sum(dim=-2)
        state_dict[f"unembed.W_U"] = state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"][:, None]
        del state_dict[f"ln_final.w"], state_dict[f"ln_final.b"]
        return state_dict
    
    
    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and W_out. This is done by subtracting the mean of the weights from the weights themselves. This is done in-place. See fold_layer_norm for more details.
        """
        state_dict['embed.W_E'] = state_dict['embed.W_E'] - state_dict['embed.W_E'].mean(-1, keepdim=True)
        state_dict['pos_embed.W_pos'] = state_dict['pos_embed.W_pos'] - state_dict['pos_embed.W_pos'].mean(-1, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f'blocks.{l}.attn.W_O'] = state_dict[f'blocks.{l}.attn.W_O'] - state_dict[f'blocks.{l}.attn.W_O'].mean(-1, keepdim=True) # W_O is [head_index, d_model, d_head]
            state_dict[f'blocks.{l}.attn.b_O'] = state_dict[f'blocks.{l}.attn.b_O'] - state_dict[f'blocks.{l}.attn.b_O'].mean() # b_O is [d_model]
            state_dict[f'blocks.{l}.mlp.W_out'] = state_dict[f'blocks.{l}.mlp.W_out'] - state_dict[f'blocks.{l}.mlp.W_out'].mean(-1, keepdim=True)
            state_dict[f'blocks.{l}.mlp.b_out'] = state_dict[f'blocks.{l}.mlp.b_out'] - state_dict[f'blocks.{l}.mlp.b_out'].mean()
        return state_dict
    
    def center_unembed(self, state_dict: Dict[str, torch.Tensor]):
        """Centers the unembedding weights W_U. This is done by subtracting the mean of the weights from the weights themselves. This is done in-place. As softmax is translation invariant, this changes the logits but not the log probs, and makes the model logits (slightly) more interpretable - when trying to understand how components contribute to the logits, we'll be less misled by components that just add something to every logit.
        """
        state_dict['unembed.W_U'] = state_dict['unembed.W_U'] - state_dict['unembed.W_U'].mean(-1, keepdim=True)
        state_dict['unembed.b_U'] = state_dict['unembed.b_U'] - state_dict['unembed.b_U'].mean()
        return state_dict
    
    def set_use_attn_result(self, use_attn_result):
        """
        Toggles whether to explicitly calculate and expose the result for each attention head - useful for interpretability but can easily burn through GPU memory.
        """
        self.cfg.use_attn_result = use_attn_result
        
    @torch.inference_mode()
    def generate(
        self,
        input: Union[str, torch.Tensor] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        num_return_sequences: int = 1,
        use_past_kv_cache: bool = True,
        prepend_bos = True,
        return_type: Optional[str] = "input",
    ):
        """
        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish (by producing an EOT token), we keep running the model on the entire batch, but throw away the output for a finished sequence and just keep adding EOTs to pad.

        This supports entering a single string, but not a list of strings - if the strings don't tokenize to exactly the same length, this gets messy. If that functionality is needed, convert them to a batch of tokens and input that instead.

        Args:
            input (int): Either a batch of tokens ([batch, pos]) or a text string (this will be converted to a batch of tokens with batch size 1)
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0, we take the top tokens with cumulative probability >= top_p
            temperature (float): Temperature for sampling. Higher values will make the model more random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is sampling from a uniform distribution)
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous tokens. Higher values will make the model more random
            use_past_kv_cache (bool): If True, create and use cache to speed up generation
            prepend_bos (bool): If True, prepend the model's bos_token_id to the input, if it's a string. Irrelevant if input is a tensor.
            return_type (str, *optional*): The type of the output to return - either a string (str), a tensor of tokens (tensor) or whatever the format of the input was (input). 
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens - by default returns same type as input
        """
        if type(input) == str:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input

        if return_type == "input":
            if type(input) == str:
                return_type = "str"
            else:
                return_type = "tensor"

        assert isinstance(tokens, torch.Tensor)
        batch_size, ctx_length = tokens.shape
        tokens = tokens.to(self.cfg.device)
        if use_past_kv_cache:
            past_kv_cache = EasyTransformerKeyValueCache.init_cache(self.cfg, self.cfg.device, batch_size)
        else:
            past_kv_cache = None

        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

            eos_token_id = self.tokenizer.eos_token_id
            
            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)
        
        # Currently nothing in EasyTransformer changes with eval, but this is here in case that changes in the future
        self.eval()
        for index in tqdm.tqdm(range(max_new_tokens)):
            # While generating, we keep generating logits, throw away all but the final logits, and then use those logits to sample from the distribution
            # We keep adding the sampled tokens to the end of tokens.
            if use_past_kv_cache:
                # We just take the final tokens, as a [batch, 1] tensor
                if index>0:
                    logits = self.forward(tokens[:, -1:], return_type="logits", past_kv_cache=past_kv_cache)
                else:
                    logits = self.forward(tokens, return_type="logits", past_kv_cache=past_kv_cache)

            else:
                # We input the entire sequence, as a [batch, pos] tensor, since we aren't using the cache
                logits = self.forward(tokens, return_type="logits")
            final_logits = logits[:, -1, :]

            sampled_tokens = sample_logits(
                final_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
                tokens=tokens,
            )
            
            if stop_at_eos:
                # For all unfinished sequences, add on the next token. If a sequence finished, we throw away the generated token and instead add an EOS token to pad.
                sampled_tokens[finished_sequences] = eos_token_id
                finished_sequences.logical_or_(sampled_tokens == eos_token_id)
            
            tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

            if stop_at_eos and finished_sequences.all():
                break
        
        if return_type == "str":
            if prepend_bos:
                # If we prepended a BOS token, remove it when returning output.
                return self.tokenizer.decode(tokens[0, 1:])
            else:
                return self.tokenizer.decode(tokens[0])

        else:
            return tokens

    # def greedy_search(
    #     self,
    #     tokens: torch.Tensor,
    #     max_new_tokens: int,
    #     stop_at_eos: bool = True,
    #     pad_token_id: Optional[int] = None,
    #     eos_token_id: Optional[int] = None,
    #     past_kv_cache: Optional[EasyTransformerKeyValueCache] = None,
    #     return_type: Optional[str] = None,
    # ):
    #     """
    #     Greedily sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
    #     Args:
    #         tokens (torch.Tensor): A batch of tokens ([batch, pos])
    #         max_new_tokens (int): Maximum number of tokens to generate
    #         stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
    #         pad_token_id (int, *optional*): The token ID to use for padding. If None, use the tokenizer's pad_token_id - required if using stop_at_eos
    #         eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
    #         past_kv_cache (EasyTransformerKeyValueCache, *optional*): Cache to use for past keys and values for the model. If None, no cache is used
    #         return_type (str, *optional*): The type of the output to return - either a string (str), a list of strings (list), or a tensor of tokens (tensor). If None, defaults to tensor.
    #     Returns:
    #         outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
    #     """
    #     B, S = tokens.shape
    #     outputs = tokens
    #     unfinished_sequences = tokens.new(tokens.shape[0]).fill_(1)

    #     if stop_at_eos and pad_token_id is None:
    #         assert (
    #             self.tokenizer is not None and self.tokenizer.pad_token_id is not None
    #         ), "Must pass a pad_token_id if stop_at_eos is True and tokenizer is None or has no pad_token_id"
    #         pad_token_id = self.tokenizer.pad_token_id
    #     if stop_at_eos and eos_token_id is None:
    #         assert (
    #             self.tokenizer is not None and self.tokenizer.eos_token_id is not None
    #         ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
    #         eos_token_id = self.tokenizer.eos_token_id

    #     for _ in tqdm.tqdm(range(max_new_tokens)):
    #         logits = self(tokens, return_type="logits", past_kv_cache=past_kv_cache)
    #         next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
    #         if stop_at_eos:
    #             next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
    #                 1 - unfinished_sequences
    #             )
    #             unfinished_sequences.mul_((next_tokens != eos_token_id).long())
    #         outputs = torch.cat([outputs, next_tokens.unsqueeze(-1)], dim=-1)
    #         if past_kv_cache is not None:
    #             tokens = next_tokens.unsqueeze(-1)
    #         else:
    #             tokens = outputs

    #     if return_type is not None and return_type == "str":
    #         assert self.tokenizer is not None
    #         outputs = self.tokenizer.batch_decode(outputs)[0]
    #     elif return_type is not None and return_type == "list":
    #         assert self.tokenizer is not None
    #         outputs = self.tokenizer.batch_decode(outputs)

    #     return outputs

    def sample(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        freq_penalty: float = 0.0,
        cache: Optional[EasyTransformerKeyValueCache] = None,
        return_type: Optional[str] = None,
    ):
        """
        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        If temperature == 0.0, greedy search is used instead.

        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            pad_token_id (int, *optional*): The token ID to use for padding. If None, use the tokenizer's pad_token_id - required if using stop_at_eos
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens
            temperature (float): Temperature for sampling. Higher values will make the model more random
            freq_penalty (float): Frequency penalty for sampling. Higher values will make the model more random
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
            return_type (str, *optional*): If "str", return a string. If "list", return a list of strings. If None, return a tensor
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
        """
        B, S = tokens.shape
        outputs = tokens

        if stop_at_eos and pad_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.pad_token_id is not None
            ), "Must pass a pad_token_id if stop_at_eos is True and tokenizer is None or has no pad_token_id"
            pad_token_id = self.tokenizer.pad_token_id
        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id


        if return_type is not None and return_type == "str":
            assert self.tokenizer is not None
            outputs = self.tokenizer.batch_decode(outputs)[0]
        elif return_type is not None and return_type == "list":
            assert self.tokenizer is not None
            outputs = self.tokenizer.batch_decode(outputs)

        return outputs