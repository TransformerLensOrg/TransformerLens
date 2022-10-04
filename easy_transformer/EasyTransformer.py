from mimetypes import init
from typing import Callable, Union, List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import logging

from functools import *

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer import EasyTransformerConfig

from easy_transformer.components import *
import easy_transformer.weight_conversion as weight_conversion
from easy_transformer.utils import lm_cross_entropy_loss


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

        if self.cfg.attn_only:
            self.blocks = nn.ModuleList(
                [
                    AttnOnlyBlock(self.cfg, block_index)
                    for block_index in range(self.cfg.n_layers)
                ]
            )
        else:
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

        if self.cfg.positional_embedding_type == "shortformer":
            # Load in positional embeddings to each attn layer to use for shortformer style attention, rather than adding to the residual stream.
            for block in self.blocks:
                block.attn.shortformer_load_pos_embeds(self.pos_embed.W_pos)

        if self.cfg.init_weights:
            self.init_weights()

        if move_to_device:
            self.to(self.cfg.device)
        
        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    def forward(self, input: Union[str, torch.Tensor], return_type: Optional[str] = "logits", prepend_bos: bool = True) -> Union[None, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            # Moves tokens to the device of the model by default
            # Maybe this is annoying - let me know if you want an option to disable
            tokens = input.to(self.cfg.device)
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(tokens))  # [batch, pos, d_model]
        if self.cfg.positional_embedding_type != "shortformer":
            residual = embed + pos_embed  # [batch, pos, d_model]
        else:
            # If we're using shortformer style attention, we don't add the positional embedding to the residual stream
            residual = embed
        for block in self.blocks:
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(residual)  # [batch, pos, d_model]
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
    
    def to_str_tokens(self, input, prepend_bos=True):
        """Method to map text or tokens to a list of tokens as strings, for a SINGLE input.

        Args:
            input (Union[str, torch.Tensor]): The input - either a string or a tensor of tokens. If tokens, should be a tensor of shape [pos] or [1, pos]
            prepend_bos (bool, optional): Whether to prepend a BOS token. Only applies if input is a string. Defaults to True.

        Returns:
            str_tokens: List of individual tokens as strings
        """
        if isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos).squeeze()
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
        """Takes in a state dict from a pretrained model, formatted to be consistent with EasyTransformer but with LayerNorm weights and biases. Folds these into the neighbouring weights.
        
        What is LayerNorm Folding?
        Mathematically, LayerNorm is the following:
        x1 = x0 - x0.mean()
        x2 = x1 / ((x1**2).mean()).sqrt()
        x3 = x2 * w
        x4 = x3 + b
        
        Apart from dividing by the norm, these are all pretty straightforwards operations from a linear algebra perspective. And from an interpretability perspective, if anything is linear, it's really easy and you can mostly ignore it (everything breaks up into sums, you can freely change basis, don't need to track interference between terms, etc) - the hard part is engaging with non-linearities!
        
        A key thing to bear in mind is that EVERY time we read from the residual stream, we apply a LayerNorm - this gives us a lot of leverage to reason about it!
        
        So let's translate this into linear algebra notation.
        x0 is a vector in R^n
        
        x1 = x0 - x0.mean() 
            = x0 - (x0.mean()) * ones (broadcasting, ones=torch.ones(n))
            = (x0 @ ones/sqrt(n)) * ones/sqrt(n). ones has norm sqrt(n), so ones/sqrt(n) is the unit vector in the diagonal direction. We're just projecting x0 onto this (fixed) vector and subtracting that value off. Alternately, we're projecting onto the n-1 dimensional subspace orthogonal to ones.
            
            Since LayerNorm is applied EVERY time we read from the stream, the model just never uses the ones direction of the residual stream, so it's essentially just decreasing d_model by one. We can simulate this by just centering all matrices writing to the residual stream.
            Why is removing this dimension useful? I have no idea! I'm not convinced it is...
        
        x2 = x1 / ((x1**2).mean()).sqrt() (Ignoring eps)
           = (x1 / x1.norm()) * sqrt(n)
           This is a projection onto the unit sphere (well, sphere of radius sqrt(n) - the norm of ones). This is fundamentally non-linear, eg doubling the input keeps the output exactly the same.  
           This is by far the most irritating part of LayerNorm. I THINK it's mostly useful for numerical stability reasons and not used to do useful computation by the model, but I could easily be wrong! And interpreting a circuit containing LayerNorm sounds like a nightmare...
           In practice, you can mostly get aware with ignore this and treating the scaling factor as a constant, since it does apply across the entire residual stream for each token - this makes it a "global" property of the model's calculation, so for any specific question it hopefully doesn't matter that much. But when you're considering a sufficiently important circuit that it's a good fraction of the norm of the residual stream, it's probably worth thinking about.
        
        x3 = x2 * w
           = x2 @ W_ln (W_ln is a diagonal matrix with the weights of the LayerNorm)
           This is really easy to deal with - we're about to be input to a linear layer, and can say (x2 @ W_ln) @ W = x2 @ (W_ln @ W) = x2 @ W_eff - we can just fold the LayerNorm weights into the linear layer weights.
        
        x4 = x3 + b is similarly easy - x4 @ W + B = x2 @ W_eff + B_eff, where W_eff = W_ln @ W and B_eff = B + b @ W
        
        This function is calculating W_eff and B_eff for each layer reading from the residual stream and replacing W and B with those
        
        
        See this for more: https://transformer-circuits.pub/2021/framework/index.html#:~:text=Handling%20Layer%20Normalization

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
        
