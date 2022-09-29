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
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig

from easy_transformer.components import *
import easy_transformer.weight_conversion as weight_conversion
from easy_transformer.utils import lm_cross_entropy_loss

# TODO: Add Bloom, GPT-J and GPT-NeoX
"""
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
    This class implements a full Transformer using the above components, with
    HookPoints on every interesting activation. It inherits from HookedRootModule.

    It can be initialised with a model name, and then will automatically load the model weights
    for that model, loads them into this model, as well as fold in LayerNorm and center
    the weights.

    It can also be initilised with an EasyTransformerConfig or a config dictionary, which can be used to instantiate a custom model without loading pretrained weights and will instead use Pytorch's default weight initialisation.
    """
    
    VALID_PRETRAINED_MODEL_NAMES = weight_conversion.VALID_PRETRAINED_MODEL_NAMES
    PRETRAINED_MODEL_NAMES_DICT = weight_conversion.PRETRAINED_MODEL_NAMES_DICT
    STANFORD_CRFM_CHECKPOINTS = weight_conversion.STANFORD_CRFM_CHECKPOINTS

    def __init__(
        self,
        model_name,
        cfg=None,
        use_attn_result=False,
        model=None,
        keep_original_model=False,
        checkpoint=None,
        fold_ln=True,
    ):
        """
        model_name (str: The name of the model to load, via HuggingFace. If
            "custom", then cfg must be provided.
        cfg (EasyTransformerConfig, *optional*): The config to use for the
            model. If not provided, a model name must be passed via model_name.
        tokenizer (*optional): The tokenizer to use for the model. If not
            provided, initialized to None, though the user must initialize one
            before passing strings to the model.
        use_attn_result (bool): Says whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        model: The model loaded from HuggingFace or separately initialized. If
            None, it is automatically loaded from HuggingFace if model_name is
            passed - this just saves memory if the model was already loaded into
            RAM.
        keep_original_model (bool): If False, the original model is deleted,
            otherwise it's kept as a self.model attribute
        center_weights (bool): If True, the weights are centered
        checkpoint (int, *optional): The checkpoint number of the model to load
            if it is a model with multiple possible checkpoints to load from.
        """
        super().__init__()
        if model_name == "custom":
            assert cfg is not None, "Must provide a config for custom model"
            self.cfg = cfg
            self.model_name = cfg.model_name
            self.model_type = cfg.model_type
            if self.cfg.tokenizer_name is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens directly. In this case, we don't need a tokenizer.
                self.tokenizer = None
            self.use_attn_result = use_attn_result
            self.hf_model = None
            self.keep_original_model = False
            # We're initializing a model, no need to load weights from a checkpoint
            self.checkpoint = None
        else:
            assert (
                model_name in self.VALID_PRETRAINED_MODEL_NAMES
            ), f"Invalid model name: {model_name}. Valid model names are: {self.VALID_PRETRAINED_MODEL_NAMES}"
            self.model_name = model_name
            if self.model_name in self.PRETRAINED_MODEL_NAMES_DICT:
                self.full_model_name = self.PRETRAINED_MODEL_NAMES_DICT[self.model_name]
            else:
                self.full_model_name = self.model_name
            self.model_type = self.get_model_type(self.full_model_name)
            if model is not None:
                self.hf_model = model
            else:
                if checkpoint is not None:
                    if "stanford" not in self.model_name:
                        logging.warning(
                            f"Loading checkpoints is not supported for the model {self.model_name}. Loading without checkpoints"
                        )
                        self.hf_model = AutoModelForCausalLM.from_pretrained(
                            self.full_model_name
                        )
                    else:
                        assert (
                            checkpoint in self.STANFORD_CRFM_CHECKPOINTS
                        ), f"Checkpoint {checkpoint} is not valid. Available checkpoints are {self.STANFORD_CRFM_CHECKPOINTS}"
                        self.hf_model = AutoModelForCausalLM.from_pretrained(
                            self.full_model_name, revision=f"checkpoint-{checkpoint}"
                        )
                else:
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        self.full_model_name
                    )

            self.cfg = self.convert_hf_config(
                self.hf_model.config, model_type=self.model_type
            )
            self.cfg.use_attn_result = use_attn_result
            self.cfg.checkpoint = checkpoint
            self.cfg.model_type = self.model_type
            self.cfg.model_name = self.model_name
            self.cfg.tokenizer_name = self.full_model_name
            self.cfg.normalization_type = "LNPre" if fold_ln else "LN"
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.cfg.d_vocab:
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

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()
        self.to(self.cfg.device)

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
            tokens = input
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(tokens))  # [batch, pos, d_model]
        residual = embed + pos_embed  # [batch, pos, d_model]
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
                        keep_original_model = False,
                        **kwargs):
        """Class method to load a pretrained model from HuggingFace and to automatically convert and load those weights into EasyTransformer format.

        Args:
            model_name (str): The model name - must be in VALID_MODEL_NAMES
            fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the subsequent linear layer. This does not change the computation. Defaults to True.
            center_writing_weights (bool, optional): Whether to center weights writing to the residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the computation. Defaults to True.
            center_unembed (bool, optional): Whether to center W_U (ie set mean to be zero). Softmax is translation invariant so this doesn't affect log probs or loss, but does change logits. Defaults to True.
            keep_original_model (bool, optional): Whether to delete the model loaded from HuggingFace (stored as model.hf_model). Defaults to False.
        """
        model = cls(model_name, fold_ln=fold_ln, **kwargs)

        # Load model weights, and fold in layer norm weights
        if model.model_type == "gpt2":
            state_dict = weight_conversion.convert_gpt2_weights(model.hf_model, model.cfg)
        elif model.model_type == "neo":
            state_dict = weight_conversion.convert_neo_weights(model.hf_model, model.cfg)
        elif model.model_type == "gptj":
            state_dict = weight_conversion.convert_gptj_weights(model.hf_model, model.cfg)
        elif model.model_type == "neox":
            state_dict = weight_conversion.convert_neox_weights(model.hf_model, model.cfg)
        elif model.model_type == "opt":
            state_dict = weight_conversion.convert_opt_weights(model.hf_model, model.cfg)
        else:
            logging.warning(f"Invalid model_type, no weights are stored to load: {model.model_type}, generated from model name {model.model_name}")
        state_dict = model.fill_missing_keys(state_dict)
        if fold_ln:
            state_dict = model.fold_layer_norm(state_dict)
        if center_writing_weights:
            state_dict = model.center_writing_weights(state_dict)
        if center_unembed:
            state_dict = model.center_unembed(state_dict)
        # Need to delete the HuggingFace model so it isn't counted as a submodule
        del model.hf_model
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_config(cls, cfg):
        """Used to generate a model from a config object to train from

        Args:
            cfg (EasyTransformerConfig): Config for the model

        Returns:
            EasyTransformer: An initialised EasyTransformer model
        """
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig(**cfg)
        model = cls(
            "custom",
            cfg,
            use_attn_result=cfg.use_attn_result,
        )
        model.init_weights()
        return model

    def get_model_type(self, model_name):
        if "gpt2" in model_name or "stanford" in model_name:
            return "gpt2"
        elif "opt" in model_name:
            return "opt"
        elif model_name == "EleutherAI/gpt-neox-20b":
            return "neox"
        elif model_name == "EleutherAI/gpt-j-6B":
            return "gptj"
        elif "neo" in model_name:
            return "neo"
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def convert_hf_config(self, hf_config, model_type):
        cfg_dict = {}
        if model_type == "neo":
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
            }
        elif model_type == "gpt2":
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
            }
        elif model_type == "opt":
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
            }
        elif model_type == "gptj":
            raise NotImplementedError
        elif model_type == "neox":
            raise NotImplementedError
        else:
            raise NotImplementedError
        cfg_dict["model_name"] = self.model_name
        cfg_dict["model_type"] = model_type
        cfg = EasyTransformerConfig.from_dict(cfg_dict)
        return cfg

    def init_weights(self):
        """
        Initialize weights according to default Pytorch initialization.

        LayerNorm weights are already initialized to 1.0 (and biases to 0.0)
        in the constructor
        """
        # Initialize weights with std 1/sqrt(d_model) so the vector has variance 1
        nn.init.normal_(self.embed.W_E, std=self.cfg.d_model ** (-0.5))
        nn.init.normal_(self.pos_embed.W_pos, std=self.cfg.d_model ** (-0.5))

        def init_linear_weight_and_bias(weight, bias):
            nn.init.kaiming_uniform_(weight, a=np.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

        for l in range(self.cfg.n_layers):
            init_linear_weight_and_bias(
                self.blocks[l].attn.W_Q, self.blocks[l].attn.b_Q
            )
            init_linear_weight_and_bias(
                self.blocks[l].attn.W_K, self.blocks[l].attn.b_K
            )
            init_linear_weight_and_bias(
                self.blocks[l].attn.W_V, self.blocks[l].attn.b_V
            )
            init_linear_weight_and_bias(
                self.blocks[l].attn.W_O, self.blocks[l].attn.b_O
            )
            if not self.cfg.attn_only:
                init_linear_weight_and_bias(
                    self.blocks[l].mlp.W_in, self.blocks[l].mlp.b_in
                )
                init_linear_weight_and_bias(
                    self.blocks[l].mlp.W_out, self.blocks[l].mlp.b_out
                )

                if self.cfg.gated_act_fn:
                    init_linear_weight_and_bias(
                        self.blocks[l].mlp.W_gate, self.blocks[l].mlp.b_gate
                    )

        init_linear_weight_and_bias(self.unembed.W_U, self.unembed.b_U)
    
    def fill_missing_keys(self, state_dict):
        """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

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

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model
        """
        for l in range(self.cfg.n_layers):
            # Fold ln1 into attention - it's important to fold biases first, 
            # since biases depend on weights but not vice versa
            state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[f"blocks.{l}.attn.b_Q"] + state_dict[f"blocks.{l}.attn.W_Q"] @ state_dict[f"blocks.{l}.ln1.b"]
            state_dict[f"blocks.{l}.attn.b_K"] = state_dict[f"blocks.{l}.attn.b_K"] + state_dict[f"blocks.{l}.attn.W_K"] @ state_dict[f"blocks.{l}.ln1.b"]
            state_dict[f"blocks.{l}.attn.b_V"] = state_dict[f"blocks.{l}.attn.b_V"] + state_dict[f"blocks.{l}.attn.W_V"] @ state_dict[f"blocks.{l}.ln1.b"]
            
            state_dict[f"blocks.{l}.attn.W_Q"] = state_dict[f"blocks.{l}.attn.W_Q"] * state_dict[f"blocks.{l}.ln1.w"]
            state_dict[f"blocks.{l}.attn.W_K"] = state_dict[f"blocks.{l}.attn.W_K"] * state_dict[f"blocks.{l}.ln1.w"]
            state_dict[f"blocks.{l}.attn.W_V"] = state_dict[f"blocks.{l}.attn.W_V"] * state_dict[f"blocks.{l}.ln1.w"]
            
            
            # Fold ln2 into MLP
            state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[f"blocks.{l}.mlp.b_in"] + state_dict[f"blocks.{l}.mlp.W_in"] @ state_dict[f"blocks.{l}.ln2.b"]
            state_dict[f"blocks.{l}.mlp.W_in"] = state_dict[f"blocks.{l}.mlp.W_in"] * state_dict[f"blocks.{l}.ln2.w"]
            del state_dict[f"blocks.{l}.ln1.w"], state_dict[f"blocks.{l}.ln1.b"], state_dict[f"blocks.{l}.ln2.w"], state_dict[f"blocks.{l}.ln2.b"]
        # Fold ln_final into Unembed
        state_dict[f"unembed.b_U"] = state_dict[f"unembed.W_U"] @ state_dict[f"ln_final.b"]
        state_dict[f"unembed.W_U"] = state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"]
        del state_dict[f"ln_final.w"], state_dict[f"ln_final.b"]
        return state_dict
    
    
    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and W_out. This is done by subtracting the mean of the weights from the weights themselves. This is done in-place. As LayerNorm centers before reading from the residual stream, this doesn't change the computation.
        """
        state_dict['embed.W_E'] = state_dict['embed.W_E'] - state_dict['embed.W_E'].mean(0, keepdim=True)
        state_dict['pos_embed.W_pos'] = state_dict['pos_embed.W_pos'] - state_dict['pos_embed.W_pos'].mean(0, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f'blocks.{l}.attn.W_O'] = state_dict[f'blocks.{l}.attn.W_O'] - state_dict[f'blocks.{l}.attn.W_O'].mean(1, keepdim=True) # W_O is [head_index, d_model, d_head]
            state_dict[f'blocks.{l}.attn.b_O'] = state_dict[f'blocks.{l}.attn.b_O'] - state_dict[f'blocks.{l}.attn.b_O'].mean() # b_O is [d_model]
            state_dict[f'blocks.{l}.mlp.W_out'] = state_dict[f'blocks.{l}.mlp.W_out'] - state_dict[f'blocks.{l}.mlp.W_out'].mean(0, keepdim=True)
            state_dict[f'blocks.{l}.mlp.b_out'] = state_dict[f'blocks.{l}.mlp.b_out'] - state_dict[f'blocks.{l}.mlp.b_out'].mean()
        return state_dict
    
    def center_unembed(self, state_dict: Dict[str, torch.Tensor]):
        """Centers the unembedding weights W_U. This is done by subtracting the mean of the weights from the weights themselves. This is done in-place. As softmax is translation invariant, this changes the logits but not the log probs, and makes the model logits more interpretable.
        """
        state_dict['unembed.W_U'] = state_dict['unembed.W_U'] - state_dict['unembed.W_U'].mean(0, keepdim=True)
        state_dict['unembed.b_U'] = state_dict['unembed.b_U'] - state_dict['unembed.b_U'].mean()
        return state_dict
        
