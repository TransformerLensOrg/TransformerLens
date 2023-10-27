"""Hooked Transformer Config.

Module with a dataclass for storing the configuration of a
:class:`transformer_lens.HookedTransformer` model.
"""
from __future__ import annotations

import logging
import pprint
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from transformer_lens import utils

SUPPORTED_ACTIVATIONS = ["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"]


@dataclass
class HookedTransformerConfig:
    """
    Configuration class to store the configuration of a HookedTransformer model.

    See further_comments.md for more details on the more complex arguments.

    Args:
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_layers (int): The number of transformer blocks (one block = one attn layer AND one MLP layer).
        n_ctx (int): The maximum sequence length.
        n_heads (int): The number of attention heads. If not
            specified, will be set to d_model // d_head. (This is represented by a default value of -1)
        d_mlp (int, *optional*): The dimensionality of the feedforward mlp
            network. Defaults to 4 * d_model, and in an attn-only model is None.
        d_vocab (int): The size of the vocabulary. Defaults to -1, which means not set. If not set, will be
            automatically set from the tokenizer's vocab size.
        act_fn (str, *optional*): The activation function to use. Always
            lowercase. Supports ['relu', 'gelu', 'silu', 'gelu_new', 'solu_ln',
            'gelu_fast']. Must be set unless using an attn-only model.
        eps (float): The epsilon value to use for layer normalization. Defaults
            to 1e-5
        use_attn_result (bool): whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        use_split_qkv_input (bool): whether to explicitly calculate the input of
            each head separately, with a hook. Defaults to false to save memory.
        use_hook_mlp_in (bool): whether to use a hook to get the input to the
            MLP layer. Defaults to false to save memory.
        use_attn_in (bool): whether to explicitly calculate the input of each
            attention head separately, with a hook. Defaults to false to save memory
        use_attn_scale (bool): whether to scale the attention weights by
            1/sqrt(d_head)
        model_name (str): the name of the model, used to load
            weights from HuggingFace or initialized to "custom" if not passed
        original_architecture (str, *optional*): the family of the model, used
        to help load
            weights from HuggingFace or initialized to "custom" if not passed
        from_checkpoint (bool): Whether the model weights were
            loaded from a checkpoint (only applies to pretrained models)
        checkpoint_index (int, *optional*): The index of the
            checkpoint loaded (only applies to pretrained models).
        checkpoint_label_type (str, *optional*): Whether
            checkpoints are labelled by the number of steps or number of tokens.
        checkpoint_value (int, *optional*): The value of the
            checkpoint label (whether of steps or tokens).
        tokenizer_name (str, *optional*): the full name of the model, passed into
            HuggingFace to access the tokenizer. Only used when passing in
            custom config, if loading from pretrained then this is not needed.
        use_local_attn (bool): whether to use local attention - ie each
            destination token can only attend to source tokens a certain distance back.
        window_size (int, *optional*): the size of the window for local
            attention
        attn_types (List[str], *optional*): the types of attention to use for
            local attention
        weight_init_mode (str): the initialization mode to use for the
            weights. Only relevant for custom models, ignored for pre-trained.
            Currently the only supported mode is 'gpt2', where biases are
            initialized to 0 and weights are standard normals of range
            initializer_range.
        normalization_type (str, *optional*): the type of normalization to use.
            Options are None (no normalization), 'LN' (use LayerNorm, including weights
            & biases) and 'LNPre' (use LayerNorm, but no weights & biases).
            Defaults to LN
        device(str): The device to use for the model. Defaults to 'cuda' if
            available, else 'cpu'. Must be 'cuda' if `n_devices` > 1.
        n_devices (int): The number of devices to use for the model. Defaults to 1. Layers are loaded
            to support "pipeline parallelism", where each device is responsible for a subset of the layers.
        attention_dir (str): Whether to use causal (aka unidirectional aka GPT-2
            style) or bidirectional attention. Options are 'causal' and
            'bidirectional'. Defaults to 'causal'
        attn_only (bool): Whether to only use attention layers, no feedforward
            layers. Defaults to False
        seed (int, *optional*): The seed to use for the model.
            Used to set sources of randomness (Python, PyTorch and
            NumPy) and to initialize weights. Defaults to None. We recommend setting a seed, so your experiments are reproducible.
        initializer_range (float): The standard deviation of the normal used to
            initialise the weights, initialized to 0.8 / sqrt(d_model) .
        init_weights (bool): Whether to initialize the weights. Defaults to
            True. If False, does not initialize weights.
        scale_attn_by_inverse_layer_idx (bool): Whether to scale the attention
            weights by 1/(layer_id+1), used by Mistral (Stanford) models for numerical stability when
            training in FP16. Defaults to False.
        positional_embedding_type (str): The positional embedding used. Options
            are 'standard' (ie GPT-2 style, absolute, randomly initialized learned positional
            embeddings, directly added to the residual stream), 'rotary'
            (described here: https://blog.eleuther.ai/rotary-embeddings/ ) and
            'shortformer' (GPT-2 style absolute & learned, but rather than being
            added to the residual stream they're only added to the inputs to the
            keys and the queries (ie key = W_K(res_stream + pos_embed), but
            values and MLPs don't get any positional info)). Sinusoidal are not
            currently supported. Defaults to 'standard'.
        final_rms (bool): Whether to replace the final normalization (just
            before the unembed) with RMSNorm (ie no centering or bias, just
            scaling + weights). Only included because of a dumb bug in my
            original SoLU code. Defaults to False.
        d_vocab_out (int, *optional*): The size of the output vocabulary. Defaults to -1, which means not set. If not
            set, will be equal to d_vocab. Mainly useful for algorithmic tasks
            where the input and output vocabularies may be different.
        parallel_attn_mlp (bool): Whether to parallelize the attention and MLP
            layers - a weird cursed thing done by GPT-J. Means that
            mlp_out=MLP(ln1(resid_pre)) and resid_post=resid_pre+attn_out+mlp_out. Defaults to False.
        rotary_dim (int, *optional*): The dimensionality of the rotary
            embeddings, may be d_head in which case only the first rotary_dim
            dimensions of each head are rotated. Defaults to None, if
            positional_embedding_type=="rotary" it defaults to d_head.
        n_params (int, *optional*): The number of (hidden weight)
            parameters in the model. This is automatically calculated and not
            intended to be set by the user. (Non embedding parameters, because
            the [scaling laws paper](https://arxiv.org/pdf/2001.08361.pdf) found
            that that was a more meaningful number. Ignoring biases and layer
            norms, for convenience)
        use_hook_tokens (bool): Will add a hook point on the token input to
            HookedTransformer.forward, which lets you cache or intervene on the tokens.
            Defaults to False.
        default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
            methods of HookedTransformer process input text to tokenize (only when input is a string).
            Defaults to True - even for models not explicitly trained with this, heads often use the
            first position as a resting position and accordingly lose information from the first token,
            so this empirically seems to give better results. To change the default behavior to False, pass in
            default_prepend_bos=False. Note that you can also locally override the default behavior by passing
            in prepend_bos=True/False when you call a method that processes the input string.
        dtype (torch.dtype, *optional*): The model's dtype. Defaults to torch.float32.
        tokenizer_prepends_bos (bool, *optional*): This flag is set by set_tokenizer. It is set to True only
            when the tokenizer automatically prepends the BOS token if initialized with add_bos_token=True.
            We need this information to dynamically control bos prepending.
        post_embedding_ln (bool): Whether to apply layer normalization after embedding the tokens. Defaults
            to False.
    """

    n_layers: int
    d_model: int
    n_ctx: int
    d_head: int
    model_name: str = "custom"
    n_heads: int = -1
    d_mlp: Optional[int] = None
    act_fn: Optional[str] = None
    d_vocab: int = -1
    eps: float = 1e-5
    use_attn_result: bool = False
    use_attn_scale: bool = True
    use_split_qkv_input: bool = False
    use_hook_mlp_in: bool = False
    use_attn_in: bool = False
    use_local_attn: bool = False
    original_architecture: Optional[str] = None
    from_checkpoint: bool = False
    checkpoint_index: Optional[int] = None
    checkpoint_label_type: Optional[str] = None
    checkpoint_value: Optional[int] = None
    tokenizer_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None
    init_mode: str = "gpt2"
    normalization_type: Optional[str] = "LN"
    device: Optional[str] = None
    n_devices: int = 1
    attention_dir: str = "causal"
    attn_only: bool = False
    seed: Optional[int] = None
    initializer_range: float = -1.0
    init_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    positional_embedding_type: str = "standard"
    final_rms: bool = False
    d_vocab_out: int = -1
    parallel_attn_mlp: bool = False
    rotary_dim: Optional[int] = None
    n_params: Optional[int] = None
    use_hook_tokens: bool = False
    gated_mlp: bool = False
    default_prepend_bos: bool = True
    dtype: torch.dtype = torch.float32
    tokenizer_prepends_bos: Optional[bool] = None
    post_embedding_ln: bool = False

    def __post_init__(self):
        if self.n_heads == -1:
            self.n_heads = self.d_model // self.d_head

            if not self.d_model % (self.d_head) == 0:
                logging.warning(
                    f"d_model {self.d_model} is not divisible by d_head {self.d_head}. n_heads was inferred to be {self.n_heads}, rounding down the ratio."
                )

        if self.seed is not None:
            self.set_seed_everywhere(self.seed)
        if self.use_local_attn:
            assert (
                self.window_size is not None
            ), "window_size must be specified for local attention"
            assert (
                self.attn_types is not None
            ), "attn_types must be specified for local attention"
        if not self.attn_only:
            if self.d_mlp is None:
                # For some reason everyone hard codes in this hyper-parameter!
                self.d_mlp = self.d_model * 4
            assert (
                self.act_fn is not None
            ), "act_fn must be specified for non-attn-only models"
            assert (
                self.act_fn in SUPPORTED_ACTIVATIONS
            ), f"act_fn={self.act_fn} must be one of {SUPPORTED_ACTIVATIONS}"
        if self.initializer_range < 0:
            # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
            self.initializer_range = 0.8 / np.sqrt(self.d_model)

        if self.d_vocab_out == -1:
            # d_vocab_out defaults to d_vocab, unless there's an algorithmic task
            # If d_vocab is not set, it'll be inferred from tokenizer_name or from a tokenizer explicitly passed to HookedTransformer initialisation.
            self.d_vocab_out = self.d_vocab

        if self.positional_embedding_type == "rotary" and self.rotary_dim is None:
            self.rotary_dim = self.d_head

        # The number of parameters in attention layers (ignoring biases and layer norm). 4 because W_Q, W_K, W_V and W_O
        self.n_params = self.n_layers * (
            (self.d_model * self.d_head * self.n_heads * 4)
        )
        if not self.attn_only:
            # Number of parameters in MLP layers (ignoring biases and layer norm). 2 because W_in and W_out
            self.n_params += self.n_layers * self.d_model * self.d_mlp * 2

        if self.device is None:
            self.device = utils.get_device()

        if self.n_devices > 1:
            assert (
                torch.cuda.device_count() >= self.n_devices
            ), f"Not enough CUDA devices to support n_devices {self.n_devices}"

        assert self.default_prepend_bos in [
            True,
            False,
        ], f"padding_side must be either True or False, but {self.default_prepend_bos} is given"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HookedTransformerConfig:
        """
        Instantiates a `HookedTransformerConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedTransformerConfig:\n" + pprint.pformat(self.to_dict())

    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
