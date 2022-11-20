from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional
from easy_transformer.utils import set_seed_everywhere
import torch
import torch.nn as nn
import random
import numpy as np
import logging

SUPPORTED_ACTIVATIONS = ["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"]


@dataclass
class EasyTransformerConfig:
    """
    Configuration class to store the configuration of a EasyTransformer model.

    See further_comments.md for more details on the more complex arguments.

    Args:
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_layers (int): The number of attention layers.
        n_ctx (int): The maximum sequence length.
        n_heads (int, *optional*): The number of attention heads. If not specified, will be set to d_model // d_head.
        d_mlp (int, *optional*): The dimensionality of the feedforward mlp network. Defaults to 4 * d_model, and in an attn-only model is None.
        d_vocab (int): The size of the vocabulary. If not set, will be automatically set
            from the tokenizer's vocab size.
        act_fn (str, *optional"): The activation function to use. Always lowercase.
            Supports ['relu', 'gelu', 'silu', 'gelu_new', 'solu_ln', 'gelu_fast']. Must be set unless using an attn-only model.
        eps (float): The epsilon value to use for layer normalization. Defaults to 1e-5
        use_attn_result (bool): whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        use_attn_scale (bool): whether to scale the attention weights by
        1/sqrt(d_head)
        use_local_attn (bool): whether to use local attention
        model_name (str): the name of the model, used to load
            weights from HuggingFace or initialized to "custom" if not passed
        model_family (str, *optional*): the family of the model, used to help load
            weights from HuggingFace or initialized to "custom" if not passed
        checkpoint (str, *optional*): the checkpoint to load weights from, if using a checkpointed pretrained model.
        tokenizer_name (str, *optional*): the full name of the model, passed into
            HuggingFace to access the tokenizer. Only used when passing in custom
            config, if loading from pretrained then this is not needed.
        window_size (int, *optional*): the size of the window for local
            attention
        attn_types (List[str], *optional*): the types of attention to use for
            local attention
        weight_init_mode (str): the initialization mode to use for the
            weights. Only relevant for custom models, ignored for pre-trained. Options
            are 'pytorch' (for PyTorch defaults) and 'gpt2' (for GPT-2 defaults),
            defaults to 'gpt2
        normalization_type (str, *optional*): the type of normalization to use. Options
            are None (no normalization), 'LN' (use LayerNorm, including weights &
            biases) and 'LNPre' (use LayerNorm, but no weights & biases). Defaults to
            None
        device(str): The device to use for the model. Defaults to 'cuda' if available,
            else 'cpu
        attention_dir (str): Whether to use causal (aka unidirectional aka GPT-2
            style) or bidirectional attention. Options are 'causal' and 'bidirectional'.
            Defaults to 'causal'
        attn_only (bool): Whether to only use attention layers, no feedforward
            layers. Defaults to False
        seed (int, *optional*): The seed to use for the model. Defaults to 42. Used to set sources of randomness (Python, PyTorch and
            NumPy) and to initialize weights. If set to None, does nothing.
        initializer_range (float): The standard deviation of the normal used to initialise the weights, initialized to 0.8 / sqrt(d_model) .
        init_weights (bool): Whether to initialize the weights. Defaults to True. If False, does not initialize weights.
        scale_attn_by_inverse_layer_idx (bool): Whether to scale the attention weights by 1/(layer_id
            +1), used by Mistral (Stanford) models for numerical stability when training in FP16.
            Defaults to False.
        positional_embedding_type (str): The positional embedding used. Options are 'standard' (ie
            GPT-2 style, absolute, randomly initialized learned positional embeddings, directly added
            to the residual stream), 'rotary' (described here: https://blog.eleuther.ai/rotary-embeddings/ ) and 'shortformer' (GPT-2 style absolute &
            learned, but rather than being added to the residual stream they're only added to the
            inputs to the keys and the queries (ie key = W_K(res_stream + pos_embed), but values and
            MLPs don't get any positional info)). Sinusoidal are not currently
            supported. Defaults to 'standard'.
        final_rms (bool): Whether to replace the final normalization (just before the unembed) with RMSNorm (ie no centering or bias, just scaling + weights). Only included because of a dumb bug in my original SoLU code. Defaults to False.
        d_vocab_out (int, *optional*): The size of the output vocabulary. If not set, will be equal to d_vocab. Mainly useful for algorithmic tasks where the input and output vocabularies may be different.
        parallel_attn_mlp (bool): Whether to parallelize the attention and MLP layers - a weird cursed thing done by GPT-J. Means that mlp_out=MLP(ln1(resid_pre)) and resid_post=resid_pre+attn_out+mlp_out. Defaults to False.
        rotary_dim (int): The dimensionality of the rotary embeddings, may be < d_head in which case only the first rotary_dim dimensions of each head are rotated. Defaults to 64, only used is positional_embedding_type=="rotary".
        dtype (torch.dtype): The float encoding to use for the model. Defaults to torch.float32.
    """

    n_layers: int
    d_model: int
    n_ctx: int
    d_head: int
    model_name: str = "custom"
    n_heads: Optional[int] = None
    d_mlp: Optional[int] = None
    act_fn: Optional[str] = None
    d_vocab: Optional[int] = None
    eps: float = 1e-5
    use_attn_result: bool = False
    use_headwise_qkv_input: bool = False  # added by Arthur
    use_attn_scale: bool = True
    use_local_attn: bool = False
    model_family: Optional[str] = None
    checkpoint: Optional[int] = None
    tokenizer_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None
    init_mode: str = "gpt2"
    normalization_type: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    attention_dir: str = "causal"
    attn_only: bool = False
    seed: int = 42
    initializer_range: float = -1.0
    init_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    positional_embedding_type: str = "standard"
    final_rms: bool = False
    d_vocab_out: Optional[int] = None
    parallel_attn_mlp: bool = False
    rotary_dim: int = 64
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.n_heads is None:
            self.n_heads = self.d_model // self.d_head

        if not self.d_model == (self.n_heads * self.d_head):
            logging.warning(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads} * d_head={self.d_head}"
            )

        if self.seed is not None:
            set_seed_everywhere(self.seed)
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

        if self.d_vocab_out is None:
            self.d_vocab_out = self.d_vocab

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `EasyTransformerConfig` from a Python dictionary of parameters.
        """
        return cls(**config_dict)
