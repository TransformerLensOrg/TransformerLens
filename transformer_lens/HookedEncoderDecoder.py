"""Hooked EncoderDecoder

Contains a T5 style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

import logging
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast, overload

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from transformers import AutoTokenizer
from typing_extensions import Literal

import transformer_lens.loading_from_pretrained as loading
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import Embed, RMSNorm, T5Block, Unembed
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities import devices


class HookedEncoderDecoder(HookedRootModule):
    """
    This class implements a T5 encoder-decoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - Also note that model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
        - The model only accepts tokens as inputs, and not strings, or lists of strings
    """

    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedEncoderDecoder.from_pretrained() instead."
            )
        self.cfg = cfg

        if self.cfg.n_devices != 1:
            raise ValueError("Multiple devices not supported for HookedEncoderDecoder")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.cfg.tokenizer_name is not None:
            huggingface_token = os.environ.get("HF_TOKEN", None)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                token=huggingface_token,
            )
        else:
            self.tokenizer = None

        if self.cfg.d_vocab == -1:
            # If we have a tokenizer, vocab size can be inferred from it.
            if self.tokenizer is None:
                raise ValueError("Must provide a tokenizer if d_vocab is not provided")

            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

        self.embed = Embed(self.cfg)
        self.encoder = nn.ModuleList(
            [
                T5Block(self.cfg, num_layer, is_decoder=False)
                for num_layer in range(self.cfg.n_layers)
            ]
        )
        self.encoder_final_ln = RMSNorm(self.cfg)
        self.decoder = nn.ModuleList(
            [
                T5Block(self.cfg, num_layer, is_decoder=True)
                for num_layer in range(self.cfg.n_layers)
            ]
        )
        self.decoder_final_ln = RMSNorm(self.cfg)
        # self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.d_vocab_out)
        self.unembed = Unembed(self.cfg)

        self.hook_embed = HookPoint()

        if move_to_device:
            self.to(self.cfg.device)

        self.setup()

    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        decoder_input: Int[torch.Tensor, "batch decoder_pos"],
        return_type: Optional[str] = "logits",
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Float[torch.Tensor, "batch decoder_pos d_vocab"]]:
        """Input must be a batch of tokens. Strings and lists of strings are not yet supported.
        decoder_input: Int[torch.Tensor, "batch decoder_pos"]: The input to the decoder. This is the sequence of tokens that the model will generate, usually with a start token at the beginning
        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), or 'logits' (return logits).
        one_zero_attention_mask: Optional[torch.Tensor]: A binary mask which indicates which tokens should be attended to (1) and which should be ignored (0). Primarily used for padding variable-length sentences in a batch. For instance, in a batch with sentences of differing lengths, shorter sentences are padded with 0s on the right. If not provided, the model assumes all tokens should be attended to.
        """

        tokens = input

        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(self.cfg.device)
            if one_zero_attention_mask is not None:
                one_zero_attention_mask = one_zero_attention_mask.to(self.cfg.device)

        resid = self.hook_embed(self.embed(tokens))

        if one_zero_attention_mask is not None:
            additive_attention_mask = (
                repeat(1 - one_zero_attention_mask, "batch pos -> batch 1 1 pos")
            ) * torch.finfo(self.cfg.dtype).min
        else:
            additive_attention_mask = None

        query_len = key_len = input.shape[1]

        encoder_positional_bias = self.encoder[0].attn.compute_relative_attention_bias(
            query_len, key_len, device=self.cfg.device
        )

        for encoder_block in self.encoder:
            resid = encoder_block(
                resid_pre=resid,
                additive_attention_mask=additive_attention_mask,
                position_bias=encoder_positional_bias,
            )

        encoder_resid = self.encoder_final_ln(resid)

        decoder_resid = self.embed(decoder_input)
        decoder_query_len = decoder_key_len = decoder_input.shape[1]
        decoder_positional_bias = self.decoder[0].attn.compute_relative_attention_bias(
            decoder_query_len, decoder_key_len, device=self.cfg.device
        )

        for decoder_block in self.decoder:
            decoder_resid = decoder_block(
                resid_pre=decoder_resid,
                position_bias=decoder_positional_bias,
                encoder_hidden_states=encoder_resid,
                encoder_additive_attention_mask=additive_attention_mask,
            )

        decoder_resid = self.decoder_final_ln(decoder_resid)

        if self.cfg.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            decoder_resid *= self.cfg.d_model**-0.5

        logits = self.unembed(decoder_resid)
        if return_type is None:
            return None
        return logits

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule. This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def to(  # type: ignore
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cuda")

    def cpu(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cpu")

    def mps(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("mps")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model=None,
        device: Optional[str] = None,
        tokenizer=None,
        move_to_device=True,
        dtype=torch.float32,
        **from_pretrained_kwargs,
    ) -> HookedEncoderDecoder:
        """Loads in the pretrained weights from huggingface. Currently supports loading weight from HuggingFace BertForMaskedLM. Unlike HookedTransformer, this does not yet do any preprocessing on the model."""
        logging.warning(
            "Support for T5 in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
            "If using T5 for interpretability research, keep in mind that T5 has some significant architectural "
            "differences to GPT. The major one is that T5 is an Encoder-Decoder model"
            "Also, it uses relative positional embeddings, different types of Attention (without bias) and LayerNorm"
        )

        if from_pretrained_kwargs.get("load_in_8bit", False) or from_pretrained_kwargs.get(
            "load_in_4bit", False
        ):
            raise ValueError("Quantization not supported")

        if "torch_dtype" in from_pretrained_kwargs:
            dtype = from_pretrained_kwargs["torch_dtype"]

        name_or_path = (
            model_name if Path(model_name).exists() else loading.get_official_model_name(model_name)
        )

        cfg = loading.get_pretrained_model_config(
            name_or_path,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=False,
            device=device,
            n_devices=1,
            dtype=dtype,
            **from_pretrained_kwargs,
        )

        state_dict = loading.get_pretrained_state_dict(
            name_or_path, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )

        model = cls(cfg, tokenizer, move_to_device=False)

        model.load_state_dict(state_dict, strict=False)

        if move_to_device:
            model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedTransformer")

        return model

    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """
        Convenience to get the unembedding matrix (ie the linear map from the final residual stream to the output logits)
        """
        return self.unembed.W_U

    @property
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        """
        Convenience to get the unembedding bias
        """
        return self.unembed.b_U

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """
        Convenience to get the embedding matrix
        """
        return self.embed.W_E

    @property
    def W_pos(self) -> None:
        """
        Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
        """
        raise NotImplementedError(
            "T5 does not have absolute positional embeddings. Uses relative positional embeddings instead."
        )

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.W_K for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.W_Q for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.W_V for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.W_O for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).mlp.W_in for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).mlp.W_out for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.b_K for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.b_Q for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.b_V for block in chain(self.encoder, self.decoder)],
            dim=0,
        )

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.b_O for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        return torch.stack(
            [cast(T5Block, block).mlp.b_in for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        return torch.stack(
            [cast(T5Block, block).mlp.b_out for block in chain(self.encoder, self.decoder)], dim=0
        )

    @property
    def QK(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the Q and K matrices for each layer and head.
        Useful for visualizing attention patterns."""
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the O and V matrices for each layer and head."""
        return FactoredMatrix(self.W_V, self.W_O)

    def all_head_labels(self) -> List[str]:
        """Returns a list of strings with the format "L{l}H{h}", where l is the layer index and h is the head index."""
        return [f"EL{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)] + [
            f"DL{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)
        ]
