"""Hooked EncoderDecoder

Contains a T5 style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

import logging
import os
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import torch
import tqdm
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing_extensions import Literal

import transformer_lens.loading_from_pretrained as loading
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import MLP, Embed, GatedMLP, RMSNorm, T5Block, Unembed
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities import devices
from transformer_lens.utils import sample_logits

T = TypeVar("T", bound="HookedEncoderDecoder")


class HookedEncoderDecoder(HookedRootModule):
    """
    This class implements a T5 encoder-decoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - Also note that model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
        - The model only accepts tokens as inputs, and not strings, or lists of strings
    """

    tokenizer: Optional[PreTrainedTokenizerBase]

    def __init__(
        self,
        cfg: Union[HookedTransformerConfig, Dict],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        move_to_device: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedEncoderDecoder.from_pretrained() instead."
            )
        self.cfg: HookedTransformerConfig = cfg

        if self.cfg.n_devices != 1:
            raise ValueError("Multiple devices not supported for HookedEncoderDecoder")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.cfg.tokenizer_name is not None:
            huggingface_token = os.environ.get("HF_TOKEN", "")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                token=huggingface_token if len(huggingface_token) > 0 else None,
            )
        else:
            self.tokenizer = None

        if self.cfg.d_vocab == -1:
            # If we have a tokenizer, vocab size can be inferred from it.
            if self.tokenizer is None:
                raise ValueError("Must provide a tokenizer if d_vocab is not provided")

            self.cfg.d_vocab = len(self.tokenizer)
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

    def to_tokens(
        self,
        input: Union[str, List[str]],
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Tuple[Int[torch.Tensor, "batch pos"], Int[torch.Tensor, "batch pos"]]:
        """Converts a string to a tensor of tokens.
        Taken mostly from the HookedTransformer implementation, but does not support default padding
        sides or prepend_bos.

        Args:
            input (Union[str, List[str]]): The input to tokenize.
            move_to_device (bool): Whether to move the output tensor of tokens to the device the
                model lives on. Defaults to True
            truncate (bool): If the output tokens are too long, whether to truncate the output
                tokens to the model's max context window. Does nothing for shorter inputs.
                Defaults to True.
        """

        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"

        encodings = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )

        tokens = encodings.input_ids
        attention_mask = encodings.attention_mask

        if move_to_device:
            tokens = tokens.to(self.cfg.device)
            attention_mask = attention_mask.to(self.cfg.device)
        return tokens, attention_mask

    @overload
    def forward(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        decoder_input: Optional[Int[torch.Tensor, "batch decoder_pos"]] = None,
        return_type: Literal["logits"] = "logits",
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_vocab"]:
        ...

    @overload
    def forward(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        decoder_input: Optional[Int[torch.Tensor, "batch decoder_pos"]] = None,
        return_type: Optional[Literal[None]] = None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Float[torch.Tensor, "batch pos d_vocab"]]:
        ...

    def forward(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        decoder_input: Optional[Int[torch.Tensor, "batch decoder_pos"]] = None,
        return_type: Optional[str] = "logits",
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Float[torch.Tensor, "batch decoder_pos d_vocab"]]:
        """Forward pass of the T5 model.

        Args:
            input: Input to be processed. Can be one of:
                - str: A single string input
                - List[str]: A batch of string inputs
                - Int[torch.Tensor, "batch pos"]: A batch of token IDs
            decoder_input: Tensor of shape (batch, decoder_pos) containing the decoder input sequence.
                If None and input is of type str or List[str], starts with batch of beginning-of-sequence (BOS) tokens.
            return_type: Specifies the model output type:
                - "logits": Return logits tensor
                - None: Returns nothing
            one_zero_attention_mask: A binary mask which indicates
                which tokens should be attended to (1) and which should be ignored (0).
                Primarily used for padding variable-length sentences in a batch.
                For instance, in a batch with sentences of differing lengths, shorter
                sentences are padded with 0s on the right. If not provided, the model
                assumes all tokens should be attended to.
                This parameter gets inferred from the tokenizer if input is a string or list of strings.
                Shape is (batch_size, sequence_length).

        Returns:
            Optional[Float[torch.Tensor, "batch decoder_pos d_vocab"]]:
                If return_type="logits": Returns logits tensor of shape (batch, decoder_pos, vocab_size)
                If return_type=None: Returns None
        """

        if isinstance(input, (str, list)):
            tokens, attention_mask = self.to_tokens(input)

            # If attention mask is not provided, use the ones from the tokenizer
            one_zero_attention_mask = (
                attention_mask if one_zero_attention_mask is None else one_zero_attention_mask
            )

            # If decoder_input is not provided, start with tensor of PAD tokens of shape (batch, 1)
            if decoder_input is None:
                assert self.tokenizer is not None
                decoder_input = torch.full(
                    (tokens.shape[0], 1),
                    self.tokenizer.pad_token_id,
                    device=self.cfg.device,
                )
        else:
            tokens = input

            if one_zero_attention_mask is None:
                logging.warning(
                    "No attention mask provided. Assuming all tokens should be attended to."
                )

            if decoder_input is None:
                raise ValueError(
                    "Must provide decoder_input if input is not a string or list of strings"
                )

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

        query_len = key_len = tokens.shape[1]

        encoder_positional_bias = cast(
            T5Block, self.encoder[0]
        ).attn.compute_relative_attention_bias(query_len, key_len, device=self.cfg.device)

        for encoder_block in self.encoder:
            resid = encoder_block(
                resid_pre=resid,
                additive_attention_mask=additive_attention_mask,
                position_bias=encoder_positional_bias,
            )

        encoder_resid = self.encoder_final_ln(resid)

        if decoder_input is None:
            raise ValueError("decoder_input cannot be None when input is not a string")
        decoder_resid = self.embed(decoder_input)
        decoder_query_len = decoder_key_len = decoder_input.shape[1]
        decoder_positional_bias = cast(
            T5Block, self.decoder[0]
        ).attn.compute_relative_attention_bias(
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

    @torch.inference_mode()
    def generate(
        self,
        input: Union[str, Int[torch.Tensor, "batch pos"]] = "",
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Union[Int[torch.Tensor, "batch new_tokens"], str]:
        """Sample tokens from the T5 encoder-decoder model.

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        This function is primarily taken from HookedTransformer but adjusted for the HookedEncoderDecoder
        architecture.
        This function does not support key value caching and no default padding sides or prepend_bos.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish
        (by producing an EOT token), we keep running the model on the entire batch, but throw away
        the output for a finished sequence and just keep adding EOTs to pad.

        This supports entering a single string, but not a list of strings - if the strings don't
        tokenize to exactly the same length, this gets messy. If that functionality is needed,
        convert them to a batch of tokens and input that instead.

        Args:
            input (Union[str, Int[torch.Tensor, "batch pos"])]): Either a batch of tokens ([batch,
                pos]) or a text string (this will be converted to a batch of tokens with batch size
                1).
            max_new_tokens (int): Maximum number of tokens to generate.
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token.
            eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end
                of sentence. If None, use the tokenizer's eos_token_id - required if using
                stop_at_eos. It's also possible to provide a list of token IDs (not just the
                eos_token_id), in which case the generation will stop when any of them are output
                (useful e.g. for stable_lm).
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use
                greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens.
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0,
                we take the top tokens with cumulative probability >= top_p.
            temperature (float): Temperature for sampling. Higher values will make the model more
                random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is
                sampling from a uniform distribution).
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous
                tokens. Higher values will make the model more random.
            return_type (Optional[str]): The type of the output to return - either a string (str),
                a tensor of tokens (tensor) or whatever the format of the input was (input).
            verbose (bool): If True, show tqdm progress bars for generation.

        Returns:
            outputs (torch.Tensor): [batch, new_tokens], generated sequence of new tokens
                (by default returns same type as input).
        """

        if isinstance(input, str):
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            encoder_input, attention_mask = self.to_tokens(input)

            # If attention mask is not provided, use the one from the tokenizer
            one_zero_attention_mask = (
                attention_mask if one_zero_attention_mask is None else one_zero_attention_mask
            )
        else:
            assert isinstance(input, torch.Tensor)  # keep mypy happy
            encoder_input = input

            # If tokens are provided, user should be aware that attention mask will not be inferred
            if one_zero_attention_mask is None:
                logging.warning(
                    "No attention mask provided. Assuming all tokens should be attended to."
                )

        if return_type == "input":
            if isinstance(input, str):
                return_type = "str"
            else:
                return_type = "tensor"

        assert isinstance(encoder_input, torch.Tensor)
        batch_size = encoder_input.shape[0]
        device = devices.get_device_for_block_index(0, self.cfg)

        # For the decoder input, we start with a tensor of PAD tokens of shape (batch, 1)
        assert self.tokenizer is not None
        decoder_input = torch.full((batch_size, 1), self.tokenizer.pad_token_id).to(device)

        stop_tokens: List[int] = []
        eos_token_for_padding = 0
        if stop_at_eos:
            tokenizer_has_eos_token = self.tokenizer.eos_token_id is not None

            local_eos_token_id: Optional[Union[int, List[int]]] = eos_token_id
            if local_eos_token_id is None:
                assert (
                    tokenizer_has_eos_token
                ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                local_eos_token_id = self.tokenizer.eos_token_id

            if isinstance(local_eos_token_id, int):
                stop_tokens = [local_eos_token_id]
                eos_token_for_padding = local_eos_token_id
            else:
                # eos_token_id is a Sequence (e.g. list or tuple)
                if local_eos_token_id is None:
                    raise ValueError("eos_token_id cannot be None here")
                stop_tokens = local_eos_token_id
                eos_token_for_padding = (
                    self.tokenizer.eos_token_id
                    if tokenizer_has_eos_token
                    else local_eos_token_id[0]
                )

        # An array to track which sequences in the batch have finished.
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

        # Currently nothing in HookedTransformer changes with eval, but this is here in case
        # that changes in the future.
        self.eval()
        for _ in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
            # While generating, we keep generating logits, throw away all but the final logits,
            # and then use those logits to sample from the distribution We keep adding the
            # sampled tokens to the end of tokens.
            # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
            # the cache.

            # Encoder input will be the same for all iterations
            # Decoder input will be appended with the new token each iteration
            logits = self.forward(
                encoder_input,
                decoder_input=decoder_input,
                one_zero_attention_mask=one_zero_attention_mask,
            )
            assert logits is not None
            final_logits = logits[:, -1, :]

            if do_sample:
                sampled_tokens = sample_logits(
                    final_logits,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    freq_penalty=freq_penalty,
                    tokens=decoder_input,
                ).to(devices.get_device_for_block_index(0, self.cfg))
            else:
                sampled_tokens = final_logits.argmax(-1).to(
                    devices.get_device_for_block_index(0, self.cfg)
                )

            if stop_at_eos:
                # For all unfinished sequences, add on the next token. If a sequence was
                # finished, throw away the generated token and add eos_token_for_padding
                # instead.
                sampled_tokens[finished_sequences] = eos_token_for_padding
                finished_sequences.logical_or_(
                    torch.isin(
                        sampled_tokens.to(self.cfg.device),
                        torch.tensor(stop_tokens).to(self.cfg.device),
                    )
                )

            # Append new token to the decoder input
            decoder_input = torch.cat([decoder_input, sampled_tokens.unsqueeze(-1)], dim=-1)

            if stop_at_eos and finished_sequences.all():
                break

        if return_type == "str":
            assert self.tokenizer is not None
            # Convert tokens to string
            return self.tokenizer.decode(decoder_input[0], skip_special_tokens=True)

        else:
            return decoder_input

    @overload
    def run_with_cache(
        self, *model_args: Any, return_cache_object: Literal[True] = True, **kwargs: Any
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args: Any, return_cache_object: Literal[False] = False, **kwargs: Any
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self,
        *model_args: Any,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs: Any,
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

    def to(self: T, *args: Any, **kwargs: Any) -> T:
        return super().to(*args, **kwargs)

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        elif device is None:
            return self.to("cuda")
        else:
            return self.to(device)

    def cpu(self: T) -> T:
        return self.to("cpu")

    def mps(self: T) -> T:
        return self.to(torch.device("mps"))

    @classmethod
    def from_pretrained(
        cls: Type[T],
        model_name: str,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model: Optional[Any] = None,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        move_to_device: bool = True,
        dtype: Optional[torch.dtype] = torch.float32,
        **from_pretrained_kwargs: Any,
    ) -> T:
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

        if dtype is None:
            dtype = torch.float32

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
            [cast(T5Block, block).attn.W_K for block in chain(self.encoder, self.decoder)],
            dim=0,
        )

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.W_Q for block in chain(self.encoder, self.decoder)],
            dim=0,
        )

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.W_V for block in chain(self.encoder, self.decoder)],
            dim=0,
        )

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.W_O for block in chain(self.encoder, self.decoder)],
            dim=0,
        )

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        weights: List[torch.Tensor] = []
        for block in chain(self.encoder, self.decoder):
            mlp = cast(T5Block, block).mlp
            if isinstance(mlp, (MLP, GatedMLP)):
                weights.append(mlp.W_in)
            else:
                raise NotImplementedError(
                    f"W_in property is not supported for MLP of type {type(mlp).__name__}"
                )
        return torch.stack(weights, dim=0)

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        weights: List[torch.Tensor] = []
        for block in chain(self.encoder, self.decoder):
            mlp = cast(T5Block, block).mlp
            if isinstance(mlp, (MLP, GatedMLP)):
                weights.append(mlp.W_out)
            else:
                raise NotImplementedError(
                    f"W_out property is not supported for MLP of type {type(mlp).__name__}"
                )
        return torch.stack(weights, dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.b_K for block in chain(self.encoder, self.decoder)],
            dim=0,
        )

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack(
            [cast(T5Block, block).attn.b_Q for block in chain(self.encoder, self.decoder)],
            dim=0,
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
            [cast(T5Block, block).attn.b_O for block in chain(self.encoder, self.decoder)],
            dim=0,
        )

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        biases: List[torch.Tensor] = []
        for block in chain(self.encoder, self.decoder):
            mlp = cast(T5Block, block).mlp
            if isinstance(mlp, (MLP, GatedMLP)):
                biases.append(mlp.b_in)
            else:
                raise NotImplementedError(
                    f"b_in property is not supported for MLP of type {type(mlp).__name__}"
                )
        return torch.stack(biases, dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        biases: List[torch.Tensor] = []
        for block in chain(self.encoder, self.decoder):
            mlp = cast(T5Block, block).mlp
            if isinstance(mlp, (MLP, GatedMLP)):
                biases.append(mlp.b_out)
            else:
                raise NotImplementedError(
                    f"b_out property is not supported for MLP of type {type(mlp).__name__}"
                )
        return torch.stack(biases, dim=0)

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
