import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType as TT

# to understand the reason for the [type: ignore] directive, see stuff like https://github.com/huggingface/transformers/issues/18464
from transformers import AutoModelForMaskedLM  # type: ignore
from transformers import PreTrainedTokenizer  # type: ignore
from transformers.models.auto.tokenization_auto import AutoTokenizer

import easy_transformer.utils as utils
from easy_transformer.hook_points import HookedRootModule

from .. import loading_from_pretrained as loading
from ..components import LayerNorm
from . import attention, embeddings, encoder, encoder_layer
from .config import Config

TokensTensor = TT["batch", "pos"]
InputForForwardLayer = Union[str, List[str], TokensTensor]
Loss = TT[()]


@dataclass
class Output:
    logits: TT["batch", "seq", "vocab"]
    embedding: TT["batch", "seq", "hidden"]
    loss: Loss


class EasyBERT(HookedRootModule):
    # written in this style because this guarantees that [self] isn't set inside here
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **model_kwargs,
    ):
        logging.info(f"Loading model: {model_name}")
        official_model_name = loading.get_official_model_name(model_name)
        config_dictionary = loading.convert_hf_model_config(official_model_name)
        # There are some keys we don't care about. TODO maybe-someday clean the out of [loading.convert_hf_model_config]
        keys_to_delete = [
            "normalization_type",
            "original_architecture",
        ]
        for key in keys_to_delete:
            if key in config_dictionary:
                del config_dictionary[key]
        config = Config(**config_dictionary)
        assert AutoModelForMaskedLM.from_pretrained is not None
        state_dict = AutoModelForMaskedLM.from_pretrained(
            official_model_name
        ).state_dict()
        model = cls(config, **model_kwargs)
        model.load_and_process_state_dict(state_dict)
        logging.info(
            f"Finished loading pretrained model {model_name} into EasyTransformer!"
        )

        return model

    @classmethod
    def __generate_tokenizer__(
        cls, config: Config, tokenizer: Optional[PreTrainedTokenizer]
    ):
        if tokenizer is not None:
            return tokenizer

        if config.tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            result: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name
            )
            return result
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens directly. In this case, we don't need a tokenizer.
            return None

    def __init__(self, config: Config, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = EasyBERT.__generate_tokenizer__(self.config, tokenizer)
        self.embeddings = embeddings.Embeddings(config)
        self.encoder = encoder.Encoder(config)
        self.out_linear = nn.Linear(config.d_model, config.d_model)
        self.out_ln = LayerNorm(cfg=config)  # type: ignore
        self.unembed = nn.parameter.Parameter(t.zeros(config.vocab_size))
        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    def __copy__(self, mine, state_dict, base_name):
        # TODO cleanup- duplicated code
        if hasattr(mine, "weight"):
            mine.weight.detach().copy_(state_dict[base_name + ".weight"])
            if base_name + ".bias" in state_dict:
                mine.bias.detach().copy_(state_dict[base_name + ".bias"])
        else:
            # TODO hack because layer norm uses w/b instead of weight/bias
            mine.w.detach().copy_(state_dict[base_name + ".weight"])
            if base_name + ".bias" in state_dict:
                mine.b.detach().copy_(state_dict[base_name + ".bias"])

    def __load_embedding_state_dict__(self, state_dict: Dict[str, t.Tensor]):
        _copy_ = lambda mine, base_name: self.__copy__(mine, state_dict, base_name)
        _copy_(
            self.embeddings.word_embeddings,
            "bert.embeddings.word_embeddings",
        )
        _copy_(
            self.embeddings.position_embeddings, "bert.embeddings.position_embeddings"
        )
        _copy_(
            self.embeddings.token_type_embeddings,
            "bert.embeddings.token_type_embeddings",
        )
        _copy_(self.embeddings.ln, "bert.embeddings.LayerNorm")

    def __load_cls_state_dict__(self, state_dict: Dict[str, t.Tensor]):
        """
        'cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', ])"""
        _copy_ = lambda mine, base_name: self.__copy__(mine, state_dict, base_name)
        _copy_(self.out_linear, "cls.predictions.transform.dense")
        _copy_(self.out_ln, "cls.predictions.transform.LayerNorm")

    def __load_layer_state_dict__(
        self,
        layer_index: int,
        state_dict: Dict[str, t.Tensor],
    ):
        base_name = f"bert.encoder.layer.{layer_index}"
        _copy_ = lambda mine, name: self.__copy__(
            mine, state_dict, base_name + "." + name
        )

        # copy attention stuff
        attention_output = self.encoder.layers[layer_index].attention
        assert isinstance(attention_output, attention.Attention)
        self_attention = attention_output.self_attention
        _copy_(mine=self_attention.w_q, name="attention.self.query")
        _copy_(mine=self_attention.w_k, name="attention.self.key")
        _copy_(mine=self_attention.w_v, name="attention.self.value")
        _copy_(mine=self_attention.w_o, name="attention.output.dense")

        # copy intermediate layer norm
        _copy_(mine=attention_output.ln, name="attention.output.LayerNorm")

        # copy mlp stuff
        mlp = self.encoder.layers[layer_index].mlp
        assert isinstance(mlp, encoder_layer.MLP)
        _copy_(mine=mlp.w_1, name="intermediate.dense")
        _copy_(mine=mlp.w_2, name="output.dense")
        _copy_(mine=mlp.ln, name="output.LayerNorm")

    def __load_encoder_state_dict__(self, state_dict):
        for layer_index in range(self.config.layers):
            self.__load_layer_state_dict__(layer_index, state_dict=state_dict)

    def load_and_process_state_dict(self, state_dict: Dict[str, t.Tensor]):
        self.__load_embedding_state_dict__(state_dict)
        self.__load_encoder_state_dict__(state_dict)
        self.__load_cls_state_dict__(state_dict)
        self.unembed.detach().copy_(state_dict["cls.predictions.bias"])

    def __make_tokens_for_forward__(
        self,
        x: InputForForwardLayer,
    ) -> TokensTensor:
        tokens: t.Tensor  # set inside the function body
        if type(x) == str or type(x) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # This is only intended to support passing in a single string
            tokens = self.to_tokens(x)
        else:
            assert isinstance(
                x, t.Tensor
            )  # typecast; we know that this is ok because of the above logic
            tokens = x
        if len(tokens.shape) == 1:
            # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
            tokens = tokens[None]
        if tokens.device.type != self.config.device:
            tokens = tokens.to(self.config.device)
        assert isinstance(tokens, t.Tensor)
        return tokens

    def __make_segment_ids__(self, x: InputForForwardLayer) -> TT["batch", "seq"]:
        assert (
            self.tokenizer is not None
        ), "Must provide a tokenizer if passing a string to the model"
        return self.tokenizer(x, return_tensors="pt", padding=True)["token_type_ids"]

    def forward(
        self,
        x: InputForForwardLayer,
        segment_ids: TT["batch", "seq"] = None,
    ) -> Output:
        # attention masking for padded token
        tokens: TokensTensor = self.__make_tokens_for_forward__(x)
        actual_segment_ids: TT["batch", "seq"] = (
            self.__make_segment_ids__(x=x) if segment_ids is None else segment_ids
        )
        mask = None  # no need for the mask because we're not doing any padding
        embedded = self.embeddings(
            tokens,
            actual_segment_ids,
        )
        last_hidden_state: TT["batch", "seq", "hidden"] = self.encoder(
            embedded, mask=mask
        )
        output = self.out_linear(last_hidden_state)
        output = F.gelu(output)
        output = self.out_ln(output)
        output = t.einsum("vh,bsh->bsv", self.embeddings.word_embeddings.weight, output)
        logits = output + self.unembed
        loss = utils.lm_cross_entropy_loss(logits=logits, tokens=tokens)
        return Output(
            embedding=embedded,
            logits=logits,
            loss=loss,
        )

    def to_tokens(
        self,
        x: Union[str, List[str]],
        move_to_device: bool = True,
    ) -> TokensTensor:
        """
        Converts a string to a tensor of tokens.

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not careful!
        """
        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
        tokens = self.tokenizer(x, return_tensors="pt", padding=True)["input_ids"]
        if move_to_device:
            assert isinstance(tokens, t.Tensor)
            tokens = tokens.to(self.config.device)
        return tokens

    def to_str_tokens(
        self,
        input: Union[str, Union[TT["pos"], TT[1, "pos"]], list],
        prepend_bos: bool = True,
    ) -> List[str]:
        """Method to map text, a list of text or tokens to a list of tokens as strings

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when inputting a prompt to the model as the first token is often treated weirdly, but should only be done at the START of the prompt. Make sure to turn it off if you're looking at the tokenization of part of the prompt!
        (Note: some models eg GPT-2 were not trained with a BOS token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not careful!

        Args:
            input (Union[str, list, torch.Tensor]): The input - either a string or a tensor of tokens. If tokens, should be a tensor of shape [pos] or [1, pos]
            prepend_bos (bool, optional): Whether to prepend a BOS token. Only applies if input is a string. Defaults to True.

        Returns:
            str_tokens: List of individual tokens as strings
        """
        if isinstance(input, list):
            return list(
                map(lambda tokens: self.to_str_tokens(tokens, prepend_bos), input)
            )  # type: ignore
        elif isinstance(input, str):
            tokens = self.to_tokens(input)[0]
        elif isinstance(input, t.Tensor):
            tokens = input
            tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
            assert (
                tokens.dim() == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        elif isinstance(input, np.ndarray):
            tokens = input
            tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
            assert (
                tokens.ndim == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        else:
            raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
        assert (
            self.tokenizer is not None
        ), "Cannot use to_str_tokens without a tokenizer"
        str_tokens = self.tokenizer.batch_decode(
            tokens, clean_up_tokenization_spaces=False
        )
        return str_tokens
