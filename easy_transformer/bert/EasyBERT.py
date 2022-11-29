import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch as t
import torch.nn as nn
from torchtyping import TensorType as TT
from transformers import (
    AutoModelForMaskedLM,  # unfortunately the suggestion to import from the non-private location doesn't work; it makes [from_pretrained == None]
)
from transformers import (
    PreTrainedTokenizer,  # TODO why is this split up- move the comment around?
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from easy_transformer.hook_points import HookedRootModule

from .. import loading_from_pretrained as loading
from . import embeddings, encoder, encoder_layer
from .EasyBERTConfig import EasyBERTConfig  # TODO can we simplify this import?

# TODO share this type declaration with [EasyTransformer.py]
TokensTensor = TT["batch", "pos"]
InputForForwardLayer = Union[str, List[str], TokensTensor]


@dataclass
class Output:
    logits: TT["batch", "seq", "vocab"]
    hidden_states: TT[
        "n_layers", "batch", "seq", "hidden"
    ]  # slightly different than HF which stacks embeddings and hidden states
    embedding: TT["batch", "seq", "hidden"]


class EasyBERT(HookedRootModule):
    # written in this style because this guarantees that [self] isn't set inside here
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **model_kwargs,
    ):
        # TODO use these other parameters
        logging.info(f"Loading model: {model_name}")
        official_model_name = loading.get_official_model_name(model_name)
        config = EasyBERTConfig(
            n_layers=12,
            n_heads=12,
            hidden_size=768,
            dropout=0.0,
            model_name=official_model_name,
            d_vocab=30522,
            max_len=512,
            tokenizer_name=official_model_name,  # TODO change
        )  # TODO fancier :P
        assert (
            AutoModelForMaskedLM.from_pretrained is not None
        )  # recommended by some github link TODO find it
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
        cls, config: EasyBERTConfig, tokenizer: Optional[PreTrainedTokenizer]
    ):
        if tokenizer is not None:
            return tokenizer

        if config.tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            result: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name
            )
            result.eos_token = (
                result.eos_token if result.eos_token is not None else "<|endoftext|>"
            )
            result.pad_token = (
                result.pad_token if result.pad_token is not None else result.eos_token
            )
            result.bos_token = (
                result.bos_token if result.bos_token is not None else result.eos_token
            )
            return result
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens directly. In this case, we don't need a tokenizer.
            return None

    def __init__(self, config: EasyBERTConfig, tokenizer=None, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = EasyBERT.__generate_tokenizer__(self.config, tokenizer)
        self.embeddings = embeddings.Embeddings(config)
        self.encoder = encoder.Encoder(config)
        self.encoding_to_vocab = nn.Linear(config.hidden_size, config.d_vocab)
        self.softmax = nn.LogSoftmax(
            dim=-1
        )  # TODO why is this with [nn] instead of just [F.softmax]?

    # TODO utils?
    def __copy__(self, mine, state_dict, base_name):
        mine.load_state_dict(
            {
                "weight": state_dict[base_name + ".weight"],
                "bias": state_dict[base_name + ".bias"],
            }
        )

    def __load_embedding_state_dict__(self, state_dict: Dict[str, t.Tensor]):
        _copy_ = lambda mine, base_name: self.__copy__(mine, state_dict, base_name)
        _copy_(self.embeddings.word_embeddings, "bert.embeddings.word_embeddings")
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
        _copy_(self.encoding_to_vocab, "cls.predictions.transform.dense")
        # TODO what about the last layernorm here?
        # _copy_(self.encoding_to_vocab.ln, "cls.predictions.transform.LayerNorm")

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
        attention = self.encoder.layers[layer_index].attention
        assert isinstance(attention, encoder_layer.Attention)
        self_attention = attention.self_attention
        _copy_(mine=self_attention.w_q, name="self.query")
        _copy_(mine=self_attention.w_k, name="self.key")
        _copy_(mine=self_attention.w_v, name="self.value")
        _copy_(mine=self_attention.w_o, name="output.dense")
        _copy_(mine=self_attention.ln, name="output.LayerNorm")

        # copy mlp stuff
        mlp = self.encoder.layers[layer_index].mlp
        assert isinstance(mlp, encoder_layer.MLP)
        _copy_(mine=mlp.w_1, name="intermediate.dense")
        _copy_(mine=mlp.w_2, name="output.dense")
        _copy_(mine=mlp.ln, name="output.LayerNorm")

    def __load_encoder_state_dict__(self, hf_bert):
        for layer_index, hf_layer in zip(
            range(self.config.n_layers), hf_bert.encoder.layer
        ):
            self.__load_layer_state_dict__(layer_index, hf_layer)

    def load_and_process_state_dict(self, state_dict: Dict[str, t.Tensor]):
        self.__load_embedding_state_dict__(state_dict)
        self.__load_encoder_state_dict__(state_dict)
        self.__load_cls_state_dict__(state_dict)

    # TODO duplicated code, share it
    def __make_tokens_for_forward__(
        self, x: InputForForwardLayer, prepend_bos: bool
    ) -> TokensTensor:
        tokens: t.Tensor  # set inside the function body
        if type(x) == str or type(x) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # This is only intended to support passing in a single string
            # TODO why does this have type errors ! ! ! ! !: (?
            # TODO can we solve it in the other place too?
            tokens = self.to_tokens(x, prepend_bos=prepend_bos)
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

    def __to_segment_ids__(self, tokens: TokensTensor) -> t.Tensor:
        # TODO this is a bit hacky, but it works. We should probably make a proper segment id tensor
        # lol thanks copilot, which suggested zeros_like
        assert self.tokenizer is not None
        tokenizer: PreTrainedTokenizer = self.tokenizer  # for pylance
        result = tokenizer(tokens, return_tensors="pt", padding=True)["token_type_ids"]
        assert isinstance(result, t.Tensor)  # TODO is this ok? done for pylance
        return result

    def __make_segment_ids__(
        self, x: InputForForwardLayer, passed_segment_ids: Optional[TT["batch", "seq"]]
    ) -> TT["batch", "seq"]:
        # TODO is this right? :) copilot did it
        result: TT["batch", "seq"] = None
        if passed_segment_ids is None:
            if type(x) == str or type(x) == list:
                # If text, convert to tokens (batch_size=1)
                assert (
                    self.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                # This is only intended to support passing in a single string
                result = self.__to_segment_ids__(x)
            else:
                assert isinstance(x, t.Tensor)
        else:
            result = passed_segment_ids
        return result

    # TODO add [return_type] and maybe [prepend_bos] and maybe [past_kv_cache] and maybe [append_eos]
    def forward(
        self, x: InputForForwardLayer, segment_ids: TT["batch", "seq"] = None
    ) -> Output:
        # TODO document [segment_ids]
        # attention masking for padded token
        # t.ByteTensor([batch_size, 1, seq_len, seq_len)
        tokens = self.__make_tokens_for_forward__(
            x, prepend_bos=False
        )  # TODO really, always False?
        actual_segment_ids: TT["batch", "seq"] = self.__make_segment_ids__(
            x=x, passed_segment_ids=segment_ids
        )  # TODO prepend_bos=False?
        mask = (
            (tokens > 0).unsqueeze(1).repeat(1, tokens.size(1), 1).unsqueeze(1)
        )  # TODO is this right..?
        embedded = self.embeddings(
            tokens,
            actual_segment_ids,
        )  # TODO is there a way to make python complain about the variable named [input]?
        hidden_states = self.encoder(embedded, mask)
        last_hidden_state = hidden_states[-1]
        # TODO return both NSP and MLM logits
        logits = self.softmax(self.encoding_to_vocab(last_hidden_state))
        return Output(logits=logits, embedding=embedded, hidden_states=hidden_states)

    # TODO maybe change the order?
    def to_tokens(
        self,
        x: Union[str, List[str]],
        prepend_bos: bool = True,
        move_to_device: bool = True,
    ) -> TT["batch", "pos"]:  # TODO change this type
        """
        Converts a string to a tensor of tokens. If prepend_bos is True, prepends the BOS token to the input - this is recommended when creating a sequence of tokens to be input to a model.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when inputting a prompt to the model as the first token is often treated weirdly, but should only be done at the START of the prompt. Make sure to turn it off if you're looking at the tokenization of part of the prompt!
        (Note: some models eg GPT-2 were not trained with a BOS token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not careful!
        """
        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
        if prepend_bos:
            if isinstance(x, str):
                x = self.tokenizer.bos_token + x
            else:
                x = [self.tokenizer.bos_token + string for string in x]
        tokens = self.tokenizer(x, return_tensors="pt", padding=True)["input_ids"]
        if move_to_device:
            # TODO why did pylance not complain about [self.cfg]
            assert isinstance(tokens, t.Tensor)
            tokens = tokens.to(self.config.device)
        return tokens
