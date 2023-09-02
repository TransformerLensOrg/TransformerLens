import types
from copy import deepcopy
from typing import Iterable, List, Optional, Union

from transformers import AutoTokenizer


def set_special_tokens(
    tokenizer,
):
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    return tokenizer


def get_tokenizer_with_manually_prepended_bos(tokenizer):
    def get_bos_prepended_input(input):
        if isinstance(input, str):
            return tokenizer.bos_token + input
        elif isinstance(input, Iterable):
            return [get_bos_prepended_input(i) for i in input]
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    def encode_plus(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        *args,
        **kwargs,
    ):
        text = get_bos_prepended_input(text)
        if text_pair:
            text_pair = get_bos_prepended_input(text_pair)
        return self.__encode_plus(text, text_pair, *args, **kwargs)

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        *args,
        **kwargs,
    ):
        batch_text_or_text_pairs = get_bos_prepended_input(batch_text_or_text_pairs)
        return self.__batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)

    tokenizer = deepcopy(tokenizer)

    # Monkey patch the tokenizer to manually prepend the bos token
    if not hasattr(tokenizer, "__encode_plus"):
        tokenizer.__encode_plus = tokenizer.encode_plus
        tokenizer.encode_plus = types.MethodType(encode_plus, tokenizer)
        tokenizer.__batch_encode_plus = tokenizer.batch_encode_plus
        tokenizer.batch_encode_plus = types.MethodType(batch_encode_plus, tokenizer)

    return tokenizer


def get_tokenizer_dict(tokenizer):
    init_kwargs = deepcopy(tokenizer.init_kwargs)
    pretrained_model_name_or_path = init_kwargs.pop("name_or_path")
    add_bos_token = init_kwargs.pop("add_bos_token", None)
    if add_bos_token is None:
        add_bos_token = getattr(tokenizer, "add_bos_token", False)

    if add_bos_token:
        tokenizer_with_bos = tokenizer
        tokenizer_without_bos = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, add_bos_token=False, **init_kwargs
        )
    else:
        tokenizer_without_bos = tokenizer
        tokenizer_with_bos = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, add_bos_token=True, **init_kwargs
        )

    tokenizer_with_bos = set_special_tokens(
        tokenizer_with_bos,
    )
    tokenizer_without_bos = set_special_tokens(
        tokenizer_without_bos,
    )

    # If the tokenizer doesn't automatically prepend a bos token
    # with add_bos_token=True, we need to manually prepend it
    if tokenizer_with_bos.encode("") == tokenizer_without_bos.encode(""):
        if len(tokenizer_with_bos.encode("")) == 0:
            tokenizer_with_bos = get_tokenizer_with_manually_prepended_bos(
                tokenizer_with_bos
            )
        else:
            raise ValueError(
                "tokenizer_without_bos canoot be properly set for this tokenizer."
            )

    return {
        True: tokenizer_with_bos,
        False: tokenizer_without_bos,
    }
