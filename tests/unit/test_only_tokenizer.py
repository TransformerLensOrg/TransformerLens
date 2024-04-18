import logging
from typing import Dict

import pytest
from transformers import AutoTokenizer

import transformer_lens.loading_from_pretrained as loading
from transformer_lens import HookedTransformer, HookedTransformerConfig


class TokenizerOnlyHookedTransformer(HookedTransformer):
    def __init__(
        self,
        cfg,
        tokenizer=None,
        move_to_device=True,
        default_padding_side="right",
    ):
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a "
                "pretrained model, use HookedTransformer.from_pretrained() instead."
            )
        self.cfg = cfg

        if tokenizer is not None:
            self.set_tokenizer(tokenizer, default_padding_side=default_padding_side)
        elif self.cfg.tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            self.set_tokenizer(
                AutoTokenizer.from_pretrained(self.cfg.tokenizer_name, add_bos_token=True),
                default_padding_side=default_padding_side,
            )
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens
            # directly. In this case, we don't need a tokenizer.
            assert self.cfg.d_vocab != -1, "Must provide a tokenizer if d_vocab is not provided"
            self.tokenizer = None
            if default_padding_side != "right":
                logging.warning(
                    "default_padding_side is explictly given but ignored because tokenizer is not set."
                )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        refactor_factored_attn_matrices=False,
        checkpoint_index=None,
        checkpoint_value=None,
        hf_model=None,
        device=None,
        n_devices=1,
        tokenizer=None,
        move_to_device=True,
        fold_value_biases=True,
        default_prepend_bos=True,
        default_padding_side="right",
        **from_pretrained_kwargs,
    ) -> "TokenizerOnlyHookedTransformer":
        # Get the model name used in HuggingFace, rather than the alias.
        official_model_name = loading.get_official_model_name(model_name)

        # Load the config into an HookedTransformerConfig object. If loading from a
        # checkpoint, the config object will contain the information about the
        # checkpoint
        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=fold_ln,
            device=device,
            n_devices=n_devices,
            default_prepend_bos=default_prepend_bos,
            **from_pretrained_kwargs,
        )

        # Create the HookedTransformer object
        model = cls(
            cfg,
            tokenizer,
            move_to_device=False,
            default_padding_side=default_padding_side,
        )

        return model


class TestTokenizer:
    prompt = "Hello world!"
    prompts = ["Italy is in Europe.", "Seoul is the capital of South Korea."]

    # helper functions
    def get_num_tokens_in_prompt(self, model, prompt, intended_prepend_bos):
        tokenizer = AutoTokenizer.from_pretrained(model.tokenizer.name_or_path, add_bos_token=False)
        tokens = tokenizer(
            prompt,
        )["input_ids"]

        return len(tokens) + int(intended_prepend_bos)

    def check_first_token(self, model, str_tokens, tokens, intended_prepend_bos):
        if intended_prepend_bos:
            assert str_tokens[0] == model.tokenizer.bos_token
            assert tokens[0][0] == model.tokenizer.bos_token_id
        else:
            assert str_tokens[0] != model.tokenizer.bos_token
            assert tokens[0][0] != model.tokenizer.bos_token_id

    def check_tokens_length(self, model, str_tokens, tokens, intended_prepend_bos):
        expected_num_tokens = self.get_num_tokens_in_prompt(
            model, self.prompt, intended_prepend_bos
        )

        assert len(str_tokens) == tokens.shape[1] == expected_num_tokens

    def check_prompt(self, model, intended_prepend_bos, overriding_prepend_bos=None):
        str_tokens = model.to_str_tokens(self.prompt, prepend_bos=overriding_prepend_bos)
        tokens = model.to_tokens(self.prompt, prepend_bos=overriding_prepend_bos)

        self.check_first_token(model, str_tokens, tokens, intended_prepend_bos)
        self.check_tokens_length(model, str_tokens, tokens, intended_prepend_bos)

    def check_prompts(
        self,
        model,
        intended_prepend_bos,
        intended_padding_side,
        overriding_prepend_bos=None,
        overriding_padding_side=None,
    ):
        tokens = model.to_tokens(
            self.prompts,
            prepend_bos=overriding_prepend_bos,
            padding_side=overriding_padding_side,
        )
        if intended_padding_side == "left":
            if model.tokenizer.pad_token_id != model.tokenizer.bos_token_id:
                assert (tokens[:, 0] == model.tokenizer.pad_token_id).sum() == 1, tokens
                assert (tokens[:, 0] == model.tokenizer.bos_token_id).sum() == (
                    1 if intended_prepend_bos else 0
                ), tokens
            else:
                assert (tokens[:, 0] == model.tokenizer.pad_token_id).sum() == (
                    tokens.shape[0] if intended_prepend_bos else 1
                ), tokens
        else:
            assert (tokens[:, -1] == model.tokenizer.pad_token_id).sum() == 1, tokens
            if intended_prepend_bos:
                assert (tokens[:, 0] == model.tokenizer.bos_token_id).all(), tokens

        if model.tokenizer.pad_token_id != model.tokenizer.bos_token_id:
            if intended_prepend_bos:
                assert (tokens == model.tokenizer.bos_token_id).sum() == tokens.shape[0], tokens
            else:
                assert (tokens == model.tokenizer.bos_token_id).sum() == 0, tokens

    # fixtures
    @pytest.fixture(
        scope="class",
        params=[
            "gpt2-small",
            "facebook/opt-125m",
            "EleutherAI/pythia-14m",
            "EleutherAI/gpt-j-6b",
        ],
    )
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self, model_name):
        model = TokenizerOnlyHookedTransformer.from_pretrained(model_name)
        return model

    # tests
    def test_defaults(self, model_name):
        intended_prepend_bos, intended_padding_side = True, "right"
        model = TokenizerOnlyHookedTransformer.from_pretrained(model_name)

        assert (
            model.cfg.default_prepend_bos == intended_prepend_bos
        ), "Default prepend_bos should be True"
        assert (
            model.tokenizer.padding_side == intended_padding_side
        ), "Default padding_side should be right"

        self.check_prompt(model, intended_prepend_bos)
        self.check_prompts(model, intended_prepend_bos, intended_padding_side)

    def test_given_defaults(self, model_name):
        intended_prepend_bos, intended_padding_side = False, "left"
        model = TokenizerOnlyHookedTransformer.from_pretrained(
            model_name, default_prepend_bos=intended_prepend_bos
        )
        if model.tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.padding_side = intended_padding_side
            model.tokenizer = tokenizer
        else:
            model.tokenizer.padding_side = intended_padding_side

        self.check_prompt(model, intended_prepend_bos)
        self.check_prompts(model, intended_prepend_bos, intended_padding_side)

    @pytest.mark.parametrize("intended_prepend_bos", [True, False])
    @pytest.mark.parametrize("intended_padding_side", ["left", "right"])
    def test_changing_defaults(self, model, intended_prepend_bos, intended_padding_side):
        model.tokenizer.padding_side = intended_padding_side
        model.cfg.default_prepend_bos = intended_prepend_bos

        self.check_prompt(model, intended_prepend_bos)
        self.check_prompts(model, intended_prepend_bos, intended_padding_side)

    @pytest.mark.parametrize("intended_prepend_bos", [True, False])
    @pytest.mark.parametrize("intended_padding_side", ["left", "right"])
    def test_overriding_defaults(self, model, intended_prepend_bos, intended_padding_side):
        self.check_prompt(model, intended_prepend_bos, intended_prepend_bos)
        self.check_prompts(
            model,
            intended_prepend_bos,
            intended_padding_side,
            intended_prepend_bos,
            intended_padding_side,
        )

    def test_same_tokenization(self, model):
        prompt = self.prompt
        prompts = [
            "Italy is in Europe.",
            "Pyeongchang Olympics was held in 2018",
            "2023-09-09",
            "287594812673495",
        ]

        model.tokenizer.padding_side = "right"

        for input in [prompt, prompts]:
            tokens_with_bos = model.to_tokens(input, prepend_bos=True)
            tokens_without_bos = model.to_tokens(input, prepend_bos=False)
            assert tokens_with_bos[..., 1:].equal(tokens_without_bos)

            str_tokens_with_bos = model.to_str_tokens(input, prepend_bos=True)
            str_tokens_without_bos = model.to_str_tokens(input, prepend_bos=False)

            if isinstance(str_tokens_with_bos[0], list):
                for i in range(len(str_tokens_with_bos)):
                    assert str_tokens_with_bos[i][1:] == str_tokens_without_bos[i]
            else:
                assert str_tokens_with_bos[1:] == str_tokens_without_bos
