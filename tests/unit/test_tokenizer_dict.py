import pytest
import torch
from transformers import AutoTokenizer

from transformer_lens import HookedTransformer


class TestTokenizerDict:
    prompt = "Hello world!"
    prompts = ["Italy is in Europe.", "Seoul is the capital of South Korea."]

    # helper functions
    def get_num_tokens_in_prompt(self, model, prompt, intended_prepend_bos):
        tokenizer = AutoTokenizer.from_pretrained(
            model.tokenizer.name_or_path, add_bos_token=False
        )
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

    def check_tokens_length(
        self, model, logits, str_tokens, tokens, intended_prepend_bos
    ):
        expected_num_tokens = self.get_num_tokens_in_prompt(
            model, self.prompt, intended_prepend_bos
        )
        assert (
            logits.shape[1] == len(str_tokens) == tokens.shape[1] == expected_num_tokens
        )

    def check_prompt(self, model, intended_prepend_bos, overriding_prepend_bos=None):
        logits = model(
            self.prompt, prepend_bos=overriding_prepend_bos
        )  # [batch pos d_vocab]
        str_tokens = model.to_str_tokens(
            self.prompt, prepend_bos=overriding_prepend_bos
        )
        tokens = model.to_tokens(self.prompt, prepend_bos=overriding_prepend_bos)

        self.check_first_token(model, str_tokens, tokens, intended_prepend_bos)
        self.check_tokens_length(
            model, logits, str_tokens, tokens, intended_prepend_bos
        )

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
            prepend_bos=intended_prepend_bos,
            padding_side=intended_padding_side,
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
                assert (tokens == model.tokenizer.bos_token_id).sum() == tokens.shape[
                    0
                ], tokens
            else:
                assert (tokens == model.tokenizer.bos_token_id).sum() == 0, tokens

    # fixtures
    @pytest.fixture(
        scope="class",
        params=["gpt2-small", "facebook/opt-125m", "EleutherAI/pythia-14m"],
    )
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self, model_name):
        model = HookedTransformer.from_pretrained(model_name)
        return model

    # tests
    def test_defaults(self, model_name):
        intended_prepend_bos, intended_padding_side = True, "right"
        model = HookedTransformer.from_pretrained(model_name)

        assert (
            model.cfg.default_prepend_bos == intended_prepend_bos
        ), "Default prepend_bos should be True"
        assert (
            model.default_padding_side == intended_padding_side
        ), "Default padding_side should be right"

        self.check_prompt(model, intended_prepend_bos)
        self.check_prompts(model, intended_prepend_bos, intended_padding_side)

    def test_given_defaults(self, model_name):
        intended_prepend_bos, intended_padding_side = False, "left"
        model = HookedTransformer.from_pretrained(
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
    def test_changing_defaults(
        self, model, intended_prepend_bos, intended_padding_side
    ):
        model.default_padding_side = intended_padding_side
        model.cfg.default_prepend_bos = intended_prepend_bos

        self.check_prompt(model, intended_prepend_bos)
        self.check_prompts(model, intended_prepend_bos, intended_padding_side)

    @pytest.mark.parametrize("intended_prepend_bos", [True, False])
    @pytest.mark.parametrize("intended_padding_side", ["left", "right"])
    def test_overriding_defaults(
        self, model, intended_prepend_bos, intended_padding_side
    ):
        self.check_prompt(model, intended_prepend_bos, intended_prepend_bos)
        self.check_prompts(
            model,
            intended_prepend_bos,
            intended_padding_side,
            intended_prepend_bos,
            intended_padding_side,
        )
