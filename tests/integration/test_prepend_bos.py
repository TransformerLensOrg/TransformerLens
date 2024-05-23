import pytest
from transformers import AutoTokenizer

from transformer_lens import HookedTransformer


class TestPrependBos:
    prompt = "Hello world!"

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

    def check_tokens_length(self, model, logits, str_tokens, tokens, intended_prepend_bos):
        expected_num_tokens = self.get_num_tokens_in_prompt(
            model, self.prompt, intended_prepend_bos
        )
        assert logits.shape[1] == len(str_tokens) == tokens.shape[1] == expected_num_tokens

    # fixtures
    @pytest.fixture(scope="class", params=["gpt2", "facebook/opt-125m"])
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self, model_name):
        return HookedTransformer.from_pretrained(model_name)

    # tests
    def test_default_prepend_bos(self, model_name):
        intended_prepend_bos = True

        model = HookedTransformer.from_pretrained(model_name)
        assert (
            model.cfg.default_prepend_bos == intended_prepend_bos
        ), "Default prepend_bos should be True"

        logits = model(self.prompt)  # [batch pos d_vocab]
        str_tokens = model.to_str_tokens(self.prompt)
        tokens = model.to_tokens(self.prompt)  # [batch pos]

        self.check_first_token(model, str_tokens, tokens, intended_prepend_bos)
        self.check_tokens_length(model, logits, str_tokens, tokens, intended_prepend_bos)

        bos_position = model.get_token_position(model.tokenizer.bos_token_id, self.prompt)
        assert bos_position == 0

    def test_default_prepend_bos_to_false(self, model_name):
        intended_prepend_bos = False

        model = HookedTransformer.from_pretrained(
            model_name, default_prepend_bos=intended_prepend_bos
        )

        logits = model(self.prompt)  # [batch pos d_vocab]
        str_tokens = model.to_str_tokens(self.prompt)
        tokens = model.to_tokens(self.prompt)

        self.check_first_token(model, str_tokens, tokens, intended_prepend_bos)
        self.check_tokens_length(model, logits, str_tokens, tokens, intended_prepend_bos)

    @pytest.mark.parametrize("intended_prepend_bos", [True, False])
    def test_override_prepend_bos(self, model, intended_prepend_bos):
        for default_prepend_bos in [True, False]:
            model.cfg.default_prepend_bos = default_prepend_bos

            logits = model(self.prompt, prepend_bos=intended_prepend_bos)  # [batch pos d_vocab]
            str_tokens = model.to_str_tokens(self.prompt, prepend_bos=intended_prepend_bos)
            tokens = model.to_tokens(self.prompt, prepend_bos=intended_prepend_bos)

            self.check_first_token(model, str_tokens, tokens, intended_prepend_bos)
            self.check_tokens_length(model, logits, str_tokens, tokens, intended_prepend_bos)

    def test_prepend_bos_with_get_token_position(self, model_name):
        model = HookedTransformer.from_pretrained(model_name)

        bos_position = model.get_token_position(model.tokenizer.bos_token_id, self.prompt)
        assert bos_position == 0

        with pytest.raises(AssertionError):
            bos_position = model.get_token_position(
                model.tokenizer.bos_token_id, self.prompt, prepend_bos=False
            )

        model.cfg.default_prepend_bos = False
        with pytest.raises(AssertionError):
            bos_position = model.get_token_position(model.tokenizer.bos_token_id, self.prompt)

        bos_position = model.get_token_position(
            model.tokenizer.bos_token_id, self.prompt, prepend_bos=True
        )
        assert bos_position == 0

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
