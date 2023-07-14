import pytest

from transformer_lens import HookedTransformer


class TestPrependBos:
    prompt = "Hello world!"

    @pytest.fixture(scope="class")
    def model(self):
        return HookedTransformer.from_pretrained("gpt2-small")

    def test_default_prepend_bos_value(self):
        model = HookedTransformer.from_pretrained("gpt2-small")
        assert model.prepend_bos == True, "Default prepend_bos should be True"

    def test_set_prepend_bos(self, model):
        model.set_prepend_bos(False)
        assert model.prepend_bos == False, "prepend_bos should be set to False"

    def test_default_prepend_bos_at_method_level(self):
        model = HookedTransformer.from_pretrained("gpt2-small")

        logits = model(self.prompt)  # [batch pos d_vocab]
        str_tokens = model.to_str_tokens(self.prompt)
        tokens = model.to_tokens(self.prompt)

        assert logits.shape[1] == len(str_tokens) == tokens.shape[1]

    def test_set_prepend_bos_at_method_level(self, model):
        for global_prepend_bos in [True, False]:
            model.set_prepend_bos(global_prepend_bos)

            logits = model(self.prompt)  # [batch pos d_vocab]
            str_tokens = model.to_str_tokens(self.prompt)
            tokens = model.to_tokens(self.prompt)

            assert logits.shape[1] == len(str_tokens) == tokens.shape[1]

    def test_pass_prepend_bos_at_method_level(self, model):
        for global_prepend_bos in [True, False]:
            model.set_prepend_bos(global_prepend_bos)

            for local_prepend_bos in [True, False]:
                logits = model(
                    self.prompt, prepend_bos=local_prepend_bos
                )  # [batch pos d_vocab]
                str_tokens = model.to_str_tokens(
                    self.prompt, prepend_bos=local_prepend_bos
                )
                tokens = model.to_tokens(self.prompt, prepend_bos=local_prepend_bos)

                assert logits.shape[1] == len(str_tokens) == tokens.shape[1]

    def test_set_prepend_bos_with_get_token_position(self):
        model = HookedTransformer.from_pretrained("gpt2-small")

        bos_position = model.get_token_position(
            model.tokenizer.bos_token_id, self.prompt
        )
        assert bos_position == 0

        with pytest.raises(AssertionError):
            bos_position = model.get_token_position(
                model.tokenizer.bos_token_id, self.prompt, prepend_bos=False
            )

        model.set_prepend_bos(False)
        with pytest.raises(AssertionError):
            bos_position = model.get_token_position(
                model.tokenizer.bos_token_id, self.prompt
            )

        bos_position = model.get_token_position(
            model.tokenizer.bos_token_id, self.prompt, prepend_bos=True
        )
        assert bos_position == 0
