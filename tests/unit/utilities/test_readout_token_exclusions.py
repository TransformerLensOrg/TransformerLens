"""Unicode-aware readout filtering over the full output vocabulary."""

import pytest

from transformer_lens.utilities import tokenize_utils


class Tokenizer:
    words = ["", "\n", "!!!", "心脏", "é", "٣", "<|im_end|>", "+", " word"]

    def decode(self, ids):
        return self.words[ids[0]] if ids[0] < len(self.words) else ""


class BatchTokenizer(Tokenizer):
    def batch_decode(self, rows):
        return [self.decode(row) for row in rows]


@pytest.mark.parametrize("tokenizer", [Tokenizer(), BatchTokenizer()])
def test_unicode_padding_and_special_tokens(tokenizer):
    assert tokenize_utils.readout_token_exclusions(tokenizer, 11) == [0, 1, 2, 7, 9, 10]
    assert tokenize_utils.readout_token_exclusions(tokenizer, 11, keep_token_ids=[7, 7, 3]) == [
        0,
        1,
        2,
        9,
        10,
    ]


@pytest.mark.parametrize("width", [0, -1, True])
def test_invalid_vocabulary_size(width):
    with pytest.raises(ValueError, match="vocab_size"):
        tokenize_utils.readout_token_exclusions(Tokenizer(), width)


@pytest.mark.parametrize("ids", [[-1], [11], [True]])
def test_invalid_retained_ids(ids):
    with pytest.raises(ValueError, match="keep_token_ids"):
        tokenize_utils.readout_token_exclusions(Tokenizer(), 11, keep_token_ids=ids)
