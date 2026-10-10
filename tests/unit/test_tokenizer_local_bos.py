"""BOS setup preserves locally constructed tokenizers without a reload source."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from transformer_lens.utilities.tokenize_utils import get_tokenizer_with_bos


@pytest.mark.parametrize("automatic_bos", [False, True])
def test_local_tokenizer_preserves_its_configured_special_token_behavior(automatic_bos):
    backend = Tokenizer(WordLevel({"<unk>": 0, "<bos>": 1, "word": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    if automatic_bos:
        backend.post_processor = TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", 1)]
        )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>", padding_side="left"
    )
    assert not tokenizer.name_or_path
    assert "name_or_path" not in tokenizer.init_kwargs
    before = tokenizer.backend_tokenizer.to_str()
    result = get_tokenizer_with_bos(tokenizer)
    assert result is tokenizer
    assert result.padding_side == "left"
    assert tokenizer.backend_tokenizer.to_str() == before
    assert result.encode("word", add_special_tokens=True) == ([1, 2] if automatic_bos else [2])
