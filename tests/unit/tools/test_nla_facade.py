"""Private facade lifecycle and import contracts without model loading."""

import pytest
import torch
from jaxtyping import TypeCheckError

from transformer_lens.tools.analysis import NaturalLanguageAutoencoder
from transformer_lens.tools.analysis._nla_artifacts import ArtifactRef, NLAPairConfig


@pytest.fixture
def nla():
    pair = NLAPairConfig(
        ArtifactRef("example/target", "a" * 40),
        ArtifactRef("example/av", "b" * 40),
        ArtifactRef("example/ar", "c" * 40),
        "qwen2",
        4,
        1,
        3,
    )
    return NaturalLanguageAutoencoder(pair, "synthetic", local_files_only=True)


def test_unknown_registry_fails_without_loading():
    with pytest.raises(ValueError, match="No verified"):
        NaturalLanguageAutoencoder.from_pretrained("unregistered", local_files_only=True)


def test_context_closes_on_error(nla):
    with pytest.raises(ValueError, match="example"):
        with nla:
            raise ValueError("example")
    assert nla._closed and nla._av._model is None and nla._ar._model is None
    nla.close()
    with pytest.raises(RuntimeError, match="closed"):
        nla.reconstruct("text")


def test_unload_is_idempotent_and_validated(nla):
    nla.unload("av")
    nla.unload("av")
    nla.unload("ar")
    with pytest.raises((ValueError, TypeCheckError)):
        nla.unload("invalid")
    assert not nla._closed


def test_operations_reject_concurrency(nla):
    assert nla._lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            nla.load_target()
        with pytest.raises(RuntimeError, match="during"):
            nla.close()
    finally:
        nla._lock.release()


def test_score_and_invalid_input(nla):
    assert nla.score(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])).normalized_mse == 2
    with pytest.raises(ValueError, match="nonempty"):
        nla.reconstruct("")
    assert nla._ar._model is None


def test_in_memory_tokenizer_bos_does_not_require_loading_path():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from transformer_lens.utilities.tokenize_utils import get_tokenizer_with_bos

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"UNK": 0, "BOS": 1}, unk_token="UNK")),
        unk_token="UNK",
        bos_token="BOS",
    )
    assert get_tokenizer_with_bos(tokenizer) is tokenizer


def test_public_support_export():
    import transformer_lens.tools.analysis as analysis
    from transformer_lens.tools.analysis.nla import (
        NaturalLanguageAutoencoder,
        NLAActivation,
    )

    assert analysis.NaturalLanguageAutoencoder is NaturalLanguageAutoencoder
    assert analysis.NLAActivation is NLAActivation
