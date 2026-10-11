"""Local prompt, loading-key, and lifecycle contracts for reconstruction."""

import pytest

from transformer_lens.tools.analysis._nla_artifacts import (
    ArtifactRef,
    NLAMetadata,
    NLAPairConfig,
)
from transformer_lens.tools.analysis._nla_reconstructor import (
    ReconstructorRuntime,
    prepare_critic_tokens,
    validate_loading_info,
)


@pytest.fixture
def metadata():
    return NLAMetadata(
        "ar",
        4,
        1,
        "marker",
        7,
        "{injection_char}",
        "Summary: {explanation} END",
        2.0,
        None,
        None,
        None,
        (8, 9),
    )


class Tokenizer:
    bos_token_id = 1

    def __init__(self):
        self.ids = [1, 3, 8, 9]
        self.calls = []

    def encode(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return self.ids


def test_raw_template_and_authored_whitespace(metadata):
    tokenizer = Tokenizer()
    tokens = prepare_critic_tokens(tokenizer, metadata, "  authored\n text  ", 4)
    assert tokenizer.calls == [("Summary:   authored\n text   END", True)]
    assert tokens.tolist() == [[1, 3, 8, 9]]


@pytest.mark.parametrize(
    "ids,message",
    [
        ([3, 4, 8, 9], "BOS"),
        ([1, 3, 9, 8], "suffix"),
        ([1, 2, 3, 8, 9], "context"),
        ([1, True, 8, 9], "integers"),
    ],
)
def test_prompt_guards(metadata, ids, message):
    tokenizer = Tokenizer()
    tokenizer.ids = ids
    with pytest.raises(ValueError, match=message):
        prepare_critic_tokens(tokenizer, metadata, "text", 4)


def test_lazy_runtime_and_cleanup():
    pair = NLAPairConfig(
        ArtifactRef("example/target", "a" * 40),
        ArtifactRef("example/av", "b" * 40),
        ArtifactRef("example/ar", "c" * 40),
        "qwen2",
        4,
        1,
        3,
    )
    runtime = ReconstructorRuntime(pair, local_files_only=True)
    assert runtime._model is None and runtime._head is None
    with pytest.raises(ValueError, match="nonempty"):
        runtime.reconstruct("")
    runtime.unload()
    runtime.unload()
    assert runtime._lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            runtime.reconstruct("text")
        with pytest.raises(RuntimeError, match="during"):
            runtime.unload()
    finally:
        runtime._lock.release()


def test_expected_missing_output_weights():
    validate_loading_info({"missing_keys": ["model.norm.weight", "lm_head.weight"]})


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"missing_keys": ["model.norm.weight"]},
        {"missing_keys": ["model.norm.weight", "lm_head.weight", "model.layers.0.weight"]},
        {"missing_keys": ["model.norm.weight", "lm_head.weight"], "unexpected_keys": ["other"]},
        {"missing_keys": ["model.norm.weight", "lm_head.weight"], "mismatched_keys": ["other"]},
        {"missing_keys": ["model.norm.weight", "lm_head.weight"], "error_msgs": ["error"]},
    ],
)
def test_loading_errors_are_not_suppressed(info):
    with pytest.raises(ValueError):
        validate_loading_info(info)
