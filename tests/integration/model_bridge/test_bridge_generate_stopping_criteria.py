"""Tests for stop_strings and stopping_criteria on TransformerBridge.generate().

Coverage:
- stop_strings (single, list) halts generation early at the match.
- stopping_criteria (bare StoppingCriteria, list, StoppingCriteriaList) halts.
- The three signals (EOS, stop_strings, stopping_criteria) are independent: in
  particular stop_at_eos=False still stops on a stop string (the early-exit and
  finished-row padding must not be gated on stop_at_eos).
- Defaults (both None) are a byte-for-byte no-op (back-compat guard).
- Batched generation stops correctly.
- Error contracts: stop_strings without a tokenizer raises ValueError, a
  stopping_criteria callable needs no tokenizer, unsupported input paths raise
  NotImplementedError, and a bad stopping_criteria type raises TypeError.
- generate_stream honors the same parameters (shared _generate_tokens loop).

Uses distilgpt2 (CI-cached). Greedy (do_sample=False) for determinism, and
use_past_kv_cache=False so the tests stay robust on macOS-arm64 CI where the
cached-eager-attention path can NaN (issue #1322).
"""

import platform

import pytest
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

_MACOS_ARM64 = platform.system() == "Darwin" and platform.machine() == "arm64"

# Common kwargs for the greedy, macOS-safe, token-returning generate calls below.
_GEN = dict(do_sample=False, use_past_kv_cache=False, return_type="tokens", verbose=False)


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias the shared session fixture for concise test signatures."""
    return distilgpt2_bridge


@pytest.fixture()
def tokenizerless_bridge():
    """A private distilgpt2 bridge with its tokenizer removed.

    Function-scoped (a fresh boot per test) so removing the tokenizer never
    contaminates the shared session fixture.
    """
    from transformer_lens.model_bridge import TransformerBridge

    b = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    b.tokenizer = None
    return b


@pytest.fixture(scope="module")
def bridge_with_pad():
    """A private distilgpt2 bridge with eos set as the pad token, for batched tests.

    A dedicated boot (not the shared session fixture) so setting the pad token does not
    leak into other tests.
    """
    from transformer_lens.model_bridge import TransformerBridge

    b = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    if b.tokenizer.pad_token is None:
        b.tokenizer.pad_token = b.tokenizer.eos_token
    return b


class _StopAfterTotalLen(StoppingCriteria):
    """Stop once the running sequence reaches a fixed total length.

    Length-based so the stop step is deterministic and independent of what the
    model emits, which makes for exact assertions.
    """

    def __init__(self, total_len: int):
        self.total_len = total_len

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[-1] >= self.total_len


def _decode(bridge, out_row):
    return bridge.tokenizer.decode(out_row, skip_special_tokens=True)


def test_stop_string_halts_early(bridge):
    """A single stop string stops generation before max_new_tokens and appears in the output."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    out = bridge.generate(tokens, max_new_tokens=40, stop_strings=".", **_GEN)

    assert out.shape[1] > prompt_len, "should have generated at least one token"
    assert out.shape[1] < prompt_len + 40, "should have stopped before the token budget"
    assert "." in _decode(bridge, out[0]), "the stop string should be present in the output"


def test_stop_string_list(bridge):
    """A list of stop strings stops on whichever appears first, deterministically."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]
    stops = [".", ",", " the"]

    out1 = bridge.generate(tokens, max_new_tokens=40, stop_strings=stops, **_GEN)
    out2 = bridge.generate(tokens, max_new_tokens=40, stop_strings=stops, **_GEN)

    assert out1.shape[1] < prompt_len + 40
    assert any(s in _decode(bridge, out1[0]) for s in stops)
    assert torch.equal(out1, out2), "greedy generation with stop strings must be deterministic"


def test_custom_stopping_criteria_exact_stop(bridge):
    """A length-based StoppingCriteria stops at exactly the expected step."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    out = bridge.generate(
        tokens,
        max_new_tokens=20,
        stopping_criteria=_StopAfterTotalLen(prompt_len + 3),
        **_GEN,
    )

    assert out.shape[1] == prompt_len + 3


@pytest.mark.parametrize("wrap", ["bare", "list", "criteria_list"])
def test_stopping_criteria_accepted_forms(bridge, wrap):
    """stopping_criteria accepts a bare StoppingCriteria, a list, or a StoppingCriteriaList."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]
    criterion = _StopAfterTotalLen(prompt_len + 3)

    if wrap == "bare":
        stopping_criteria = criterion
    elif wrap == "list":
        stopping_criteria = [criterion]
    else:
        stopping_criteria = StoppingCriteriaList([criterion])

    out = bridge.generate(tokens, max_new_tokens=20, stopping_criteria=stopping_criteria, **_GEN)

    assert out.shape[1] == prompt_len + 3


def test_stop_at_eos_false_still_stops_on_string(bridge):
    """With stop_at_eos=False, a stop string still halts generation.

    The early-exit and finished-row padding are shared with the EOS path but must
    not be gated on stop_at_eos, otherwise a stop string could never end generation
    when EOS stopping is off.
    """
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    stopped = bridge.generate(
        tokens, max_new_tokens=40, stop_at_eos=False, stop_strings=".", **_GEN
    )
    baseline = bridge.generate(tokens, max_new_tokens=40, stop_at_eos=False, **_GEN)

    assert baseline.shape[1] == prompt_len + 40, "no stop signal should run to the full budget"
    assert stopped.shape[1] < prompt_len + 40, "the stop string should stop before the budget"
    assert "." in _decode(bridge, stopped[0])


def test_no_stopping_is_noop(bridge):
    """Passing both new params as None is byte-identical to not passing them."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    out_explicit_none = bridge.generate(
        tokens,
        max_new_tokens=5,
        stop_at_eos=False,
        stop_strings=None,
        stopping_criteria=None,
        **_GEN,
    )
    out_baseline = bridge.generate(tokens, max_new_tokens=5, stop_at_eos=False, **_GEN)

    assert torch.equal(out_explicit_none, out_baseline)
    assert out_explicit_none.shape[1] == prompt_len + 5, "nothing should stop generation"


def test_empty_stop_strings_is_noop(bridge):
    """An empty stop_strings list is a no-op rather than an error."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    out = bridge.generate(tokens, max_new_tokens=4, stop_at_eos=False, stop_strings=[], **_GEN)

    assert out.shape[1] == prompt_len + 4


def test_stopping_criteria_takes_precedence_over_unmet_stop_string(bridge):
    """When a stop string never appears, a criterion still stops generation."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    out = bridge.generate(
        tokens,
        max_new_tokens=20,
        stop_strings="zzzzzqqqqq",  # not emitted by greedy distilgpt2
        stopping_criteria=_StopAfterTotalLen(prompt_len + 3),
        **_GEN,
    )

    assert out.shape[1] == prompt_len + 3


def test_batched_generation_stops(bridge_with_pad):
    """Batched generation stops on stop strings, deterministically, for every row."""
    bridge = bridge_with_pad
    prompts = ["The capital of France is", "Hello, my name is"]
    tokens = bridge.to_tokens(prompts, prepend_bos=False, padding_side="left")
    prompt_len = tokens.shape[1]

    out1 = bridge.generate(tokens, max_new_tokens=40, stop_strings=".", **_GEN)
    out2 = bridge.generate(tokens, max_new_tokens=40, stop_strings=".", **_GEN)

    assert out1.shape[0] == 2, "batch dimension preserved"
    assert out1.shape[1] < prompt_len + 40, "the batch stopped before the budget"
    for i in range(2):
        assert "." in _decode(bridge, out1[i]), f"row {i} should contain its stop string"
    assert torch.equal(out1, out2), "batched greedy generation must be deterministic"


@pytest.mark.skipif(_MACOS_ARM64, reason="Upstream macOS-arm64 KV-cache NaN, see issue #1322.")
def test_stop_string_with_kv_cache(bridge):
    """stop_strings also works on the default KV-cache path (not only the no-cache path)."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    out = bridge.generate(
        tokens,
        max_new_tokens=40,
        stop_strings=".",
        do_sample=False,
        use_past_kv_cache=True,
        return_type="tokens",
        verbose=False,
    )

    assert out.shape[1] < prompt_len + 40
    assert "." in _decode(bridge, out[0])


def test_stop_strings_requires_tokenizer(tokenizerless_bridge):
    """stop_strings without a tokenizer raises a clear ValueError, but a criterion does not."""
    b = tokenizerless_bridge
    tokens = torch.tensor([[15496, 11, 314, 1101, 257]], dtype=torch.long)

    # stop_at_eos=False isolates the stop_strings tokenizer requirement from the
    # pre-existing eos-needs-a-tokenizer assertion.
    with pytest.raises(ValueError, match="tokenizer"):
        b.generate(tokens, max_new_tokens=3, stop_at_eos=False, stop_strings=".", **_GEN)

    # A token-based stopping_criteria needs no tokenizer and must still work.
    prompt_len = tokens.shape[1]
    out = b.generate(
        tokens,
        max_new_tokens=20,
        stop_at_eos=False,
        stopping_criteria=_StopAfterTotalLen(prompt_len + 3),
        **_GEN,
    )
    assert out.shape[1] == prompt_len + 3


def test_invalid_stopping_criteria_type_raises(bridge):
    """A stopping_criteria of the wrong type raises TypeError."""
    tokens = bridge.to_tokens("The quick brown")
    with pytest.raises(TypeError, match="StoppingCriteria"):
        bridge.generate(tokens, max_new_tokens=3, stopping_criteria=123, **_GEN)


def test_unsupported_input_path_raises(bridge):
    """stop_strings with an inputs_embeds input raises NotImplementedError pointing to hf_generate."""
    embeds = bridge.original_model.get_input_embeddings()(torch.tensor([[15496, 11, 314]])).float()
    with pytest.raises(NotImplementedError, match="hf_generate"):
        bridge.generate(embeds, max_new_tokens=3, stop_strings=".", verbose=False)


def test_generate_stream_honors_stop_strings(bridge):
    """generate_stream stops on a stop string too (shares the _generate_tokens loop)."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    chunks = list(
        bridge.generate_stream(
            tokens,
            max_new_tokens=40,
            stop_strings=".",
            do_sample=False,
            use_past_kv_cache=False,
            return_type="tokens",
            verbose=False,
        )
    )
    # First chunk includes the prompt, later chunks are deltas.
    # Concatenating gives the full prompt + generated sequence.
    full = torch.cat(chunks, dim=1)

    assert full.shape[1] < prompt_len + 40, "stream should have stopped before the budget"
    assert "." in _decode(bridge, full[0])


def test_stop_string_with_output_logits(bridge):
    """output_logits=True returns one logits entry per generated token, even on early stop."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    out = bridge.generate(tokens, max_new_tokens=40, stop_strings=".", output_logits=True, **_GEN)
    n_generated = out.sequences.shape[1] - prompt_len

    assert n_generated < 40, "should have stopped before the budget"
    assert len(out.logits) == n_generated, "one logits tensor per generated token"
    assert out.logits[0].shape[0] == 1, "batch dimension on the per-step logits"


def test_stop_string_return_type_str(bridge):
    """return_type='str' decodes the early-stopped sequence, which contains the stop string."""
    out = bridge.generate(
        "The quick brown",
        max_new_tokens=40,
        stop_strings=".",
        do_sample=False,
        use_past_kv_cache=False,
        return_type="str",
        verbose=False,
    )
    assert isinstance(out, str)
    assert "." in out


def test_malformed_stopping_criteria_shape_raises(bridge):
    """A criterion returning a non per-row shape gets a clear ValueError, not an opaque error."""

    class _BadShape(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.zeros(input_ids.shape[0], 1, dtype=torch.bool)  # [batch, 1], not [batch]

    tokens = bridge.to_tokens("The quick brown")
    with pytest.raises(ValueError, match="per-row bool"):
        bridge.generate(tokens, max_new_tokens=3, stopping_criteria=_BadShape(), **_GEN)


def test_batched_stopping_criteria_without_pad_token_raises(tokenizerless_bridge):
    """Batched stopping_criteria with no pad/eos id and stop_at_eos=False raises a clear error.

    Without a padding token, finished rows could not be frozen while the rest of the batch
    continues, so generate refuses rather than emitting token-0 padding.
    """
    b = tokenizerless_bridge
    batched = torch.tensor([[15496, 11, 314], [15496, 11, 314]], dtype=torch.long)
    with pytest.raises(ValueError, match="padding token"):
        b.generate(
            batched,
            max_new_tokens=5,
            stop_at_eos=False,
            stopping_criteria=_StopAfterTotalLen(5),
            **_GEN,
        )


# Unsupported-path guards. These exercise the NotImplementedError branches in
# generate() without booting a heavy enc-dec / multimodal / SSM model, by flipping the
# exact flag each guard reads on the existing distilgpt2 bridge. The guard fires before
# any architecture-specific generation code runs, so a stand-in model is sufficient.


def test_stop_strings_encoder_decoder_raises(bridge, monkeypatch):
    """stop_strings on an encoder-decoder model raises NotImplementedError."""
    monkeypatch.setattr(bridge.original_model.config, "is_encoder_decoder", True)
    tokens = bridge.to_tokens("The quick brown")
    with pytest.raises(NotImplementedError, match="encoder-decoder"):
        bridge.generate(tokens, max_new_tokens=3, stop_strings=".", verbose=False)


def test_stop_strings_multimodal_raises(bridge):
    """stop_strings with a multimodal input (pixel_values) raises NotImplementedError."""
    tokens = bridge.to_tokens("The quick brown")
    with pytest.raises(NotImplementedError, match="multimodal"):
        bridge.generate(
            tokens,
            max_new_tokens=3,
            stop_strings=".",
            pixel_values=torch.zeros(1, 3, 8, 8),
            verbose=False,
        )


def test_stop_strings_stateful_fallback_raises(bridge, monkeypatch):
    """A stateful/SSM model with use_past_kv_cache=False raises, pointing to the cached path.

    With use_past_kv_cache=False the stateful cache is off, so generate() would fall back to
    hf_generate() and drop the kwargs. The error points to use_past_kv_cache=True, which keeps
    generation on the hooked loop where stopping is applied.
    """
    monkeypatch.setattr(bridge.cfg, "is_stateful", True)
    tokens = bridge.to_tokens("The quick brown")
    with pytest.raises(NotImplementedError, match="use_past_kv_cache=True"):
        bridge.generate(
            tokens, max_new_tokens=3, stop_strings=".", use_past_kv_cache=False, verbose=False
        )


def test_generate_stream_stop_at_eos_false_stops_on_string(bridge):
    """generate_stream with stop_at_eos=False still stops on a stop string."""
    tokens = bridge.to_tokens("The quick brown")
    prompt_len = tokens.shape[1]

    chunks = list(
        bridge.generate_stream(
            tokens,
            max_new_tokens=40,
            stop_at_eos=False,
            stop_strings=".",
            do_sample=False,
            use_past_kv_cache=False,
            return_type="tokens",
            verbose=False,
        )
    )
    full = torch.cat(chunks, dim=1)

    assert full.shape[1] < prompt_len + 40, "stream should have stopped before the budget"
    assert "." in _decode(bridge, full[0])


def test_generate_stream_batched_without_pad_token_raises(tokenizerless_bridge):
    """Batched generate_stream stopping_criteria with no pad token and stop_at_eos=False raises."""
    b = tokenizerless_bridge
    batched = torch.tensor([[15496, 11, 314], [15496, 11, 314]], dtype=torch.long)
    stream = b.generate_stream(
        batched,
        max_new_tokens=5,
        stop_at_eos=False,
        stopping_criteria=_StopAfterTotalLen(5),
        do_sample=False,
        use_past_kv_cache=False,
        return_type="tokens",
        verbose=False,
    )
    with pytest.raises(ValueError, match="padding token"):
        list(stream)


def test_stop_at_eos_false_with_pad_token(bridge_with_pad):
    """Covers the pad_token_id arm of the no-eos padding fallback (a pad token is present)."""
    tokens = bridge_with_pad.to_tokens("The quick brown")
    out = bridge_with_pad.generate(
        tokens, max_new_tokens=40, stop_at_eos=False, stop_strings=".", **_GEN
    )
    assert out.shape[1] < tokens.shape[1] + 40
    assert "." in _decode(bridge_with_pad, out[0])
