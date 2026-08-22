"""bridge.generate(str) on encoder-decoder models must tokenize with the
tokenizer's native recipe. to_tokens' decoder-style BOS policy injected a
stray <s> and dropped the trailing </s>, corrupting encoder input — m2m100
degenerated into token loops; Marian/T5 degraded silently.

A tiny-random M2M100 is used because its lang-code recipe genuinely differs
from to_tokens output (Marian's happens to coincide, so it cannot
discriminate); random weights are fine — greedy decoding is deterministic, so
outputs match iff the encoder input matches.
"""

import pytest
import torch

pytest.importorskip("transformers")


def test_m2m100_string_generation_matches_native_recipe():
    from transformer_lens.model_bridge import TransformerBridge

    try:
        bridge = TransformerBridge.boot_transformers(
            "hf-internal-testing/tiny-random-M2M100ForConditionalGeneration", device="cpu"
        )
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"tiny-random-m2m100 unavailable offline: {exc}")

    text = "Ik moet nu echt gaan slapen."
    native_ids = bridge.tokenizer(text, return_tensors="pt")["input_ids"]
    to_tokens_ids = bridge.to_tokens(text)
    assert (
        native_ids[0].tolist() != to_tokens_ids[0].tolist()
    ), "precondition: recipes must differ or this test cannot discriminate"
    # Assert on the tokens generate() actually consumed (random tiny weights
    # emit input-independent output, so generated text cannot discriminate).
    _, fed = bridge.generate(
        text, max_new_tokens=4, temperature=0.0, return_type="tokens", return_input_tokens=True
    )
    assert isinstance(fed, torch.Tensor)
    assert fed[0].tolist() == native_ids[0].tolist(), (fed[0].tolist(), native_ids[0].tolist())


def test_m2m100_batched_list_generation_matches_native_recipe():
    """The list-input branch had the same corruption (unpatched in the first
    fix): batched generate on M2M100/MBart fed to_tokens-mangled encoder
    input. Both rows must match the tokenizer's own padded batch encoding."""
    from transformer_lens.model_bridge import TransformerBridge

    try:
        bridge = TransformerBridge.boot_transformers(
            "hf-internal-testing/tiny-random-M2M100ForConditionalGeneration", device="cpu"
        )
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"tiny-random-m2m100 unavailable offline: {exc}")

    texts = ["Ik moet nu echt gaan slapen.", "Ik kan niet zo leven."]
    native = bridge.tokenizer(texts, return_tensors="pt", padding=True)["input_ids"]
    _, fed = bridge.generate(
        texts, max_new_tokens=4, temperature=0.0, return_type="tokens", return_input_tokens=True
    )
    assert isinstance(fed, torch.Tensor)
    assert fed.tolist() == native.tolist(), (fed.tolist(), native.tolist())


def test_generation_config_forced_bos_applied_by_default():
    """HF's generate() applies generation_config defaults; bart-large-cnn pins
    forced_bos_token_id=0 there and its summaries degrade without it. The
    bridge must honor the config value when the caller passes none."""
    from transformer_lens.model_bridge import TransformerBridge

    try:
        bridge = TransformerBridge.boot_transformers(
            "hf-internal-testing/tiny-random-M2M100ForConditionalGeneration", device="cpu"
        )
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"tiny-random-m2m100 unavailable offline: {exc}")

    forced = 7
    bridge.original_model.generation_config.forced_bos_token_id = forced
    out = bridge.generate(
        "Ik moet nu echt gaan slapen.", max_new_tokens=4, temperature=0.0, return_type="tokens"
    )
    assert out[0, 1].item() == forced


def test_generation_config_min_length_suppresses_early_eos():
    """bart-large-cnn pins min_length=56 in its generation config; HF's
    generate() suppresses EOS until then. Without it the bridge loop can EOS
    on step one and emit an empty summary (observed live, scored 0)."""
    from transformer_lens.model_bridge import TransformerBridge

    try:
        bridge = TransformerBridge.boot_transformers(
            "hf-internal-testing/tiny-random-M2M100ForConditionalGeneration", device="cpu"
        )
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"tiny-random-m2m100 unavailable offline: {exc}")

    from unittest import mock

    from transformer_lens import utilities as tl_utils

    eos = bridge.original_model.config.eos_token_id
    # Sample EOS whenever its logit is finite: the loop's -inf suppression is
    # then the ONLY thing that can delay it, so this discriminates exactly
    # that mechanism (tiny-random weights never prefer EOS on their own).
    real_sample = tl_utils.sample_logits

    def eos_greedy(logits, **kwargs):
        out = real_sample(logits, **kwargs)
        finite = torch.isfinite(logits[:, eos])
        out[finite] = eos
        return out

    bridge.original_model.generation_config.min_length = 10
    with mock.patch.object(tl_utils, "sample_logits", eos_greedy):
        out = bridge.generate(
            "Ik moet nu echt gaan slapen.",
            max_new_tokens=16,
            temperature=0.0,
            return_type="tokens",
            stop_at_eos=True,
        )
    decoder_part = out[0, 1:].tolist()
    # Without suppression EOS lands at decoder position 1; with it, no EOS
    # before the floor and EOS immediately after it lifts.
    assert not any(t == eos for t in decoder_part[:8]), decoder_part
    assert eos in decoder_part, decoder_part


def test_generation_config_no_repeat_ngram_applied():
    """bart-large-cnn pins no_repeat_ngram_size=3; HF applies it by default.
    Without it greedy decoding falls into a BOS attractor (observed live:
    empty summary, scored 0). Force an attractor token and assert the
    processor breaks the loop."""
    from unittest import mock

    from transformer_lens import utilities as tl_utils
    from transformer_lens.model_bridge import TransformerBridge

    try:
        bridge = TransformerBridge.boot_transformers(
            "hf-internal-testing/tiny-random-M2M100ForConditionalGeneration", device="cpu"
        )
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"tiny-random-m2m100 unavailable offline: {exc}")

    attractor = 5
    real_sample = tl_utils.sample_logits

    def prefer_attractor(logits, **kwargs):
        out = real_sample(logits, **kwargs)
        allowed = torch.isfinite(logits[:, attractor])
        out[allowed] = attractor
        return out

    bridge.original_model.generation_config.no_repeat_ngram_size = 2
    with mock.patch.object(tl_utils, "sample_logits", prefer_attractor):
        out = bridge.generate(
            "Ik moet nu echt gaan slapen.",
            max_new_tokens=8,
            temperature=0.0,
            return_type="tokens",
            stop_at_eos=False,
        )
    seq = out[0].tolist()
    runs = [seq[i] == seq[i + 1] == attractor for i in range(len(seq) - 1)]
    # A (5,5) bigram may occur once, but 5,5,5 requires repeating it — banned.
    assert not any(
        seq[i] == seq[i + 1] == seq[i + 2] == attractor for i in range(len(seq) - 2)
    ), seq


def test_batched_unequal_rows_match_solo_generation():
    """Id equality can't see mask handling: the batched enc-dec path fed
    native ids but no attention mask, so the short row of an unequal batch
    attended over pads. Greedy decoding of the short prompt must be identical
    batched and solo."""
    from transformer_lens.model_bridge import TransformerBridge

    try:
        bridge = TransformerBridge.boot_transformers(
            "hf-internal-testing/tiny-random-M2M100ForConditionalGeneration", device="cpu"
        )
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"tiny-random-m2m100 unavailable offline: {exc}")

    short = "Ik slaap."
    long = "Ik moet nu echt heel snel gaan slapen want het is al veel te laat geworden."
    # Logits-level: argmax can survive unmasked pads on a tiny model, the
    # step-0 distribution cannot.
    solo = bridge.generate(
        short, max_new_tokens=2, temperature=0.0, return_type="tokens", output_logits=True
    )
    batched = bridge.generate(
        [short, long], max_new_tokens=2, temperature=0.0, return_type="tokens", output_logits=True
    )
    solo_step0 = solo.logits[0][0]
    batched_step0_row0 = batched.logits[0][0]
    assert torch.allclose(solo_step0, batched_step0_row0, atol=1e-4), float(
        (solo_step0 - batched_step0_row0).abs().max()
    )
