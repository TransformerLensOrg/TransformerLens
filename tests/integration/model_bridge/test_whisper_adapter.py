"""Whisper Bridge parity on openai/whisper-tiny against an independent eager HF reference.

Slow: not in the CI cache (test_whisper_tiny.py keeps the same surfaces in CI on
random weights). The audio is a synthetic tone, so no dataset download.
"""

import numpy as np
import pytest
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "openai/whisper-tiny"
PARITY_ATOL = 1e-4
PROMPT = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def hf_eager():
    return WhisperForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).eval()


@pytest.fixture(scope="module")
def features(bridge):
    t = np.arange(16000) / 16000.0
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    extractor = bridge.processor.feature_extractor
    return extractor(audio, sampling_rate=16000, return_tensors="pt").input_features


@pytest.fixture(scope="module")
def decoder_ids(bridge):
    tokenizer = bridge.processor.tokenizer
    prompt = tokenizer.convert_tokens_to_ids(PROMPT)
    text = tokenizer(" Hello world.", add_special_tokens=False).input_ids
    return torch.tensor([prompt + text])


class TestWhisperBridgeCreation:
    def test_adapter_and_stacks(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.whisper import (
            WhisperArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, WhisperArchitectureAdapter)
        assert len(bridge.encoder_blocks) == 4
        assert len(bridge.decoder_blocks) == 4
        assert bridge.cfg.n_layers == 4
        assert bridge.cfg.n_ctx == 448

    def test_tokenizer_and_processor_loaded(self, bridge):
        assert bridge.tokenizer is not None
        assert isinstance(bridge.processor, WhisperProcessor)

    def test_features_are_log_mel_window(self, features):
        assert features.shape == (1, 80, 3000)


class TestWhisperForwardEquivalence:
    def test_logits_match_independent_hf_eager(self, bridge, hf_eager, features, decoder_ids):
        assert bridge.original_model is not hf_eager
        with torch.no_grad():
            bridge_logits = bridge(features, decoder_input=decoder_ids)
            hf_logits = hf_eager(input_features=features, decoder_input_ids=decoder_ids).logits
        torch.testing.assert_close(bridge_logits, hf_logits, rtol=0.0, atol=PARITY_ATOL)

    def test_k_proj_has_no_bias(self, bridge, hf_eager):
        """Whisper's k_proj is bias-free while q/v/out carry biases, on both stacks."""
        for i, block in enumerate(bridge.encoder_blocks):
            assert block.attn.k.original_component.bias is None
            assert hf_eager.model.encoder.layers[i].self_attn.k_proj.bias is None
            for proj in (block.attn.q, block.attn.v, block.attn.o):
                assert proj.original_component.bias is not None
        for i, block in enumerate(bridge.decoder_blocks):
            hf_layer = hf_eager.model.decoder.layers[i]
            for attn, hf_attn in (
                (block.self_attn, hf_layer.self_attn),
                (block.cross_attn, hf_layer.encoder_attn),
            ):
                assert attn.k.original_component.bias is None
                assert hf_attn.k_proj.bias is None
                for proj in (attn.q, attn.v, attn.o):
                    assert proj.original_component.bias is not None

    def test_shared_prompt_across_clips_matches_hf(self, bridge, hf_eager, features, decoder_ids):
        t = np.arange(16000) / 16000.0
        audio = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        extractor = bridge.processor.feature_extractor
        second = extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        clips = torch.cat([features, second])
        with torch.no_grad():
            logits = bridge(clips, decoder_input=decoder_ids)
            per_clip = torch.cat(
                [
                    hf_eager(input_features=clip[None], decoder_input_ids=decoder_ids).logits
                    for clip in clips
                ]
            )
        torch.testing.assert_close(logits, per_clip, rtol=0.0, atol=PARITY_ATOL)


class TestWhisperDecoderPrompt:
    def test_to_tokens_is_prefix_then_text(self, bridge):
        tokenizer = bridge.tokenizer
        tokens = bridge.to_tokens(" Hello world.")[0].tolist()
        text_ids = tokenizer(" Hello world.", add_special_tokens=False).input_ids
        assert tokens == tokenizer.prefix_tokens + text_ids
        assert tokenizer.eos_token_id not in tokens


class TestWhisperHooks:
    def test_encoder_and_decoder_hooks(self, bridge, features, decoder_ids):
        with torch.no_grad():
            _, cache = bridge.run_with_cache(features, decoder_input=decoder_ids)
        dec_len = decoder_ids.shape[1]
        assert cache["conv1.hook_out"].shape == (1, 384, 3000)
        assert cache["conv2.hook_out"].shape == (1, 384, 1500)
        assert cache["encoder_blocks.3.hook_out"].shape == (1, 1500, 384)
        assert cache["decoder_blocks.0.cross_attn.hook_pattern"].shape == (1, 6, dec_len, 1500)
        assert cache["decoder_blocks.3.hook_out"].shape == (1, dec_len, 384)


class TestWhisperGenerate:
    @pytest.fixture(scope="class")
    def two_clips(self, bridge, features):
        t = np.arange(16000) / 16000.0
        audio = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        extractor = bridge.processor.feature_extractor
        second = extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        return torch.cat([features, second])

    def test_greedy_matches_hf_whisper_generate(self, bridge, hf_eager, two_clips):
        """Same tokens as HF's Whisper generate, which also suppresses the checkpoint's
        suppress_tokens / begin_suppress_tokens and strips the prompt and end-of-text."""
        prompt = torch.tensor([bridge.processor.tokenizer.convert_tokens_to_ids(PROMPT)])
        eot = bridge.tokenizer.eos_token_id
        tokens = bridge.generate(
            two_clips, decoder_input=prompt, do_sample=False, max_new_tokens=20, verbose=False
        )
        with torch.no_grad():
            expected = hf_eager.generate(
                two_clips, language="en", task="transcribe", do_sample=False, max_new_tokens=20
            )
        expected = expected if isinstance(expected, torch.Tensor) else expected.sequences
        for ours, theirs in zip(tokens[:, prompt.shape[1] :].tolist(), expected.tolist()):
            ours = ours[: ours.index(eot)] if eot in ours else ours
            assert ours == [t for t in theirs if t != eot]

    def test_hf_generate_passes_features_through(self, bridge, hf_eager, features):
        with torch.no_grad():
            ours = bridge.hf_generate(
                features, language="en", task="transcribe", do_sample=False, max_new_tokens=20
            )
            expected = hf_eager.generate(
                features, language="en", task="transcribe", do_sample=False, max_new_tokens=20
            )
        expected = expected if isinstance(expected, torch.Tensor) else expected.sequences
        assert torch.equal(ours, expected)


class TestWhisperPromptAndTimestamps:
    """generate() builds HF's prompt; long audio goes through hf_generate, under the hooks."""

    @staticmethod
    def _generated(tokens, prompt_len, eot):
        rows = []
        for row in tokens[:, prompt_len:].tolist():
            rows.append(row[: row.index(eot)] if eot in row else row)
        return rows

    def test_default_prompt_detects_the_language_as_hf_does(self, bridge, hf_eager, features):
        tokens = bridge.generate(features, do_sample=False, max_new_tokens=20, verbose=False)
        detected = hf_eager.detect_language(input_features=features).tolist()
        tokenizer = bridge.processor.tokenizer
        prompt = [tokenizer.convert_tokens_to_ids("<|startoftranscript|>"), detected[0]]
        prompt += tokenizer.convert_tokens_to_ids(["<|transcribe|>", "<|notimestamps|>"])
        assert tokens[0, :4].tolist() == prompt
        with torch.no_grad():
            expected = hf_eager.generate(features, do_sample=False, max_new_tokens=20)
        eot = bridge.tokenizer.eos_token_id
        assert self._generated(tokens, 4, eot) == [
            [t for t in row if t != eot] for row in expected.tolist()
        ]

    def test_timestamps_match_hf(self, bridge, hf_eager, features):
        tokens = bridge.generate(
            features,
            language="en",
            return_timestamps=True,
            do_sample=False,
            max_new_tokens=20,
            verbose=False,
        )
        with torch.no_grad():
            expected = hf_eager.generate(
                features, language="en", return_timestamps=True, do_sample=False, max_new_tokens=20
            )
        expected = expected if isinstance(expected, torch.Tensor) else expected["sequences"]
        eot = bridge.tokenizer.eos_token_id
        assert self._generated(tokens, 3, eot) == [
            [t for t in row if t != eot] for row in expected.tolist()
        ]

    def test_long_form_hf_generate_fires_the_hooks_on_every_window(self, bridge, hf_eager):
        """45 s of audio: HF's sequential long-form decoding, which runs through the
        bridge's components, matches the unwrapped model."""
        t = np.arange(45 * 16000) / 16000.0
        audio = (0.5 * np.sin(2 * np.pi * (220.0 + 10.0 * t) * t)).astype(np.float32)
        inputs = bridge.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        assert inputs.input_features.shape[-1] > 3000
        kwargs = dict(
            attention_mask=inputs.attention_mask,
            language="en",
            return_timestamps=True,
            do_sample=False,
        )
        calls = []
        hook = ("encoder_blocks.0.hook_out", lambda tensor, hook: calls.append(1))
        with torch.no_grad(), bridge.hooks(fwd_hooks=[hook]):
            ours = bridge.hf_generate(inputs.input_features, **kwargs)
        with torch.no_grad():
            expected = hf_eager.generate(inputs.input_features, **kwargs)
        assert torch.equal(ours, expected)
        assert len(calls) >= 2


class TestWhisperCompatibilityMode:
    """Folded, centered weights on whisper-tiny: same predictions as HF."""

    @pytest.fixture(scope="class")
    def compat_bridge(self):
        bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
        bridge.enable_compatibility_mode(disable_warnings=True)
        return bridge

    def test_logits_match_hf_up_to_a_per_position_constant(
        self, compat_bridge, hf_eager, features, decoder_ids
    ):
        with torch.no_grad():
            logits = compat_bridge(features, decoder_input=decoder_ids)
            expected = hf_eager(input_features=features, decoder_input_ids=decoder_ids).logits

        def centered(x):
            return x - x.mean(dim=-1, keepdim=True)

        torch.testing.assert_close(centered(logits), centered(expected), rtol=0.0, atol=PARITY_ATOL)

    def test_transcription_unchanged(self, compat_bridge, hf_eager, features):
        prompt = torch.tensor([compat_bridge.tokenizer.convert_tokens_to_ids(PROMPT)])
        eot = compat_bridge.tokenizer.eos_token_id
        tokens = compat_bridge.generate(
            features, decoder_input=prompt, do_sample=False, max_new_tokens=20, verbose=False
        )
        with torch.no_grad():
            expected = hf_eager.generate(
                features, language="en", task="transcribe", do_sample=False, max_new_tokens=20
            )
        expected = expected if isinstance(expected, torch.Tensor) else expected.sequences
        ours = tokens[0, prompt.shape[1] :].tolist()
        ours = ours[: ours.index(eot)] if eot in ours else ours
        assert ours == [t for t in expected[0].tolist() if t != eot]
