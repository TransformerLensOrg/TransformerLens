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
