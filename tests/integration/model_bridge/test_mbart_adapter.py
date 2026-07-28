"""Integration tests for the MBart architecture adapter.

Uses ai4bharat/IndicBART (244M) — the smallest symmetric MBart checkpoint;
sshleifer/tiny-mbart is asymmetric and mislabeled as Bart.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "ai4bharat/IndicBART"


@pytest.fixture(scope="module")
def mbart_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_inputs(mbart_bridge):
    tokens = mbart_bridge.tokenizer("Hello world, this is a test.", return_tensors="pt")
    config = mbart_bridge.original_model.config
    # MBart decodes from EOS + language code; IndicBART leaves start unset.
    start = config.decoder_start_token_id
    if start is None:
        start = config.eos_token_id
    decoder_input_ids = torch.tensor([[start, 5, 6, 7]])
    return tokens.input_ids, tokens.attention_mask, decoder_input_ids


class TestMBartBridgeCreation:
    def test_adapter_and_structure(self, mbart_bridge):
        from transformer_lens.model_bridge.supported_architectures.mbart import (
            MBartArchitectureAdapter,
        )

        assert isinstance(mbart_bridge.adapter, MBartArchitectureAdapter)
        assert hasattr(mbart_bridge, "embed_ln")
        assert hasattr(mbart_bridge, "decoder_embed_ln")
        assert hasattr(mbart_bridge, "encoder_ln_final")
        assert hasattr(mbart_bridge, "decoder_ln_final")

    def test_scale_embedding_propagated(self, mbart_bridge):
        assert (
            mbart_bridge.cfg.scale_embedding == mbart_bridge.original_model.config.scale_embedding
        )


class TestMBartForwardEquivalence:
    def test_forward_matches_hf(self, mbart_bridge, sample_inputs):
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        hf_model = mbart_bridge.original_model
        with torch.no_grad():
            bridge_out = mbart_bridge(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
            )
            hf_out = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestMBartHooks:
    def test_pre_norm_and_stack_norm_hooks_fire(self, mbart_bridge, sample_inputs):
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        d_model = mbart_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "embed_ln.hook_out",
            "encoder_blocks.0.ln1.hook_out",
            "encoder_ln_final.hook_out",
            "decoder_embed_ln.hook_out",
            "decoder_blocks.0.cross_attn.hook_out",
            "decoder_ln_final.hook_out",
        ]
        with torch.no_grad():
            mbart_bridge.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                fwd_hooks=[(name, grab) for name in hooks],
            )
        enc_len = input_ids.shape[1]
        dec_len = decoder_input_ids.shape[1]
        assert captured["embed_ln.hook_out"] == (1, enc_len, d_model)
        assert captured["encoder_blocks.0.ln1.hook_out"] == (1, enc_len, d_model)
        assert captured["encoder_ln_final.hook_out"] == (1, enc_len, d_model)
        assert captured["decoder_embed_ln.hook_out"] == (1, dec_len, d_model)
        assert captured["decoder_blocks.0.cross_attn.hook_out"] == (1, dec_len, d_model)
        assert captured["decoder_ln_final.hook_out"] == (1, dec_len, d_model)
