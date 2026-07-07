"""Integration tests for the M2M100 / NLLB architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "stas/tiny-m2m_100"


@pytest.fixture(scope="module")
def m2m_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_inputs(m2m_bridge):
    tokens = m2m_bridge.tokenizer("Hello world, this is a test.", return_tensors="pt")
    start = m2m_bridge.original_model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[start, 5, 6, 7]])
    return tokens.input_ids, tokens.attention_mask, decoder_input_ids


class TestM2M100BridgeCreation:
    def test_adapter_and_block_counts(self, m2m_bridge):
        from transformer_lens.model_bridge.supported_architectures.m2m100 import (
            M2M100ArchitectureAdapter,
        )

        assert isinstance(m2m_bridge.adapter, M2M100ArchitectureAdapter)
        assert len(m2m_bridge.encoder_blocks) == len(m2m_bridge.decoder_blocks)

    def test_has_stack_final_norms(self, m2m_bridge):
        assert hasattr(m2m_bridge, "encoder_ln_final")
        assert hasattr(m2m_bridge, "decoder_ln_final")


class TestM2M100ForwardEquivalence:
    def test_forward_matches_hf(self, m2m_bridge, sample_inputs):
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        hf_model = m2m_bridge.original_model
        with torch.no_grad():
            bridge_out = m2m_bridge(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
            )
            hf_out = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestM2M100EmbedScale:
    def test_embed_hook_captures_scaled_output(self, m2m_bridge, sample_inputs):
        """The sqrt(d_model) scale lives inside M2M100ScaledWordEmbedding, so
        the embed hook output is already scaled — the opposite of Marian."""
        import math

        input_ids, attention_mask, decoder_input_ids = sample_inputs
        embed_module = m2m_bridge.original_model.model.encoder.embed_tokens
        assert math.isclose(embed_module.embed_scale, math.sqrt(m2m_bridge.cfg.d_model))

        captured = {}

        def grab(tensor, hook):
            captured["embed"] = tensor.detach().clone()

        with torch.no_grad():
            m2m_bridge.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                fwd_hooks=[("hook_embed", grab)],
            )
            scaled = embed_module(input_ids)
        assert torch.equal(captured["embed"], scaled)


class TestM2M100SinusoidalPositions:
    def test_w_pos_resolves_from_weights_buffer(self, m2m_bridge):
        """M2M100's sinusoidal module stores its table as a 'weights' buffer."""
        w_pos = m2m_bridge.pos_embed.W_pos
        assert w_pos.ndim == 2
        assert w_pos.shape[1] == m2m_bridge.cfg.d_model


class TestM2M100Hooks:
    def test_pre_norm_and_final_norm_hooks_fire(self, m2m_bridge, sample_inputs):
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        d_model = m2m_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "encoder_blocks.0.ln1.hook_out",
            "encoder_blocks.0.hook_out",
            "encoder_ln_final.hook_out",
            "decoder_blocks.0.cross_attn.hook_out",
            "decoder_ln_final.hook_out",
        ]
        with torch.no_grad():
            m2m_bridge.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                fwd_hooks=[(name, grab) for name in hooks],
            )
        enc_len = input_ids.shape[1]
        dec_len = decoder_input_ids.shape[1]
        assert captured["encoder_blocks.0.ln1.hook_out"] == (1, enc_len, d_model)
        assert captured["encoder_blocks.0.hook_out"] == (1, enc_len, d_model)
        assert captured["encoder_ln_final.hook_out"] == (1, enc_len, d_model)
        assert captured["decoder_blocks.0.cross_attn.hook_out"] == (1, dec_len, d_model)
        assert captured["decoder_ln_final.hook_out"] == (1, dec_len, d_model)
