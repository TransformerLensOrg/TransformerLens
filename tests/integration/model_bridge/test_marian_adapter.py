"""Integration tests for the Marian (opus-mt) architecture adapter."""

import math

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "Helsinki-NLP/opus-mt-en-de"


@pytest.fixture(scope="module")
def marian_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_inputs(marian_bridge):
    tokens = marian_bridge.tokenizer("Hello world, this is a test.", return_tensors="pt")
    start = marian_bridge.original_model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[start, 100, 200, 300]])
    return tokens.input_ids, tokens.attention_mask, decoder_input_ids


class TestMarianBridgeCreation:
    def test_block_counts(self, marian_bridge):
        assert len(marian_bridge.encoder_blocks) == 6
        assert len(marian_bridge.decoder_blocks) == 6

    def test_has_core_components(self, marian_bridge):
        assert hasattr(marian_bridge, "embed")
        assert hasattr(marian_bridge, "pos_embed")
        assert hasattr(marian_bridge, "decoder_embed")
        assert hasattr(marian_bridge, "decoder_pos_embed")
        assert hasattr(marian_bridge, "unembed")

    def test_cfg_flags(self, marian_bridge):
        assert marian_bridge.cfg.normalization_type == "LN"
        assert marian_bridge.cfg.positional_embedding_type == "standard"
        assert marian_bridge.cfg.gated_mlp is False

    def test_scale_embedding_propagated_from_hf_config(self, marian_bridge):
        hf_config = marian_bridge.original_model.config
        assert marian_bridge.cfg.scale_embedding == hf_config.scale_embedding


class TestMarianForwardEquivalence:
    def test_forward_matches_hf(self, marian_bridge, sample_inputs):
        """Bridge delegates to HF native forward — output should be identical."""
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        hf_model = marian_bridge.original_model
        with torch.no_grad():
            bridge_out = marian_bridge(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
            )
            hf_out = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_forward_includes_final_logits_bias(self, marian_bridge, sample_inputs):
        """HF adds a trained final_logits_bias after lm_head; the bridge output must include it."""
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        hf_model = marian_bridge.original_model
        bias = hf_model.final_logits_bias
        assert bias.abs().max().item() > 0, "opus-mt checkpoints carry a trained bias"
        with torch.no_grad():
            decoder_hidden = hf_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).last_hidden_state
            unbiased = hf_model.lm_head(decoder_hidden)
            bridge_out = marian_bridge(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
            )
        max_diff = (bridge_out - (unbiased + bias)).abs().max().item()
        assert max_diff < 1e-5, f"Bridge output missing final_logits_bias, diff = {max_diff}"


class TestMarianHFDelegation:
    def test_attention_projections_are_live_hf_modules(self, marian_bridge):
        """The bridge components and original_model paths must hold the same wrapper objects."""
        hf_model = marian_bridge.original_model
        assert (
            marian_bridge.encoder_blocks[0].attn.q
            is hf_model.model.encoder.layers[0].self_attn.q_proj
        )
        assert (
            marian_bridge.decoder_blocks[0].self_attn.q
            is hf_model.model.decoder.layers[0].self_attn.q_proj
        )
        assert (
            marian_bridge.decoder_blocks[0].cross_attn.o
            is hf_model.model.decoder.layers[0].encoder_attn.out_proj
        )

    def test_mlp_projections_are_live_hf_modules(self, marian_bridge):
        hf_model = marian_bridge.original_model
        assert (
            marian_bridge.encoder_blocks[0].mlp.submodules["in"]
            is hf_model.model.encoder.layers[0].fc1
        )


class TestMarianHookShapes:
    def test_encoder_decoder_hooks_fire_with_expected_shapes(self, marian_bridge, sample_inputs):
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        d_model = marian_bridge.cfg.d_model
        acts = {}

        def grab(tensor, hook):
            acts[hook.name] = tuple(tensor.shape)

        hooks = [
            "hook_embed",
            "encoder_blocks.0.hook_out",
            "encoder_blocks.5.hook_out",
            "decoder_blocks.0.hook_out",
            "decoder_blocks.0.cross_attn.hook_out",
        ]
        with torch.no_grad():
            marian_bridge.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                fwd_hooks=[(name, grab) for name in hooks],
            )
        enc_len = input_ids.shape[1]
        dec_len = decoder_input_ids.shape[1]
        assert acts["embed.hook_out"] == (1, enc_len, d_model)
        assert acts["encoder_blocks.0.hook_out"] == (1, enc_len, d_model)
        assert acts["encoder_blocks.5.hook_out"] == (1, enc_len, d_model)
        assert acts["decoder_blocks.0.hook_out"] == (1, dec_len, d_model)
        assert acts["decoder_blocks.0.cross_attn.hook_out"] == (1, dec_len, d_model)


class TestMarianEmbedScale:
    def test_hook_embed_captures_unscaled_embeddings(self, marian_bridge, sample_inputs):
        """HF multiplies by sqrt(d_model) after embed_tokens; the hook sees the module output."""
        input_ids, attention_mask, decoder_input_ids = sample_inputs
        hf_model = marian_bridge.original_model
        captured = {}

        def grab(tensor, hook):
            captured["embed"] = tensor.detach().clone()

        with torch.no_grad():
            marian_bridge.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                fwd_hooks=[("hook_embed", grab)],
            )
            raw = hf_model.model.encoder.embed_tokens(input_ids)
        assert torch.equal(captured["embed"], raw)
        assert math.isclose(
            hf_model.model.encoder.embed_scale,
            math.sqrt(marian_bridge.cfg.d_model),
            rel_tol=1e-9,
        )


class TestMarianSinusoidalPositions:
    def test_position_embeddings_are_deterministic_and_shared(self, marian_bridge):
        """Sinusoidal tables are computed, not learned — encoder and decoder must match."""
        enc_pos = marian_bridge.original_model.model.encoder.embed_positions.weight
        dec_pos = marian_bridge.original_model.model.decoder.embed_positions.weight
        assert torch.equal(enc_pos, dec_pos)
        # First row of a sinusoidal table is sin(0)=0 for the first half.
        half = enc_pos.shape[1] // 2
        assert torch.allclose(enc_pos[0, :half], torch.zeros(half))


class TestMarianGeneration:
    def test_greedy_translation_produces_text(self, marian_bridge):
        out = marian_bridge.generate(
            "The cat sat on the mat.", max_new_tokens=20, do_sample=False, verbose=False
        )
        assert isinstance(out, str)
        assert len(out.strip()) > 0
