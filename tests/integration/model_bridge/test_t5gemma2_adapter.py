"""Integration tests for the T5Gemma2 architecture adapter (text-only).

Uses a tiny randomly-initialized T5Gemma2ForConditionalGeneration built from
config (no HF Hub download), following test_deepseek_adapter.py. The full
786M-param google/t5gemma-2-270m-270m is exercised by registry verification
(verify_models), not here.
"""

import tempfile

import pytest
import torch
from transformers import AutoTokenizer, T5Gemma2Config, T5Gemma2ForConditionalGeneration
from transformers.models.t5gemma2 import T5Gemma2DecoderConfig, T5Gemma2EncoderConfig

from transformer_lens.model_bridge.bridge import TransformerBridge

N_LAYERS = 2
N_HEADS = 4
D_MODEL = 64
D_HEAD = 16
D_VOCAB = 1000


@pytest.fixture(scope="module")
def tiny_t5gemma2_bridge():
    text_cfg = dict(
        vocab_size=D_VOCAB,
        hidden_size=D_MODEL,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=2,
        head_dim=D_HEAD,
        max_position_embeddings=128,
        sliding_window=16,
    )
    vision_cfg = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    )
    config = T5Gemma2Config(
        encoder=T5Gemma2EncoderConfig(text_config=text_cfg, vision_config=vision_cfg),
        decoder=T5Gemma2DecoderConfig(**dict(text_cfg, cross_attention_hidden_size=D_MODEL)),
    )
    torch.manual_seed(0)
    hf_model = T5Gemma2ForConditionalGeneration(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_model.save_pretrained(tmpdir)
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.save_pretrained(tmpdir)
        bridge = TransformerBridge.boot_transformers(tmpdir, device="cpu")
    return bridge


@pytest.fixture(scope="module")
def tokens():
    return torch.tensor([[1, 5, 9, 13]])


class TestT5Gemma2BridgeCreation:
    def test_block_counts(self, tiny_t5gemma2_bridge):
        assert len(tiny_t5gemma2_bridge.encoder_blocks) == N_LAYERS
        assert len(tiny_t5gemma2_bridge.decoder_blocks) == N_LAYERS

    def test_has_core_components(self, tiny_t5gemma2_bridge):
        for name in (
            "encoder_embed",
            "encoder_rotary_emb",
            "encoder_ln_final",
            "decoder_embed",
            "decoder_rotary_emb",
            "decoder_ln_final",
            "unembed",
        ):
            assert hasattr(tiny_t5gemma2_bridge, name), f"missing {name}"

    def test_config_flags(self, tiny_t5gemma2_bridge):
        cfg = tiny_t5gemma2_bridge.cfg
        assert cfg.act_fn == "gelu_pytorch_tanh"
        assert cfg.rmsnorm_uses_offset is True
        assert cfg.gated_mlp is True
        assert cfg.positional_embedding_type == "rotary"


class TestT5Gemma2ForwardEquivalence:
    def test_forward_returns_logits(self, tiny_t5gemma2_bridge, tokens):
        with torch.no_grad():
            logits = tiny_t5gemma2_bridge(tokens)
        assert logits.shape == (1, tokens.shape[1], D_VOCAB)
        assert not torch.isnan(logits).any()

    def test_forward_matches_hf(self, tiny_t5gemma2_bridge, tokens):
        """Bridge delegates to HF native forward — output should be identical."""
        hf_model = tiny_t5gemma2_bridge.original_model
        # Mirror the bridge's decoder_input_ids auto-generation (bridge.py):
        # with no decoder_start_token_id, input_ids are passed through as-is.
        start = getattr(hf_model.config, "decoder_start_token_id", None)
        if start is None:
            decoder_input_ids = tokens
        else:
            start_col = torch.full((tokens.shape[0], 1), start, dtype=tokens.dtype)
            decoder_input_ids = torch.cat([start_col, tokens[:, :-1]], dim=1)
        with torch.no_grad():
            bridge_out = tiny_t5gemma2_bridge(tokens)
            hf_out = hf_model(input_ids=tokens, decoder_input_ids=decoder_input_ids).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"


class TestT5Gemma2HFDelegation:
    """Boot replaces HF submodules in-place with bridge wrappers; each wrapper
    holds the raw HF module as original_component sharing the same weight tensor."""

    def test_encoder_attn_projections_are_bridge_wrappers(self, tiny_t5gemma2_bridge):
        hf_model = tiny_t5gemma2_bridge.original_model
        hf_attn = hf_model.model.encoder.text_model.layers[0].self_attn
        bridge_attn = tiny_t5gemma2_bridge.encoder_blocks[0].attn
        assert hf_attn.q_proj is bridge_attn.q
        assert hf_attn.o_proj is bridge_attn.o
        assert bridge_attn.q.original_component.weight is hf_attn.q_proj.weight

    def test_decoder_merged_attn_projections_are_bridge_wrappers(self, tiny_t5gemma2_bridge):
        hf_model = tiny_t5gemma2_bridge.original_model
        hf_attn = hf_model.model.decoder.layers[0].self_attn
        bridge_attn = tiny_t5gemma2_bridge.decoder_blocks[0].self_attn
        assert hf_attn.q_proj is bridge_attn.q
        assert hf_attn.k_proj is bridge_attn.k
        assert bridge_attn.v.original_component.weight is hf_attn.v_proj.weight

    def test_unembed_is_lm_head_out_proj(self, tiny_t5gemma2_bridge):
        hf_model = tiny_t5gemma2_bridge.original_model
        assert hf_model.lm_head.out_proj is tiny_t5gemma2_bridge.unembed


class TestT5Gemma2HookShapes:
    @pytest.fixture(scope="class")
    def cache(self, tiny_t5gemma2_bridge, tokens):
        _, cache = tiny_t5gemma2_bridge.run_with_cache(tokens)
        return cache

    def test_encoder_residual_hooks(self, cache, tokens):
        seq = tokens.shape[1]
        for i in range(N_LAYERS):
            assert cache[f"encoder_blocks.{i}.hook_resid_pre"].shape == (1, seq, D_MODEL)
            assert cache[f"encoder_blocks.{i}.hook_resid_post"].shape == (1, seq, D_MODEL)

    def test_decoder_residual_hooks(self, cache, tokens):
        seq = tokens.shape[1]
        for i in range(N_LAYERS):
            assert cache[f"decoder_blocks.{i}.hook_resid_pre"].shape == (1, seq, D_MODEL)
            assert cache[f"decoder_blocks.{i}.hook_resid_mid"].shape == (1, seq, D_MODEL)
            assert cache[f"decoder_blocks.{i}.hook_resid_post"].shape == (1, seq, D_MODEL)

    def test_no_decoder_resid_mid2(self, cache):
        """Merged attention collapses self+cross into one sub-layer — no second mid hook."""
        assert not any("hook_resid_mid2" in k for k in cache.keys())

    def test_encoder_attention_pattern(self, cache, tokens):
        seq = tokens.shape[1]
        assert cache["encoder_blocks.0.attn.hook_pattern"].shape == (1, N_HEADS, seq, seq)

    def test_mlp_hooks_fire(self, cache, tokens):
        seq = tokens.shape[1]
        for prefix in ("encoder_blocks.0", "decoder_blocks.0"):
            assert cache[f"{prefix}.mlp.hook_pre"].shape[:2] == (1, seq)
            assert cache[f"{prefix}.mlp.hook_post"].shape[:2] == (1, seq)


class TestT5Gemma2MergedAttention:
    """The decoder uses a single merged self+cross attention with shared q/k/v/o;
    the bridge exposes both the self pattern and the cross pattern as hooks."""

    @pytest.fixture(scope="class")
    def cache(self, tiny_t5gemma2_bridge, tokens):
        _, cache = tiny_t5gemma2_bridge.run_with_cache(tokens)
        return cache

    def test_no_separate_cross_attn_module(self, tiny_t5gemma2_bridge):
        for block in tiny_t5gemma2_bridge.decoder_blocks:
            assert not hasattr(block, "cross_attn")

    def test_self_and_cross_pattern_hooks_fire(self, cache, tokens):
        dec_seq = enc_seq = tokens.shape[1]
        for i in range(N_LAYERS):
            self_pattern = cache[f"decoder_blocks.{i}.self_attn.hook_pattern"]
            cross_pattern = cache[f"decoder_blocks.{i}.self_attn.hook_cross_pattern"]
            assert self_pattern.shape == (1, N_HEADS, dec_seq, dec_seq)
            assert cross_pattern.shape == (1, N_HEADS, dec_seq, enc_seq)
            assert not torch.isnan(self_pattern).any()
            assert not torch.isnan(cross_pattern).any()

    def test_patterns_are_slices_of_one_merged_softmax(self, cache):
        """Self and cross weights come from a single softmax over concatenated
        decoder+encoder key positions, so each query row's probability mass is
        split between the two hooks and only their sum is normalized."""
        for i in range(N_LAYERS):
            self_pattern = cache[f"decoder_blocks.{i}.self_attn.hook_pattern"]
            cross_pattern = cache[f"decoder_blocks.{i}.self_attn.hook_cross_pattern"]
            joint_row_sums = self_pattern.sum(dim=-1) + cross_pattern.sum(dim=-1)
            assert torch.allclose(joint_row_sums, torch.ones_like(joint_row_sums), atol=1e-5)
            assert self_pattern.sum(dim=-1).max() < 1.0


class TestT5Gemma2QKNorm:
    def test_qk_norm_hooks_fire_per_head(self, tiny_t5gemma2_bridge, tokens):
        _, cache = tiny_t5gemma2_bridge.run_with_cache(tokens)
        seq = tokens.shape[1]
        q_norm_out = cache["encoder_blocks.0.attn.q_norm.hook_out"]
        assert q_norm_out.shape == (1, N_HEADS, seq, D_HEAD)
        assert "decoder_blocks.0.self_attn.k_norm.hook_out" in cache
