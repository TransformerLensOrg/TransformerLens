"""Inference-time ops HookedTransformer must not silently drop.

HT re-implements each architecture's forward from converted weights, so an op HF
applies but TL's config never represents diverges silently — no error, just wrong
numbers. Each test here pins one such op found by sweeping the HF modeling
sources against TL's flags.
"""

from types import SimpleNamespace
from unittest import mock

import torch

from transformer_lens.loading_from_pretrained import convert_hf_model_config


def _llama_shaped_config(**extra):
    return SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        num_hidden_layers=2,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        vocab_size=32000,
        hidden_act="silu",
        **extra,
    )


class TestGenericLlamaRotaryBase:
    """Llama-arch checkpoints with no name-matched branch (Yi ships rope_theta
    5e6) fell through to the config default 10000 — every rotary angle wrong."""

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_rope_theta_is_read(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = _llama_shaped_config(rope_theta=5000000.0)
        assert convert_hf_model_config("01-ai/Yi-6B")["rotary_base"] == 5000000.0

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_rope_theta_from_transformers_5_dict(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = _llama_shaped_config(
            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0}
        )
        assert convert_hf_model_config("01-ai/Yi-6B")["rotary_base"] == 5000000.0

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_llama3_scaling_is_threaded(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = _llama_shaped_config(
            rope_theta=500000.0,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        cfg = convert_hf_model_config("01-ai/Yi-34B")
        assert cfg["use_NTK_by_parts_rope"] is True
        assert cfg["NTK_by_parts_factor"] == 8.0
        assert cfg["NTK_original_ctx_len"] == 8192

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_default_theta_when_config_is_silent(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = _llama_shaped_config()
        assert convert_hf_model_config("01-ai/Yi-6B-Chat")["rotary_base"] == 10000.0


class TestGemma2SlidingWindowPhase:
    """HF makes layer 0 sliding; TL alternated the other way, windowing exactly
    the layers HF leaves global past 4096 tokens."""

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_even_layers_are_local(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = SimpleNamespace(
            architectures=["Gemma2ForCausalLM"]
        )
        for name, n_layers in [
            ("google/gemma-2-2b", 26),
            ("google/gemma-2-9b", 42),
            ("google/gemma-2-27b", 46),
        ]:
            cfg = convert_hf_model_config(name)
            types = cfg["attn_types"]
            assert len(types) == n_layers == cfg["n_layers"], name
            assert types[0] == "local", name
            assert types[1] == "global", name
            # Windowing only binds because n_ctx outruns the window.
            assert cfg["n_ctx"] > cfg["window_size"], name

    def test_hf_derivation_still_puts_sliding_on_layer_zero(self) -> None:
        """Pin the upstream rule this fix mirrors, so a phase flip in
        transformers is caught here rather than by drifting logits."""
        from transformers.models.gemma2.configuration_gemma2 import Gemma2Config

        derived = Gemma2Config(num_hidden_layers=4).layer_types
        assert derived[:2] == ["sliding_attention", "full_attention"], derived


class TestApertusQKNorm:
    """HF applies q_norm/k_norm unconditionally; TL gated them on a config field
    that does not exist, so they were never built or converted."""

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_qk_norm_is_on(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = SimpleNamespace(
            architectures=["ApertusForCausalLM"],
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=256,
            num_hidden_layers=2,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            hidden_act="xielu",
            rope_theta=12000000.0,
        )
        assert convert_hf_model_config("swiss-ai/Apertus-8B-2509")["use_qk_norm"] is True

    def test_hf_config_still_has_no_gate(self) -> None:
        """The fix hardcodes True because HF has nothing to read; if a gate ever
        appears, this fails and the branch should read it instead."""
        from transformers.models.apertus.configuration_apertus import ApertusConfig

        assert not hasattr(ApertusConfig(), "qk_norm")


class TestPhi3SlidingWindow:
    """Phi-3-mini-4k ships sliding_window=2047 inside a 4096 n_ctx; TL never read
    the field, leaving attention full-causal past 2047 tokens."""

    @staticmethod
    def _phi3_config(**extra):
        return SimpleNamespace(
            architectures=["Phi3ForCausalLM"],
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=256,
            num_hidden_layers=3,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            vocab_size=32064,
            hidden_act="silu",
            initializer_range=0.02,
            rope_theta=10000.0,
            **extra,
        )

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_window_is_populated(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = self._phi3_config(sliding_window=2047)
        cfg = convert_hf_model_config("microsoft/Phi-3-mini-4k-instruct")
        assert cfg["use_local_attn"] is True
        assert cfg["window_size"] == 2047
        assert cfg["attn_types"] == ["local"] * 3

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_no_window_stays_global(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = self._phi3_config(sliding_window=None)
        cfg = convert_hf_model_config("microsoft/Phi-3-mini-4k-instruct")
        assert cfg["use_local_attn"] is False
        assert cfg["attn_types"] is None


class TestMixtralAndGPT2ConfigReads:
    """Two fields the branches hardcoded instead of reading."""

    @staticmethod
    def _mixtral_config(**extra):
        return SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            num_hidden_layers=3,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            hidden_act="silu",
            rope_theta=1000000.0,
            num_local_experts=8,
            num_experts_per_tok=2,
            **extra,
        )

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_mixtral_window_follows_the_config(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = self._mixtral_config(sliding_window=4096)
        cfg = convert_hf_model_config("mistralai/Mixtral-8x7B-Instruct-v0.1")
        assert cfg["use_local_attn"] is True
        assert cfg["attn_types"] == ["local"] * 3

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_mixtral_attn_types_length_tracks_n_layers(
        self, mock_auto_config: mock.MagicMock
    ) -> None:
        """The old ["global"] * 32 was inert only because use_local_attn was
        hardcoded False; reading the config makes its length load-bearing."""
        mock_auto_config.from_pretrained.return_value = self._mixtral_config(sliding_window=None)
        cfg = convert_hf_model_config("mistralai/Mixtral-8x7B-v0.1")
        assert cfg["use_local_attn"] is False
        assert len(cfg["attn_types"]) == cfg["n_layers"] == 3

    @mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
    def test_gpt2_reads_scale_attn_weights(self, mock_auto_config: mock.MagicMock) -> None:
        mock_auto_config.from_pretrained.return_value = SimpleNamespace(
            architectures=["GPT2LMHeadModel"],
            n_embd=128,
            n_head=4,
            n_layer=2,
            n_ctx=1024,
            layer_norm_epsilon=1e-5,
            vocab_size=50257,
            activation_function="gelu_new",
            scale_attn_by_inverse_layer_idx=False,
            scale_attn_weights=False,
        )
        assert convert_hf_model_config("gpt2")["use_attn_scale"] is False


class TestT5DecoderRelativeBias:
    """HF buckets decoder self-attention with bidirectional=False; TL hardcoded
    True, so every decoder forward used the encoder's scheme."""

    @staticmethod
    def _cfg():
        from transformer_lens import HookedTransformerConfig

        return HookedTransformerConfig.from_dict(
            {
                "d_model": 32,
                "d_head": 8,
                "n_heads": 4,
                "d_mlp": 64,
                "n_layers": 2,
                "n_ctx": 16,
                "d_vocab": 100,
                "act_fn": "relu",
                "normalization_type": "RMS",
                "positional_embedding_type": "relative_positional_bias",
                "relative_attention_num_buckets": 32,
                "relative_attention_max_distance": 128,
                "use_attn_scale": False,
                # What the real T5 branch sets — one cfg shared by both stacks.
                "attention_dir": "bidirectional",
            }
        )

    def _tl_buckets(self, is_decoder: bool, seq: int = 8) -> torch.Tensor:
        from transformer_lens.components.t5_attention import T5Attention

        attn = T5Attention(self._cfg(), has_relative_attention_bias=True, is_decoder=is_decoder)
        context = torch.arange(seq)[:, None]
        memory = torch.arange(seq)[None, :]
        return T5Attention._relative_position_bucket(
            memory - context,
            bidirectional=not attn.is_decoder,
            num_buckets=32,
            max_distance=128,
        )

    def _hf_buckets(self, is_decoder: bool, seq: int = 8) -> torch.Tensor:
        from transformers.models.t5.modeling_t5 import T5Attention as HFT5Attention

        context = torch.arange(seq)[:, None]
        memory = torch.arange(seq)[None, :]
        return HFT5Attention._relative_position_bucket(
            memory - context,
            bidirectional=not is_decoder,
            num_buckets=32,
            max_distance=128,
        )

    def test_decoder_buckets_match_hf(self) -> None:
        torch.testing.assert_close(self._tl_buckets(True), self._hf_buckets(True))

    def test_encoder_buckets_match_hf(self) -> None:
        torch.testing.assert_close(self._tl_buckets(False), self._hf_buckets(False))

    def test_the_two_schemes_actually_differ(self) -> None:
        """Engagement check: if decoder and encoder bucketing coincided, the two
        tests above would pass no matter which flag the component used."""
        assert not torch.equal(self._tl_buckets(True), self._tl_buckets(False))

    def test_decoder_blocks_build_decoder_attention(self) -> None:
        from transformer_lens.components.t5_block import T5Block

        assert T5Block(self._cfg(), 0, is_decoder=True).attn.is_decoder is True
        assert T5Block(self._cfg(), 0, is_decoder=False).attn.is_decoder is False

    def _pattern(self, is_decoder: bool, seq: int = 6) -> torch.Tensor:
        from transformer_lens.components.t5_block import T5Block

        attn = T5Block(self._cfg(), 0, is_decoder=is_decoder).attn
        with torch.random.fork_rng():
            torch.manual_seed(0)
            for p in attn.parameters():
                torch.nn.init.normal_(p, std=0.2)
            resid = torch.randn(1, seq, 32)
        seen: dict = {}
        attn.hook_pattern.add_hook(lambda t, hook: seen.__setitem__("p", t))
        bias = attn.compute_relative_attention_bias(seq, seq)
        with torch.no_grad():
            attn(resid, resid, resid, position_bias=bias)
        return seen["p"]

    def test_decoder_self_attention_masks_the_future(self) -> None:
        """T5's cfg says attention_dir="bidirectional" for the encoder, and
        HookedEncoderDecoder never built a causal mask — so decoder self-attention
        read future tokens on every multi-token forward."""
        pattern = self._pattern(is_decoder=True)
        seq = pattern.shape[-1]
        future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        assert pattern[..., future].abs().max() == 0.0

    def test_encoder_self_attention_stays_bidirectional(self) -> None:
        pattern = self._pattern(is_decoder=False)
        seq = pattern.shape[-1]
        future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        assert pattern[..., future].abs().max() > 0.0

    def test_cross_attention_is_not_masked(self) -> None:
        """Only decoder SELF-attention is causal; cross-attention sees the whole
        encoder sequence."""
        from transformer_lens.components.t5_block import T5Block

        assert T5Block(self._cfg(), 0, is_decoder=True).cross_attn._attention_dir_override is None

    def test_bias_values_differ_between_stacks(self) -> None:
        """End to end through the component that actually feeds attn_scores."""
        from transformer_lens.components.t5_block import T5Block

        with torch.random.fork_rng():
            torch.manual_seed(0)
            enc = T5Block(self._cfg(), 0, is_decoder=False).attn
            dec = T5Block(self._cfg(), 0, is_decoder=True).attn
            dec.rel_pos_bias.weight.data.copy_(enc.rel_pos_bias.weight.data)
        assert not torch.equal(
            enc.compute_relative_attention_bias(8, 8),
            dec.compute_relative_attention_bias(8, 8),
        )
