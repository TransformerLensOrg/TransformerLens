"""Tiny from_config Whisper integration tests — CI coverage without a real checkpoint.

The openai/whisper-tiny parity suite (test_whisper_adapter.py) is ``slow``-marked
since the checkpoint is not in the CI cache. This file builds a random-weight
WhisperForConditionalGeneration with asymmetric stacks (3 encoder / 2 decoder
layers, as on large-v3-turbo and distil-whisper) and exercises the same surfaces:
live wiring, teacher-forced logit and loss parity (including several clips sharing one
prompt), the decoder prompt from to_tokens, encoder and decoder hooks,
the channel-first conv hooks, and an encoder-side intervention.
"""
import copy

import pytest
import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.whisper import (
    WhisperArchitectureAdapter,
)

ARCH = "WhisperForConditionalGeneration"
N_MELS = 16
N_FRAMES = 20  # max_source_positions; the mel input is twice this (conv2 has stride 2)
D_MODEL = 32
ENC_LAYERS = 3
DEC_LAYERS = 2
DECODER_START = 3


class _Tok:
    pass


@pytest.fixture(scope="module")
def hf_model():
    torch.manual_seed(0)
    cfg = WhisperConfig(
        vocab_size=128,
        num_mel_bins=N_MELS,
        d_model=D_MODEL,
        encoder_layers=ENC_LAYERS,
        decoder_layers=DEC_LAYERS,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=N_FRAMES,
        max_target_positions=24,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=DECODER_START,
    )
    cfg.architectures = [ARCH]
    model = AutoModelForSpeechSeq2Seq.from_config(cfg, attn_implementation="eager")
    return model.to(torch.float32).eval()


@pytest.fixture(scope="module")
def reference_model(hf_model):
    """Unwrapped copy: the bridge swaps its components into hf_model in place."""
    return copy.deepcopy(hf_model)


@pytest.fixture(scope="module")
def bridge(hf_model, reference_model):
    bridge_cfg = build_bridge_config_from_hf(hf_model.config, ARCH, "whisper-tiny", torch.float32)
    return TransformerBridge(hf_model, WhisperArchitectureAdapter(bridge_cfg), tokenizer=_Tok())


@pytest.fixture(scope="module")
def features():
    torch.manual_seed(1)
    return torch.randn(2, N_MELS, 2 * N_FRAMES)


@pytest.fixture(scope="module")
def decoder_ids():
    return torch.tensor([[DECODER_START, 5, 6, 7], [DECODER_START, 9, 10, 11]])


@pytest.fixture(scope="module")
def reference_logits(reference_model, features, decoder_ids):
    with torch.no_grad():
        return reference_model(input_features=features, decoder_input_ids=decoder_ids).logits


@pytest.fixture(scope="module")
def cache(bridge, features, decoder_ids):
    with torch.no_grad():
        _, c = bridge.run_with_cache(features, decoder_input=decoder_ids)
    return c


class TestWhisperTinyStructure:
    def test_asymmetric_stacks(self, bridge):
        assert len(bridge.encoder_blocks) == ENC_LAYERS
        assert len(bridge.decoder_blocks) == DEC_LAYERS
        assert bridge.cfg.n_layers == DEC_LAYERS

    def test_n_ctx_is_decoder_context(self, bridge, hf_model):
        assert bridge.cfg.n_ctx == hf_model.config.max_target_positions

    def test_front_end_wraps_live_modules(self, bridge, hf_model):
        """The bridge components sit in HF's module tree, so HF's forward fires them."""
        encoder = hf_model.model.encoder
        assert encoder.conv1 is bridge.conv1
        assert encoder.conv2 is bridge.conv2
        assert encoder.embed_positions is bridge.pos_embed
        assert hf_model.proj_out is bridge.unembed
        assert isinstance(bridge.conv1.original_component, torch.nn.Conv1d)
        assert isinstance(bridge.pos_embed.original_component, torch.nn.Embedding)

    def test_unembed_tied_to_decoder_embed(self, bridge):
        unembed_w = bridge.unembed.original_component.weight
        embed_w = bridge.decoder_embed.original_component.weight
        assert unembed_w.data_ptr() == embed_w.data_ptr()

    def test_k_proj_has_no_bias(self, bridge):
        """Whisper builds k_proj with bias=False; q/v/out keep theirs."""
        attns = [b.attn for b in bridge.encoder_blocks]
        attns += [b.self_attn for b in bridge.decoder_blocks]
        attns += [b.cross_attn for b in bridge.decoder_blocks]
        for attn in attns:
            assert attn.k.original_component.bias is None
            for proj in (attn.q, attn.v, attn.o):
                assert proj.original_component.bias is not None


class TestWhisperTinyForwardEquivalence:
    def test_logits_match_hf(self, bridge, features, decoder_ids, reference_logits):
        with torch.no_grad():
            logits = bridge(features, decoder_input=decoder_ids)
        assert logits.shape == (2, decoder_ids.shape[1], 128)
        max_diff = (logits - reference_logits).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_run_with_cache_logits_match_hf(self, bridge, features, decoder_ids, reference_logits):
        with torch.no_grad():
            logits, _ = bridge.run_with_cache(features, decoder_input=decoder_ids)
        assert (logits - reference_logits).abs().max().item() < 1e-5

    def test_one_dimensional_decoder_input_is_batched(self, bridge, features, decoder_ids):
        with torch.no_grad():
            flat = bridge(features[:1], decoder_input=decoder_ids[0])
            batched = bridge(features[:1], decoder_input=decoder_ids[:1])
        assert torch.equal(flat, batched)

    @pytest.mark.parametrize("prompt_shape", ["1d", "single_row"])
    def test_shared_prompt_gives_each_clip_its_own_logits(
        self, bridge, reference_model, features, decoder_ids, prompt_shape
    ):
        prompt = decoder_ids[0] if prompt_shape == "1d" else decoder_ids[:1]
        with torch.no_grad():
            logits = bridge(features, decoder_input=prompt)
            per_clip = torch.cat(
                [
                    reference_model(
                        input_features=features[i : i + 1], decoder_input_ids=decoder_ids[:1]
                    ).logits
                    for i in range(features.shape[0])
                ]
            )
        assert logits.shape == per_clip.shape
        assert (logits - per_clip).abs().max().item() < 1e-5
        assert not torch.allclose(logits[0], logits[1])

    def test_mismatched_decoder_batch_rejected(self, bridge, features, decoder_ids):
        three_rows = torch.cat([decoder_ids, decoder_ids[:1]])
        with pytest.raises(ValueError, match="batch size 3 but the audio features have 2"):
            bridge(features, decoder_input=three_rows)

    def test_shared_labels_match_per_clip_logits_and_loss(self, bridge, reference_model, features):
        labels = torch.tensor([[5, 6, 7, 2]])
        with torch.no_grad():
            expected = reference_model(input_features=features, labels=labels.repeat(2, 1))
            logits = bridge(features, labels=labels)
            loss = bridge(features, labels=labels, return_type="loss")
        assert logits.shape == expected.logits.shape
        assert (logits - expected.logits).abs().max().item() < 1e-5
        assert torch.allclose(loss, expected.loss, atol=1e-6)

    def test_labels_loss_matches_hf(self, bridge, reference_model, features):
        labels = torch.tensor([[5, 6, 7, 2], [9, 10, 11, 2]])
        with torch.no_grad():
            hf_loss = reference_model(input_features=features, labels=labels).loss
            loss = bridge(features, labels=labels, return_type="loss")
        assert torch.allclose(loss, hf_loss, atol=1e-6)


class TestWhisperTinyHooks:
    def test_encoder_front_end_shapes(self, cache):
        assert cache["conv1.hook_in"].shape == (2, N_MELS, 2 * N_FRAMES)
        assert cache["conv1.hook_out"].shape == (2, D_MODEL, 2 * N_FRAMES)
        assert cache["conv2.hook_out"].shape == (2, D_MODEL, N_FRAMES)
        assert cache["pos_embed.hook_out"].shape == (N_FRAMES, D_MODEL)

    def test_both_stacks_fire(self, cache, decoder_ids):
        dec_len = decoder_ids.shape[1]
        assert cache[f"encoder_blocks.{ENC_LAYERS - 1}.hook_out"].shape == (2, N_FRAMES, D_MODEL)
        assert cache["encoder_ln_final.hook_out"].shape == (2, N_FRAMES, D_MODEL)
        assert cache["decoder_embed.hook_out"].shape == (2, dec_len, D_MODEL)
        assert cache[f"decoder_blocks.{DEC_LAYERS - 1}.hook_out"].shape == (2, dec_len, D_MODEL)
        assert cache["decoder_blocks.0.cross_attn.hook_pattern"].shape == (2, 4, dec_len, N_FRAMES)
        assert cache["unembed.hook_out"].shape == (2, dec_len, 128)

    def test_front_end_matches_hf_math(self, cache):
        """hook_out on the convs is pre-GELU; the first block sees conv2 + positions."""
        conv2_act = torch.nn.functional.gelu(cache["conv2.hook_out"]).permute(0, 2, 1)
        expected = conv2_act + cache["pos_embed.hook_out"]
        assert torch.allclose(cache["encoder_blocks.0.hook_in"], expected, atol=1e-6)

    def test_pos_slice_uses_declared_position_axes(self, bridge, features, decoder_ids):
        with torch.no_grad():
            _, sliced = bridge.run_with_cache(features, decoder_input=decoder_ids, pos_slice=3)
        assert sliced["conv1.hook_out"].shape == (2, D_MODEL, 1)
        assert sliced["conv2.hook_out"].shape == (2, D_MODEL, 1)
        assert sliced["pos_embed.hook_out"].shape == (1, D_MODEL)
        assert sliced["encoder_blocks.0.hook_out"].shape == (2, 1, D_MODEL)

    def test_encoder_patch_matches_hf_with_patched_encoder_outputs(
        self, bridge, reference_model, features, decoder_ids, cache
    ):
        """Overwriting the encoder output through a hook equals handing HF that output."""
        patched = cache["encoder_ln_final.hook_out"].clone()
        patched[:, 5:10] = 0.0

        def patch(tensor, hook):
            return patched

        with torch.no_grad():
            logits = bridge.run_with_hooks(
                features,
                decoder_input=decoder_ids,
                fwd_hooks=[("encoder_ln_final.hook_out", patch)],
            )
        bridge.reset_hooks()
        with torch.no_grad():
            expected = reference_model(
                encoder_outputs=(patched,), decoder_input_ids=decoder_ids
            ).logits
            unpatched = reference_model(input_features=features, decoder_input_ids=decoder_ids)
        assert (logits - expected).abs().max().item() < 1e-5
        assert (logits - unpatched.logits).abs().max().item() > 1e-3


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|en|>",
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
]


@pytest.fixture
def prompt_bridge(tmp_path):
    """Tiny Whisper with a real byte-level WhisperTokenizer, built without a download.

    Saved and reloaded so it carries the from_pretrained metadata the boot path relies on.
    """
    from transformers import WhisperTokenizer
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    from transformer_lens.model_bridge.sources._bridge_builder import (
        configure_tokenizer,
    )

    vocab = {ch: i for i, ch in enumerate(bytes_to_unicode().values())}
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)
    WhisperTokenizer(
        vocab=vocab, merges=[], additional_special_tokens=SPECIAL_TOKENS[1:]
    ).save_pretrained(tmp_path)
    tokenizer = WhisperTokenizer.from_pretrained(tmp_path)
    eot = vocab["<|endoftext|>"]
    cfg = WhisperConfig(
        vocab_size=len(vocab),
        num_mel_bins=N_MELS,
        d_model=D_MODEL,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=N_FRAMES,
        max_target_positions=24,
        pad_token_id=eot,
        bos_token_id=eot,
        eos_token_id=eot,
        decoder_start_token_id=vocab["<|startoftranscript|>"],
    )
    cfg.architectures = [ARCH]
    model = AutoModelForSpeechSeq2Seq.from_config(cfg, attn_implementation="eager").eval()
    bridge_cfg = build_bridge_config_from_hf(model.config, ARCH, "whisper-tok", torch.float32)
    adapter = WhisperArchitectureAdapter(bridge_cfg)
    return TransformerBridge(model, adapter, tokenizer=configure_tokenizer(tokenizer, adapter.cfg))


class TestWhisperTinyDecoderPrompt:
    def test_to_tokens_is_prefix_then_text(self, prompt_bridge):
        """No <|endoftext|> lands inside the prompt, before or after the text."""
        tokenizer = prompt_bridge.tokenizer
        tokens = prompt_bridge.to_tokens(" Hello")[0].tolist()
        text_ids = tokenizer(" Hello", add_special_tokens=False).input_ids
        assert tokens == tokenizer.prefix_tokens + text_ids
        assert tokenizer.eos_token_id not in tokens

    def test_language_and_task_prefix(self, prompt_bridge):
        tokenizer = prompt_bridge.tokenizer
        tokenizer.set_prefix_tokens(language="english", task="transcribe")
        tokens = prompt_bridge.to_tokens(" Hello")[0].tolist()
        prefix = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
        assert tokens[:4] == tokenizer.convert_tokens_to_ids(prefix)
        assert tokenizer.eos_token_id not in tokens

    def test_prompt_drives_a_multi_clip_forward(self, prompt_bridge):
        prompt = prompt_bridge.to_tokens(" Hello")
        with torch.no_grad():
            logits = prompt_bridge(torch.randn(3, N_MELS, 2 * N_FRAMES), decoder_input=prompt)
        assert logits.shape == (3, prompt.shape[1], prompt_bridge.cfg.d_vocab)


class TestWhisperTinyErrors:
    def test_decoder_input_required(self, bridge, features):
        with pytest.raises(ValueError, match="decoder_input"):
            bridge(features)

    def test_text_input_rejected(self, bridge, decoder_ids):
        with pytest.raises(ValueError, match="input_features"):
            bridge("hello", decoder_input=decoder_ids)

    def test_input_values_rejected(self, bridge, features, decoder_ids):
        with pytest.raises(ValueError, match="input_values"):
            bridge(features, input_values=features, decoder_input=decoder_ids)

    def test_generate_not_supported(self, bridge, features):
        with pytest.raises(NotImplementedError):
            bridge.generate(features, max_new_tokens=2, verbose=False)

    def test_compatibility_mode_not_supported(self, bridge):
        with pytest.raises(NotImplementedError, match="audio seq2seq"):
            bridge.enable_compatibility_mode()


class TestWhisperTinyPhase1Benchmarks:
    """The verify_models Phase 1 checks, run on the tiny model."""

    def test_forward_pass_benchmark(self, bridge, reference_model):
        from transformer_lens.benchmarks.forward_pass import benchmark_forward_pass
        from transformer_lens.benchmarks.utils import build_modality_input

        features = build_modality_input(bridge)
        assert features is not None
        assert features.shape == (1, N_MELS, 2 * N_FRAMES)
        result = benchmark_forward_pass(bridge, features, reference_model=reference_model)
        assert result.passed, result.message

    def test_component_benchmark(self, bridge, reference_model):
        from transformer_lens.benchmarks.component_benchmark import (
            benchmark_all_components,
        )

        result = benchmark_all_components(bridge, reference_model)
        assert result.passed, result.message
