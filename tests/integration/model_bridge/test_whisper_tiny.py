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

    def test_shared_decoder_mask_matches_per_clip_hf(self, bridge, reference_model, decoder_ids):
        torch.manual_seed(3)
        clips = torch.randn(3, N_MELS, 2 * N_FRAMES)
        prompt = decoder_ids[:1]
        mask = torch.tensor([[1, 1, 0, 1]])
        with torch.no_grad():
            logits = bridge(clips, decoder_input=prompt, decoder_attention_mask=mask)
            unmasked = bridge(clips, decoder_input=prompt)
            per_clip = torch.cat(
                [
                    reference_model(
                        input_features=clips[i : i + 1],
                        decoder_input_ids=prompt,
                        decoder_attention_mask=mask,
                    ).logits
                    for i in range(clips.shape[0])
                ]
            )
        assert logits.shape == per_clip.shape
        assert (logits - per_clip).abs().max().item() < 1e-5
        assert not torch.allclose(logits, unmasked)

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


SUPPRESS_TOKENS = [10, 11, 12]


@pytest.fixture
def prompt_pair(tmp_path):
    """Tiny Whisper with a real byte-level WhisperTokenizer, built without a download, plus
    an unwrapped HF copy. The tokenizer is saved and reloaded so it carries the
    from_pretrained metadata the boot path relies on; the generation config sets
    suppress_tokens and begin_suppress_tokens, as Whisper checkpoints do.
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
    torch.manual_seed(0)
    model = AutoModelForSpeechSeq2Seq.from_config(cfg, attn_implementation="eager").eval()
    model.generation_config.suppress_tokens = SUPPRESS_TOKENS
    model.generation_config.begin_suppress_tokens = [eot]
    reference = copy.deepcopy(model)
    bridge_cfg = build_bridge_config_from_hf(model.config, ARCH, "whisper-tok", torch.float32)
    adapter = WhisperArchitectureAdapter(bridge_cfg)
    bridge = TransformerBridge(
        model, adapter, tokenizer=configure_tokenizer(tokenizer, adapter.cfg)
    )
    return bridge, reference


@pytest.fixture
def prompt_bridge(prompt_pair):
    return prompt_pair[0]


@pytest.fixture
def clips():
    torch.manual_seed(2)
    return torch.randn(2, N_MELS, 2 * N_FRAMES)


def _hf_generate(reference, clips, prompt, max_new_tokens):
    """HF's own generation loop (GenerationMixin, which applies the generation config's
    suppress processors) on the unwrapped copy."""
    from transformers import GenerationMixin

    with torch.no_grad():
        out = GenerationMixin.generate(
            reference,
            input_features=clips,
            decoder_input_ids=prompt.repeat(clips.shape[0], 1),
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    return out if isinstance(out, torch.Tensor) else out.sequences


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


class TestWhisperTinyGenerate:
    def test_greedy_matches_hf_generate(self, prompt_pair, clips):
        """Default prompt is the tokenizer prefix; output keeps it, as HF's does."""
        bridge, reference = prompt_pair
        prompt = torch.tensor([bridge.tokenizer.prefix_tokens])
        tokens = bridge.generate(clips, do_sample=False, max_new_tokens=8, verbose=False)
        assert torch.equal(tokens, _hf_generate(reference, clips, prompt, 8))
        assert torch.equal(tokens[:, : prompt.shape[1]], prompt.repeat(2, 1))

    def test_explicit_decoder_input_matches_hf_generate(self, prompt_pair, clips):
        bridge, reference = prompt_pair
        prompt = bridge.to_tokens(" a")
        tokens = bridge.generate(
            clips, decoder_input=prompt, do_sample=False, max_new_tokens=6, verbose=False
        )
        assert torch.equal(tokens, _hf_generate(reference, clips, prompt, 6))

    def test_suppressed_tokens_are_never_generated(self, prompt_pair, clips):
        bridge, reference = prompt_pair
        generation_config = bridge.original_model.generation_config
        prompt_len = len(bridge.tokenizer.prefix_tokens)
        generation_config.suppress_tokens = None
        generation_config.begin_suppress_tokens = None
        free = bridge.generate(clips, do_sample=False, max_new_tokens=4, verbose=False)
        # Begin-suppress the free run's first token: a misplaced begin step leaves step one as is.
        first = int(free[0, prompt_len])
        later = next(int(t) for t in free[0, prompt_len:] if int(t) != first)
        for config in (generation_config, reference.generation_config):
            config.suppress_tokens = [later]
            config.begin_suppress_tokens = [first]
        tokens = bridge.generate(clips, do_sample=False, max_new_tokens=4, verbose=False)
        assert int(tokens[0, prompt_len]) != first
        assert later not in tokens[:, prompt_len:].tolist()[0]
        prompt = torch.tensor([bridge.tokenizer.prefix_tokens])
        assert torch.equal(tokens, _hf_generate(reference, clips, prompt, 4))

    def test_return_type_str_gives_transcripts(self, prompt_bridge, clips):
        tokens = prompt_bridge.generate(clips, do_sample=False, max_new_tokens=5, verbose=False)
        texts = prompt_bridge.generate(
            clips, do_sample=False, max_new_tokens=5, verbose=False, return_type="str"
        )
        expected = [prompt_bridge.tokenizer.decode(t, skip_special_tokens=True) for t in tokens]
        assert texts == expected
        single = prompt_bridge.generate(
            clips[:1], do_sample=False, max_new_tokens=5, verbose=False, return_type="str"
        )
        assert single == expected[0]

    def test_encoder_runs_once_decoder_every_step(self, prompt_bridge, clips):
        calls = {"encoder": 0, "pattern": 0, "decoder": 0}

        def count(name):
            def hook(tensor, hook):
                calls[name] += 1
                return tensor

            return hook

        fwd_hooks = [
            ("encoder_blocks.0.hook_out", count("encoder")),
            ("encoder_blocks.0.attn.hook_pattern", count("pattern")),
            ("decoder_blocks.0.hook_out", count("decoder")),
        ]
        with prompt_bridge.hooks(fwd_hooks=fwd_hooks):
            tokens = prompt_bridge.generate(
                clips, do_sample=False, max_new_tokens=5, stop_at_eos=False, verbose=False
            )
        prompt_len = len(prompt_bridge.tokenizer.prefix_tokens)
        assert calls["encoder"] == 1
        assert calls["pattern"] == 1
        assert calls["decoder"] == tokens.shape[1] - prompt_len

    def test_text_input_rejected(self, prompt_bridge):
        with pytest.raises(ValueError, match="audio features"):
            prompt_bridge.generate("hello", verbose=False)

    def test_too_many_new_tokens_rejected(self, prompt_bridge, clips):
        with pytest.raises(ValueError, match="positions"):
            prompt_bridge.generate(clips, max_new_tokens=prompt_bridge.cfg.n_ctx, verbose=False)

    def test_generate_stream_not_supported(self, prompt_bridge, clips):
        with pytest.raises(NotImplementedError, match="generate_stream"):
            next(prompt_bridge.generate_stream(clips, verbose=False))


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


def _compat_pair():
    """Fresh tiny Whisper with non-trivial LayerNorm affine and q/v biases (so folding
    has something to move), an unwrapped copy, and a bridge over the original."""
    torch.manual_seed(3)
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
    model = AutoModelForSpeechSeq2Seq.from_config(cfg, attn_implementation="eager").eval()
    model.generation_config.suppress_tokens = None
    model.generation_config.begin_suppress_tokens = None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "layer_norm" in name or name.endswith(("q_proj.bias", "v_proj.bias")):
                param.add_(0.3 * torch.randn_like(param))
    reference = copy.deepcopy(model)
    bridge_cfg = build_bridge_config_from_hf(model.config, ARCH, "whisper-compat", torch.float32)
    tokenizer = _Tok()
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    bridge = TransformerBridge(model, WhisperArchitectureAdapter(bridge_cfg), tokenizer=tokenizer)
    return bridge, reference


def _centered(logits):
    return logits - logits.mean(dim=-1, keepdim=True)


class TestWhisperTinyCompatibilityMode:
    @pytest.fixture(scope="class")
    def processed(self, features, decoder_ids):
        bridge, reference = _compat_pair()
        with torch.no_grad():
            expected = reference(input_features=features, decoder_input_ids=decoder_ids).logits
        bridge.enable_compatibility_mode(disable_warnings=True)
        return bridge, reference, expected

    def test_logits_match_hf_up_to_a_per_position_constant(self, processed, features, decoder_ids):
        """center_unembed shifts each position's logits by a constant; nothing else moves."""
        bridge, _, expected = processed
        with torch.no_grad():
            logits = bridge(features, decoder_input=decoder_ids)
        assert (_centered(logits) - _centered(expected)).abs().max().item() < 1e-5

    def test_norms_folded_into_their_readers(self, processed):
        bridge, _, _ = processed
        hf = bridge.original_model.model
        norms = [hf.encoder.layer_norm, hf.decoder.layer_norm]
        for layer in hf.encoder.layers:
            norms += [layer.self_attn_layer_norm, layer.final_layer_norm]
        for layer in hf.decoder.layers:
            norms += [
                layer.self_attn_layer_norm,
                layer.encoder_attn_layer_norm,
                layer.final_layer_norm,
            ]
        for norm in norms:
            assert torch.equal(norm.weight, torch.ones(D_MODEL))
            assert torch.equal(norm.bias, torch.zeros(D_MODEL))

    def test_value_biases_moved_and_writes_centered(self, processed):
        bridge, _, _ = processed
        hf = bridge.original_model.model
        attentions = [layer.self_attn for layer in hf.encoder.layers]
        attentions += [
            a for layer in hf.decoder.layers for a in (layer.self_attn, layer.encoder_attn)
        ]
        for attn in attentions:
            assert attn.v_proj.bias.abs().max() == 0
            assert attn.out_proj.weight.mean(dim=0).abs().max() < 1e-6
        for layer in [*hf.encoder.layers, *hf.decoder.layers]:
            assert layer.fc2.weight.mean(dim=0).abs().max() < 1e-6
        for table in (
            hf.encoder.embed_positions,
            hf.decoder.embed_tokens,
            hf.decoder.embed_positions,
        ):
            assert table.weight.mean(dim=1).abs().max() < 1e-6

    def test_key_hooks_match_the_unprocessed_model(self, processed, features, decoder_ids):
        """The folded norm bias lands in a created k bias rather than being dropped."""
        bridge, _, _ = processed
        unprocessed, _ = _compat_pair()
        unprocessed.enable_compatibility_mode(no_processing=True, disable_warnings=True)
        with torch.no_grad():
            _, cache = bridge.run_with_cache(features, decoder_input=decoder_ids)
            _, expected = unprocessed.run_with_cache(features, decoder_input=decoder_ids)
        names = [name for name in expected.keys() if name.endswith("hook_k")]
        assert len(names) == ENC_LAYERS + 2 * DEC_LAYERS
        for name in names:
            assert (cache[name] - expected[name]).abs().max().item() < 1e-5, name
        hf = bridge.original_model.model
        for layer in [*hf.encoder.layers, *hf.decoder.layers]:
            assert isinstance(layer.self_attn.k_proj.bias, torch.nn.Parameter)

    def test_unembed_untied_from_decoder_embed(self, processed):
        bridge, _, _ = processed
        unembed = bridge.unembed.original_component.weight
        embed = bridge.decoder_embed.original_component.weight
        assert unembed.data_ptr() != embed.data_ptr()
        assert unembed.mean(dim=0).abs().max() < 1e-6

    @pytest.mark.parametrize("fold_value_biases", [False, True])
    def test_unembed_untied_when_nothing_rewrites_it(self, fold_value_biases):
        """Without fold_ln or center_unembed, processing returns the tied tensor as is."""
        bridge, _ = _compat_pair()
        bridge.enable_compatibility_mode(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=fold_value_biases,
            disable_warnings=True,
        )
        unembed = bridge.unembed.original_component.weight
        embed = bridge.decoder_embed.original_component.weight
        before = embed.detach().clone()
        with torch.no_grad():
            unembed.add_(1.0)
        assert torch.equal(embed, before)

    def test_greedy_generate_unchanged(self, processed, features):
        bridge, reference, _ = processed
        prompt = torch.tensor([[DECODER_START]])
        tokens = bridge.generate(
            features, decoder_input=prompt, do_sample=False, max_new_tokens=5, verbose=False
        )
        assert torch.equal(tokens, _hf_generate(reference, features, prompt, 5))

    def test_no_processing_is_exact(self, features, decoder_ids):
        bridge, reference = _compat_pair()
        bridge.enable_compatibility_mode(no_processing=True, disable_warnings=True)
        with torch.no_grad():
            logits = bridge(features, decoder_input=decoder_ids)
            expected = reference(input_features=features, decoder_input_ids=decoder_ids).logits
        assert (logits - expected).abs().max().item() < 1e-5


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
