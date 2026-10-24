"""Tests for TransformerBridge.generate(return_cache=True).

distilgpt2 tests pass use_past_kv_cache=False: cached-eager attention can NaN on macOS-arm64
CI (issue #1322), and return_cache recomputes run_with_cache on the output regardless.
"""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias the session fixture for concise test signatures."""
    return distilgpt2_bridge


@pytest.fixture
def tiny_bridge():
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        model = GPT2LMHeadModel(config).eval()
    return build_bridge_from_module(model, architecture="GPT2LMHeadModel", hf_config=config)


# Representative position-indexed hooks present in the non-compat distilgpt2 bridge cache.
# All have the sequence axis at dim 1, so the cache length is shape[1].
_PROBE_HOOKS = [
    "hook_embed",
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_resid_post",
    "blocks.0.hook_mlp_out",
    "blocks.0.attn.hook_q",
    "blocks.0.attn.hook_z",
    "blocks.5.hook_resid_post",
    "ln_final.hook_normalized",
]
_PATTERN_HOOK = "blocks.0.attn.hook_pattern"  # [batch, heads, q, k]


class TestGenerateReturnCache:
    """generate(return_cache=True) returns a run_with_cache-equivalent cache over the output."""

    def test_returns_output_and_cache_tuple(self, bridge):
        """return_cache=True returns an (output, ActivationCache) tuple over the full sequence."""
        from transformer_lens.ActivationCache import ActivationCache

        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            out, cache = bridge.generate(
                tokens,
                max_new_tokens=5,
                do_sample=False,
                return_type="tokens",
                return_cache=True,
                use_past_kv_cache=False,
            )
        assert isinstance(cache, ActivationCache)
        assert out.shape[1] > tokens.shape[1], "Should generate additional tokens"

    def test_cache_matches_run_with_cache_over_full_sequence(self, bridge):
        """The returned cache equals run_with_cache(output) and spans prompt + generated tokens."""
        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            out, gen_cache = bridge.generate(
                tokens,
                max_new_tokens=5,
                do_sample=False,
                return_type="tokens",
                return_cache=True,
                use_past_kv_cache=False,
            )
            _, ref_cache = bridge.run_with_cache(out)

        for name in _PROBE_HOOKS:
            assert name in gen_cache, f"{name} missing from generate cache"
            g, r = gen_cache[name], ref_cache[name]
            assert g.shape == r.shape, f"{name}: {g.shape} vs {r.shape}"
            # The cache spans the full output (not just the prompt) - the point of #697.
            assert g.shape[1] == out.shape[1], f"{name} seq dim {g.shape[1]} != {out.shape[1]}"
            assert torch.allclose(g, r, atol=1e-5, rtol=1e-4), f"{name} differs from run_with_cache"

    def test_includes_attention_patterns(self, bridge):
        """The cache includes full [batch, heads, q, k] attention patterns."""
        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            out, cache = bridge.generate(
                tokens,
                max_new_tokens=4,
                do_sample=False,
                return_type="tokens",
                return_cache=True,
                use_past_kv_cache=False,
            )
        assert _PATTERN_HOOK in cache
        pattern = cache[_PATTERN_HOOK]
        seq_len = out.shape[1]
        assert pattern.shape[0] == 1 and tuple(pattern.shape[-2:]) == (
            seq_len,
            seq_len,
        ), pattern.shape

    def test_return_cache_false_is_unchanged(self, bridge):
        """Default return_cache=False returns only the output (no tuple), preserving back-compat."""
        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            out = bridge.generate(
                tokens,
                max_new_tokens=5,
                do_sample=False,
                return_type="tokens",
                use_past_kv_cache=False,
            )
        assert isinstance(out, torch.Tensor)

    def test_with_output_logits(self, bridge):
        """output_logits=True + return_cache=True -> (ModelOutput, cache); cache matches the sequences."""
        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            out, cache = bridge.generate(
                tokens,
                max_new_tokens=4,
                do_sample=False,
                output_logits=True,
                return_cache=True,
                use_past_kv_cache=False,
            )
            _, ref_cache = bridge.run_with_cache(out.sequences)
        assert hasattr(out, "sequences") and hasattr(out, "logits")
        name = "blocks.0.hook_resid_post"
        assert torch.allclose(cache[name], ref_cache[name], atol=1e-5, rtol=1e-4)

    def test_names_filter_scopes_cache(self, bridge):
        """names_filter is passed through to run_with_cache: same scoped key set, not the full cache."""
        tokens = bridge.to_tokens("The quick brown")
        wanted = "blocks.0.hook_resid_post"
        with torch.no_grad():
            out, cache = bridge.generate(
                tokens,
                max_new_tokens=4,
                do_sample=False,
                return_type="tokens",
                return_cache=True,
                names_filter=wanted,
                use_past_kv_cache=False,
            )
            _, ref = bridge.run_with_cache(out, names_filter=wanted)
        assert wanted in cache
        # Same scoped key set as run_with_cache (passthrough), and far smaller than the full cache.
        assert set(cache.cache_dict) == set(ref.cache_dict)
        assert len(cache.cache_dict) < 20

    def test_device_offload_lands_on_requested_device(self, bridge):
        """device= offloads cache tensors to the requested device."""
        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            _, cache = bridge.generate(
                tokens,
                max_new_tokens=4,
                do_sample=False,
                return_type="tokens",
                return_cache=True,
                device="cpu",
                use_past_kv_cache=False,
            )
        assert str(cache["blocks.0.hook_resid_post"].device) == "cpu"


class TestGenerateReturnCacheGuards:
    """return_cache raises a clear error on unsupported paths (v1 = single-sequence decoder-only text)."""

    def test_batched_raises(self, bridge):
        """Batched/multi-prompt return_cache raises NotImplementedError."""
        batched = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(NotImplementedError, match="batched"):
            bridge.generate(batched, max_new_tokens=3, do_sample=False, return_cache=True)

    def test_inputs_embeds_raises(self, bridge):
        """inputs_embeds (a float tensor) + return_cache raises NotImplementedError."""
        embeds = torch.randn(1, 4, bridge.cfg.d_model)
        with pytest.raises(NotImplementedError, match="inputs_embeds"):
            bridge.generate(embeds, max_new_tokens=3, do_sample=False, return_cache=True)


class TestGenerateReturnCacheAttentionMask:
    """The returned cache replays the prompt attention mask generation ran with."""

    @pytest.mark.parametrize("mask_values", [[0, 0, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], None])
    @pytest.mark.parametrize("mask_dtype", [torch.long, torch.bool])
    @pytest.mark.parametrize("use_past", [False, True])
    @pytest.mark.parametrize("early_eos", [False, True])
    def test_returned_cache_preserves_generation_context(
        self, tiny_bridge, mask_values, mask_dtype, use_past, early_eos
    ):
        bridge = tiny_bridge
        tokens = torch.tensor([[3, 4, 7, 8]])
        mask = None if mask_values is None else torch.tensor([mask_values], dtype=mask_dtype)
        original_mask = None if mask is None else mask.clone()
        with torch.no_grad():
            first_logits = bridge(tokens, attention_mask=mask)[:, -1]
            eos = first_logits.argmax(-1).item() if early_eos else None
            output, cache, input_tokens = bridge.generate(
                tokens,
                attention_mask=mask,
                max_new_tokens=3,
                do_sample=False,
                stop_at_eos=early_eos,
                eos_token_id=eos,
                use_past_kv_cache=use_past,
                output_logits=True,
                return_cache=True,
                return_input_tokens=True,
                verbose=False,
            )
            generated = output.sequences.shape[1] - tokens.shape[1]
            assert generated == (1 if early_eos else 3)
            extended_mask = (
                None if mask is None else torch.cat([mask, mask.new_ones((1, generated))], dim=1)
            )
            _, reference = bridge.run_with_cache(output.sequences, attention_mask=extended_mask)
        assert set(cache.cache_dict) == set(reference.cache_dict)
        for name in cache.cache_dict:
            torch.testing.assert_close(cache[name], reference[name])
        for step, logits in enumerate(output.logits):
            torch.testing.assert_close(
                cache["unembed.hook_out"][:, tokens.shape[1] - 1 + step],
                logits,
                atol=1e-6,
                rtol=1e-5,
            )
        torch.testing.assert_close(input_tokens, tokens)
        if mask is not None:
            torch.testing.assert_close(mask, original_mask)
            masked_keys = mask[0] == 0
            pattern = cache["blocks.0.attn.hook_pattern"]
            assert (
                torch.count_nonzero(
                    pattern[:, :, tokens.shape[1] - 1 :, : tokens.shape[1]][..., masked_keys]
                )
                == 0
            )

    def test_inferred_left_padding_mask_survives_filtered_cache(self, tiny_bridge):
        bridge = tiny_bridge
        vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"t{i}": i for i in range(3, 32)}}
        bridge.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
            pad_token="<pad>",
            eos_token="<eos>",
            unk_token="<unk>",
        )
        tokens = torch.tensor([[0, 0, 7, 8]])
        key = "blocks.0.hook_resid_post"
        with torch.no_grad():
            output, cache = bridge.generate(
                tokens,
                padding_side="left",
                max_new_tokens=2,
                do_sample=False,
                stop_at_eos=False,
                use_past_kv_cache=False,
                return_cache=True,
                names_filter=key,
                device="cpu",
                verbose=False,
            )
            _, reference = bridge.run_with_cache(
                output,
                attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]),
                names_filter=key,
            )
        assert set(cache.cache_dict) == set(reference.cache_dict)
        torch.testing.assert_close(cache[key], reference[key])
        assert cache[key].device.type == "cpu"
