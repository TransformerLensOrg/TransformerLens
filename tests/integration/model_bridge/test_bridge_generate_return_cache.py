"""Tests for TransformerBridge.generate(return_cache=True).

return_cache makes generate() also return an ActivationCache for the full prompt +
generated sequence, identical to run_with_cache(output) (issue #697). v1 supports
single-sequence, decoder-only text generation; other paths raise NotImplementedError.

Uses distilgpt2 (CI-cached).
"""

import warnings

import pytest
import torch


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Alias the session fixture for concise test signatures."""
    return distilgpt2_bridge


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
                tokens, max_new_tokens=5, do_sample=False, return_type="tokens", return_cache=True
            )
        assert isinstance(cache, ActivationCache)
        assert out.shape[1] > tokens.shape[1], "Should generate additional tokens"

    def test_cache_matches_run_with_cache_over_full_sequence(self, bridge):
        """The returned cache equals run_with_cache(output) and spans prompt + generated tokens."""
        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            out, gen_cache = bridge.generate(
                tokens, max_new_tokens=5, do_sample=False, return_type="tokens", return_cache=True
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
        """The cache includes full [batch, heads, q, k] attention patterns (an Option B benefit)."""
        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            out, cache = bridge.generate(
                tokens, max_new_tokens=4, do_sample=False, return_type="tokens", return_cache=True
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
            out = bridge.generate(tokens, max_new_tokens=5, do_sample=False, return_type="tokens")
        assert isinstance(out, torch.Tensor)

    def test_with_output_logits(self, bridge):
        """output_logits=True + return_cache=True -> (ModelOutput, cache); cache matches the sequences."""
        tokens = bridge.to_tokens("The quick brown")
        with torch.no_grad():
            out, cache = bridge.generate(
                tokens, max_new_tokens=4, do_sample=False, output_logits=True, return_cache=True
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
            )
            _, ref = bridge.run_with_cache(out, names_filter=wanted)
        assert wanted in cache
        # Same scoped key set as run_with_cache (passthrough), and far smaller than the full cache.
        assert set(cache.cache_dict) == set(ref.cache_dict)
        assert len(cache.cache_dict) < 20

    def test_device_offload_no_spurious_warning(self, bridge):
        """device= offloads cache tensors (cpu here) without ActivationCache.to's move_model warning."""
        tokens = bridge.to_tokens("The quick brown")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                _, cache = bridge.generate(
                    tokens,
                    max_new_tokens=4,
                    do_sample=False,
                    return_type="tokens",
                    return_cache=True,
                    device="cpu",
                )
        assert str(cache["blocks.0.hook_resid_post"].device) == "cpu"
        assert not any("move_model" in str(w.message) for w in caught), [
            str(w.message) for w in caught
        ]


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
