"""Manual KV-cache injection into TransformerBridge.forward (HF-native).

forward(past_key_values=...) hands HF back its own cache object and computes
only the new tokens' keys/values. The correctness anchor is that incremental
decoding (token-by-token or chunked) matches a single full forward, and that
use_cache is enabled automatically when a cache is passed.
"""

import pytest
import torch


@pytest.fixture()
def bridge(distilgpt2_bridge):
    return distilgpt2_bridge


def _seq():
    return torch.randint(0, 100, (1, 7))


def test_token_by_token_matches_full_forward(bridge):
    seq = _seq()
    with torch.no_grad():
        ref = bridge.forward(seq)

    cache = None
    step_logits = []
    with torch.no_grad():
        for i in range(seq.shape[1]):
            logits, cache = bridge.forward(
                seq[:, i : i + 1], past_key_values=cache, return_type="logits_and_cache"
            )
            step_logits.append(logits[:, -1, :])
    incremental = torch.stack(step_logits, dim=1)

    assert torch.allclose(ref, incremental, atol=1e-3)


def test_chunked_prefill_matches_full_forward(bridge):
    seq = _seq()
    with torch.no_grad():
        ref = bridge.forward(seq)
        _, cache = bridge.forward(seq[:, :4], return_type="logits_and_cache")
        tail_logits, _ = bridge.forward(
            seq[:, 4:], past_key_values=cache, return_type="logits_and_cache"
        )
    assert torch.allclose(ref[:, -1], tail_logits[:, -1], atol=1e-3)


def test_use_cache_enabled_automatically(bridge):
    """Passing a cache alone (no explicit use_cache) must return a grown cache."""
    seq = _seq()
    with torch.no_grad():
        _, cache = bridge.forward(seq[:, :3], return_type="logits_and_cache")
        assert cache is not None
        _, cache2 = bridge.forward(
            seq[:, 3:5], past_key_values=cache, return_type="logits_and_cache"
        )
    assert cache2 is not None


def test_normal_forward_unaffected(bridge):
    seq = _seq()
    with torch.no_grad():
        out = bridge.forward(seq)
    assert out.shape == (1, seq.shape[1], bridge.cfg.d_vocab)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
