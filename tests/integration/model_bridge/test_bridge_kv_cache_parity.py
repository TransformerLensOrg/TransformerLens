"""Integration tests for HookedTransformer KV cache parity."""

import numpy as np
import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


@pytest.fixture(scope="module")
def model_name():
    """Use a small model for fast testing."""
    return "distilgpt2"


@pytest.fixture(scope="module")
def hooked_transformer(model_name):
    """Create HookedTransformer for comparison."""
    model = HookedTransformer.from_pretrained(
        model_name, move_to_device=True, default_prepend_bos=False
    )
    model.eval()
    return model


@pytest.fixture
def test_prompt():
    """Standard test prompt for cache parity tests."""
    return "Hello world, this is a quick kv-cache test."


class TestHookedTransformerCacheParity:
    """Test KV cache parity for HookedTransformer."""

    def test_hooked_transformer_cache_parity(self, hooked_transformer, test_prompt):
        """Test that HookedTransformer produces identical results with and without cache."""
        model = hooked_transformer
        tokens = model.to_tokens(test_prompt, prepend_bos=False)

        # Full forward (no cache)
        with torch.inference_mode():
            logits_full = model(tokens, return_type="logits")
            last_logits_full = logits_full[:, -1]

        # Cached forward: split by tokens
        pre_tokens = tokens[:, :-1]
        next_tokens = tokens[:, -1:]

        past_kv_cache = TransformerLensKeyValueCache.init_cache(
            model.cfg, model.cfg.device, pre_tokens.shape[0]
        )
        with torch.inference_mode():
            # Prime the cache
            _ = model(pre_tokens, return_type="logits", past_kv_cache=past_kv_cache)
            # Run only the new token
            logits_cached = model(next_tokens, return_type="logits", past_kv_cache=past_kv_cache)
            last_logits_cached = logits_cached[:, -1]

        # Compare with appropriate tolerance for HookedTransformer
        max_diff = (last_logits_full - last_logits_cached).abs().max().item()
        assert (
            max_diff < 1e-4
        ), f"KV cache parity failed for HookedTransformer, max_diff: {max_diff}"
