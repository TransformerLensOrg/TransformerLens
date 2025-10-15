"""Integration tests for TransformerBridge KV cache parity with HookedTransformer."""

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache
from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.supported_architectures.gpt2 import (
    GPT2ArchitectureAdapter,
)

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


@pytest.fixture(scope="module")
def transformer_bridge(model_name, hooked_transformer):
    """Create TransformerBridge for testing."""
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_model.eval()

    # Create TransformerBridgeConfig manually from HookedTransformer config
    # Copy relevant fields from HookedTransformerConfig to TransformerBridgeConfig
    bridge_cfg = TransformerBridgeConfig(
        d_model=hooked_transformer.cfg.d_model,
        d_head=hooked_transformer.cfg.d_head,
        n_heads=hooked_transformer.cfg.n_heads,
        d_mlp=hooked_transformer.cfg.d_mlp,
        n_layers=hooked_transformer.cfg.n_layers,
        n_ctx=hooked_transformer.cfg.n_ctx,
        d_vocab=hooked_transformer.cfg.d_vocab,
        architecture="gpt2",  # Set architecture for adapter selection
        device=hooked_transformer.cfg.device,
    )
    adapter = GPT2ArchitectureAdapter(bridge_cfg)
    bridge = TransformerBridge(hf_model, adapter, hf_tokenizer)

    return bridge


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


@pytest.mark.skip(reason="KV cache support for TransformerBridge is currently incomplete")
class TestTransformerBridgeCacheParity:
    """Test KV cache parity for TransformerBridge."""

    def test_bridge_cache_parity(self, transformer_bridge, test_prompt):
        """Test that TransformerBridge produces consistent results with and without cache."""
        bridge = transformer_bridge
        tokens = bridge.to_tokens(test_prompt, prepend_bos=False)

        # Full forward via bridge (no cache) - ensure consistent attention mask handling
        with torch.inference_mode():
            # Create attention mask for consistency
            attention_mask = torch.ones_like(tokens, dtype=torch.long)
            logits_full = bridge.forward(
                tokens, return_type="logits", attention_mask=attention_mask
            )
            last_logits_full = logits_full[:, -1]

        # Cached forward via bridge: use TransformerLensKeyValueCache like HookedTransformer
        pre_tokens = tokens[:, :-1]
        next_tokens = tokens[:, -1:]

        # Initialize TransformerLens cache using bridge's config
        past_kv_cache = TransformerLensKeyValueCache.init_cache(
            bridge.cfg, bridge.cfg.device, pre_tokens.shape[0]
        )
        with torch.inference_mode():
            # Create attention masks for consistency
            pre_attention_mask = torch.ones_like(pre_tokens, dtype=torch.long)
            next_attention_mask = torch.ones_like(next_tokens, dtype=torch.long)

            # Prime the cache with prefix tokens
            _ = bridge.forward(
                pre_tokens,
                return_type="logits",
                past_kv_cache=past_kv_cache,
                attention_mask=pre_attention_mask,
            )
            # Run only the new token with the cache
            logits_cached = bridge.forward(
                next_tokens,
                return_type="logits",
                past_kv_cache=past_kv_cache,
                attention_mask=next_attention_mask,
            )
            last_logits_cached = logits_cached[:, -1]

        # Compare with appropriate tolerance for TransformerBridge
        # (allows for numerical precision differences in cache conversion)
        max_diff = (last_logits_full - last_logits_cached).abs().max().item()
        assert (
            max_diff < 2e-4
        ), f"KV cache parity failed for TransformerBridge, max_diff: {max_diff}"

    def test_bridge_cache_multiple_tokens(self, transformer_bridge):
        """Test cache parity with multiple new tokens."""
        bridge = transformer_bridge
        pre_prompt = "The weather today is"
        post_prompt = " sunny and warm outside"

        pre_tokens = bridge.to_tokens(pre_prompt, prepend_bos=False)
        post_tokens = bridge.to_tokens(post_prompt, prepend_bos=False)
        full_tokens = torch.cat([pre_tokens, post_tokens], dim=1)

        # Full forward
        with torch.inference_mode():
            attention_mask = torch.ones_like(full_tokens, dtype=torch.long)
            logits_full = bridge.forward(
                full_tokens, return_type="logits", attention_mask=attention_mask
            )

        # Cached forward
        past_kv_cache = TransformerLensKeyValueCache.init_cache(
            bridge.cfg, bridge.cfg.device, pre_tokens.shape[0]
        )
        with torch.inference_mode():
            pre_attention_mask = torch.ones_like(pre_tokens, dtype=torch.long)
            post_attention_mask = torch.ones_like(post_tokens, dtype=torch.long)

            # Prime cache
            _ = bridge.forward(
                pre_tokens,
                return_type="logits",
                past_kv_cache=past_kv_cache,
                attention_mask=pre_attention_mask,
            )
            # Generate with cache
            logits_cached = bridge.forward(
                post_tokens,
                return_type="logits",
                past_kv_cache=past_kv_cache,
                attention_mask=post_attention_mask,
            )

        # Compare the overlapping logits (last tokens of full vs cached)
        post_token_count = post_tokens.shape[1]
        full_post_logits = logits_full[:, -post_token_count:]

        max_diff = (full_post_logits - logits_cached).abs().max().item()
        assert max_diff < 2e-4, f"Multi-token cache parity failed, max_diff: {max_diff}"

    def test_bridge_cache_consistency(self, transformer_bridge, test_prompt):
        """Test that cache state is properly maintained across calls."""
        bridge = transformer_bridge
        tokens = bridge.to_tokens(test_prompt, prepend_bos=False)

        # Split into three parts
        part1 = tokens[:, :3]
        part2 = tokens[:, 3:6]
        part3 = tokens[:, 6:]

        # Initialize cache
        past_kv_cache = TransformerLensKeyValueCache.init_cache(
            bridge.cfg, bridge.cfg.device, part1.shape[0]
        )

        with torch.inference_mode():
            # Process parts sequentially with cache
            attention_mask1 = torch.ones_like(part1, dtype=torch.long)
            attention_mask2 = torch.ones_like(part2, dtype=torch.long)
            attention_mask3 = torch.ones_like(part3, dtype=torch.long)

            _ = bridge.forward(
                part1,
                return_type="logits",
                past_kv_cache=past_kv_cache,
                attention_mask=attention_mask1,
            )
            _ = bridge.forward(
                part2,
                return_type="logits",
                past_kv_cache=past_kv_cache,
                attention_mask=attention_mask2,
            )
            logits_cached = bridge.forward(
                part3,
                return_type="logits",
                past_kv_cache=past_kv_cache,
                attention_mask=attention_mask3,
            )

            # Compare with full forward
            full_attention_mask = torch.ones_like(tokens, dtype=torch.long)
            logits_full = bridge.forward(
                tokens, return_type="logits", attention_mask=full_attention_mask
            )

        # Compare final logits
        max_diff = (logits_full[:, -1] - logits_cached[:, -1]).abs().max().item()
        assert max_diff < 2e-4, f"Cache consistency failed, max_diff: {max_diff}"


class TestCacheFormatConversion:
    """Test cache format conversion between TransformerLens and HuggingFace formats."""

    @pytest.mark.skip(reason="KV cache format conversion failing due to architectural differences")
    def test_cache_format_conversion(self, transformer_bridge, test_prompt):
        """Test that cache format conversion preserves information correctly."""
        bridge = transformer_bridge
        tokens = bridge.to_tokens(test_prompt, prepend_bos=False)
        pre_tokens = tokens[:, :-1]

        # Initialize cache and populate it
        past_kv_cache = TransformerLensKeyValueCache.init_cache(
            bridge.cfg, bridge.cfg.device, pre_tokens.shape[0]
        )

        with torch.inference_mode():
            attention_mask = torch.ones_like(pre_tokens, dtype=torch.long)
            _ = bridge.forward(
                pre_tokens,
                return_type="logits",
                past_kv_cache=past_kv_cache,
                attention_mask=attention_mask,
            )

        # Check that cache entries have been populated
        for i, entry in enumerate(past_kv_cache.entries):
            assert entry.past_keys.shape[1] == pre_tokens.shape[1], f"Layer {i} keys shape mismatch"
            assert (
                entry.past_values.shape[1] == pre_tokens.shape[1]
            ), f"Layer {i} values shape mismatch"
            assert (
                entry.past_keys.shape[0] == pre_tokens.shape[0]
            ), f"Layer {i} keys batch size mismatch"
            assert (
                entry.past_values.shape[0] == pre_tokens.shape[0]
            ), f"Layer {i} values batch size mismatch"

        # Check attention mask is properly maintained
        assert past_kv_cache.previous_attention_mask.shape[1] == pre_tokens.shape[1]
        assert torch.all(
            past_kv_cache.previous_attention_mask == 1
        ), "Attention mask should be all ones"


@pytest.mark.slow
@pytest.mark.skip(reason="KV cache support for TransformerBridge is currently incomplete")
class TestLargerModelParity:
    """Test cache parity with larger models (marked as slow)."""

    @pytest.mark.parametrize("model_name", ["gpt2"])
    def test_larger_model_parity(self, model_name):
        """Test parity with a larger model."""
        # Create models
        hooked_transformer = HookedTransformer.from_pretrained(
            model_name, move_to_device=True, default_prepend_bos=False
        )
        hooked_transformer.eval()

        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        hf_model.eval()

        # Create TransformerBridgeConfig manually from HookedTransformer config
        bridge_cfg = TransformerBridgeConfig(
            d_model=hooked_transformer.cfg.d_model,
            d_head=hooked_transformer.cfg.d_head,
            n_heads=hooked_transformer.cfg.n_heads,
            d_mlp=hooked_transformer.cfg.d_mlp,
            n_layers=hooked_transformer.cfg.n_layers,
            n_ctx=hooked_transformer.cfg.n_ctx,
            d_vocab=hooked_transformer.cfg.d_vocab,
            architecture="gpt2",
            device=hooked_transformer.cfg.device,
        )
        adapter = GPT2ArchitectureAdapter(bridge_cfg)
        bridge = TransformerBridge(hf_model, adapter, hf_tokenizer)

        test_prompt = "The quick brown fox jumps over the lazy dog."

        # Test HookedTransformer parity
        tokens = hooked_transformer.to_tokens(test_prompt, prepend_bos=False)
        pre_tokens = tokens[:, :-1]
        next_tokens = tokens[:, -1:]

        with torch.inference_mode():
            logits_full = hooked_transformer(tokens, return_type="logits")
            last_logits_full = logits_full[:, -1]

            past_kv_cache = TransformerLensKeyValueCache.init_cache(
                hooked_transformer.cfg, hooked_transformer.cfg.device, pre_tokens.shape[0]
            )
            _ = hooked_transformer(pre_tokens, return_type="logits", past_kv_cache=past_kv_cache)
            logits_cached = hooked_transformer(
                next_tokens, return_type="logits", past_kv_cache=past_kv_cache
            )
            last_logits_cached = logits_cached[:, -1]

        max_diff_ht = (last_logits_full - last_logits_cached).abs().max().item()
        assert max_diff_ht < 1e-4, f"HookedTransformer parity failed, max_diff: {max_diff_ht}"

        # Test TransformerBridge parity
        bridge_tokens = bridge.to_tokens(test_prompt, prepend_bos=False)
        bridge_pre_tokens = bridge_tokens[:, :-1]
        bridge_next_tokens = bridge_tokens[:, -1:]

        with torch.inference_mode():
            attention_mask = torch.ones_like(bridge_tokens, dtype=torch.long)
            bridge_logits_full = bridge.forward(
                bridge_tokens, return_type="logits", attention_mask=attention_mask
            )
            bridge_last_logits_full = bridge_logits_full[:, -1]

            bridge_past_kv_cache = TransformerLensKeyValueCache.init_cache(
                bridge.cfg, bridge.cfg.device, bridge_pre_tokens.shape[0]
            )
            pre_attention_mask = torch.ones_like(bridge_pre_tokens, dtype=torch.long)
            next_attention_mask = torch.ones_like(bridge_next_tokens, dtype=torch.long)

            _ = bridge.forward(
                bridge_pre_tokens,
                return_type="logits",
                past_kv_cache=bridge_past_kv_cache,
                attention_mask=pre_attention_mask,
            )
            bridge_logits_cached = bridge.forward(
                bridge_next_tokens,
                return_type="logits",
                past_kv_cache=bridge_past_kv_cache,
                attention_mask=next_attention_mask,
            )
            bridge_last_logits_cached = bridge_logits_cached[:, -1]

        max_diff_bridge = (bridge_last_logits_full - bridge_last_logits_cached).abs().max().item()
        assert (
            max_diff_bridge < 3e-4
        ), f"TransformerBridge parity failed, max_diff: {max_diff_bridge}"
