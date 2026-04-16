"""Test comprehensive comparison between Bridge and HookedTransformer."""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge


class TestBridgeVsHookedComparison:
    """Comprehensive tests comparing Bridge and HookedTransformer behavior."""

    @pytest.fixture(scope="class")
    def models_with_processing(self):
        """Create both models with weight processing."""
        # HookedTransformer with processing (using distilgpt2 for faster tests)
        hooked = HookedTransformer.from_pretrained(
            "distilgpt2",
            device="cpu",
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
        )

        # Bridge with equivalent processing
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
        bridge.enable_compatibility_mode()

        return hooked, bridge

    @pytest.fixture
    def test_texts(self):
        """Various test texts for comparison."""
        return [
            "Hello world",
            "The cat sat on the mat",
            "Natural language processing is fascinating",
            "Short",
            "This is a longer sentence with more tokens to test the models thoroughly.",
        ]

    @pytest.mark.slow
    def test_loss_comparison_multiple_texts(self, models_with_processing, test_texts):
        """Test loss comparison across multiple text samples."""
        hooked, bridge = models_with_processing

        for text in test_texts:
            with torch.no_grad():
                hooked_loss = hooked(text, return_type="loss")
                bridge_loss = bridge(text, return_type="loss")

            diff = abs(hooked_loss - bridge_loss)
            assert (
                diff < 0.01
            ), f"Loss difference too large for '{text}': {diff} (hooked: {hooked_loss}, bridge: {bridge_loss})"

            # Both should have reasonable losses (single-token inputs can have high loss)
            assert (
                0.0 < hooked_loss < 15.0
            ), f"HookedTransformer loss unreasonable for '{text}': {hooked_loss}"
            assert 0.0 < bridge_loss < 15.0, f"Bridge loss unreasonable for '{text}': {bridge_loss}"

    @pytest.mark.slow
    def test_logits_comparison(self, models_with_processing):
        """Test that logits match between models."""
        hooked, bridge = models_with_processing
        test_text = "Compare logits"

        with torch.no_grad():
            hooked_logits = hooked(test_text, return_type="logits")
            bridge_logits = bridge(test_text, return_type="logits")

        # Check shapes match
        assert (
            hooked_logits.shape == bridge_logits.shape
        ), f"Logits shapes should match: {hooked_logits.shape} vs {bridge_logits.shape}"

        # Check values are close
        max_diff = (hooked_logits - bridge_logits).abs().max()
        assert max_diff < 0.01, f"Logits should match closely, max diff: {max_diff}"

        # Check that both have reasonable distributions
        hooked_std = hooked_logits.std()
        bridge_std = bridge_logits.std()
        assert (
            1.0 < hooked_std < 10.0
        ), f"HookedTransformer logits std should be reasonable: {hooked_std}"
        assert 1.0 < bridge_std < 10.0, f"Bridge logits std should be reasonable: {bridge_std}"

    @pytest.mark.slow
    def test_attention_output_comparison(self, models_with_processing):
        """Test attention outputs match via cached activations."""
        hooked, bridge = models_with_processing
        test_text = "Attention test"

        # Use run_with_cache to capture attention outputs cleanly
        with torch.no_grad():
            _, hooked_cache = hooked.run_with_cache(test_text)
            _, bridge_cache = bridge.run_with_cache(test_text)

        # Compare first layer attention output (hook_z = pre-output-projection attention)
        hooked_attn_out = hooked_cache["blocks.0.attn.hook_z"]
        bridge_attn_out = bridge_cache["blocks.0.attn.hook_z"]

        assert (
            hooked_attn_out.shape == bridge_attn_out.shape
        ), f"Attention output shapes should match: {hooked_attn_out.shape} vs {bridge_attn_out.shape}"

        attn_diff = (hooked_attn_out - bridge_attn_out).abs().max()
        assert attn_diff < 0.01, f"Attention outputs should match closely: {attn_diff}"

    @pytest.mark.slow
    def test_hook_v_values_match(self, models_with_processing):
        """Test that hook_v values match between models."""
        hooked, bridge = models_with_processing
        test_text = "Hook V test"

        hooked_v_values = []
        bridge_v_values = []

        def collect_hooked_v(activation, hook):
            hooked_v_values.append(activation.clone())
            return activation

        def collect_bridge_v(activation, hook):
            bridge_v_values.append(activation.clone())
            return activation

        # Collect V values from both models
        hooked.add_hook("blocks.0.attn.hook_v", collect_hooked_v)
        bridge.add_hook("blocks.0.attn.hook_v", collect_bridge_v)

        with torch.no_grad():
            hooked(test_text, return_type="logits")
            bridge(test_text, return_type="logits")

        # Clean up hooks
        hooked.reset_hooks()
        bridge.reset_hooks()

        # Compare V values
        assert len(hooked_v_values) == 1, "Should have collected one V value from hooked"
        assert len(bridge_v_values) == 1, "Should have collected one V value from bridge"

        hooked_v = hooked_v_values[0]
        bridge_v = bridge_v_values[0]

        assert (
            hooked_v.shape == bridge_v.shape
        ), f"V shapes should match: {hooked_v.shape} vs {bridge_v.shape}"

        v_diff = (hooked_v - bridge_v).abs().max()
        # Observed: 0.000000 for distilgpt2 with matching weight processing
        assert v_diff < 0.01, f"V values should match closely: {v_diff}"

    @pytest.mark.slow
    def test_generation_consistency(self, models_with_processing):
        """Test that text generation is consistent."""
        hooked, bridge = models_with_processing
        prompt = "The future of AI"
        tokens = hooked.to_tokens(prompt)

        # Generate from both models using token input
        with torch.no_grad():
            hooked_tokens = hooked.generate(
                tokens, max_new_tokens=5, temperature=0.0, do_sample=False
            )
            bridge_tokens = bridge.generate(
                tokens, max_new_tokens=5, temperature=0.0, do_sample=False
            )

        # Convert to text for comparison
        hooked_text = hooked.to_string(hooked_tokens[0])
        bridge_text = bridge.to_string(bridge_tokens[0])

        # Deterministic generation should produce identical output
        assert (
            hooked_text == bridge_text
        ), f"Generation should match:\n  hooked: {repr(hooked_text)}\n  bridge: {repr(bridge_text)}"

    def test_batch_processing(self, models_with_processing):
        """Test batch processing works correctly for both models."""
        hooked, bridge = models_with_processing
        texts = ["First text", "Second text", "Third text for batch"]

        # Process as batch
        tokens_list = [hooked.to_tokens(text)[0] for text in texts]  # Remove batch dimension
        max_len = max(len(tokens) for tokens in tokens_list)

        # Pad tokens to same length
        padded_tokens = []
        for tokens in tokens_list:
            if len(tokens) < max_len:
                padding = torch.full(
                    (max_len - len(tokens),), hooked.tokenizer.pad_token_id or 0, dtype=tokens.dtype
                )
                tokens = torch.cat([tokens, padding])
            padded_tokens.append(tokens)

        batch_tokens = torch.stack(padded_tokens)

        with torch.no_grad():
            hooked_batch_logits = hooked(batch_tokens, return_type="logits")
            bridge_batch_logits = bridge(batch_tokens, return_type="logits")

        # Check batch dimensions
        assert (
            hooked_batch_logits.shape == bridge_batch_logits.shape
        ), f"Batch logits shapes should match: {hooked_batch_logits.shape} vs {bridge_batch_logits.shape}"

        assert hooked_batch_logits.shape[0] == len(texts), "Batch size should match input"

        # Logits should be reasonably close
        batch_diff = (hooked_batch_logits - bridge_batch_logits).abs().max()
        assert batch_diff < 0.1, f"Batch processing should produce similar results: {batch_diff}"
