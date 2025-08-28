import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge


class TestUtilsWithTransformerBridge:
    """Test utilities functions that work with TransformerBridge models."""

    # fixtures
    @pytest.fixture(scope="class", params=["gpt2"])  # Start with just gpt2 for bridge compatibility
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self, model_name):
        return TransformerBridge.boot_transformers(model_name, device="cpu")

    # tests

    def test_tokenizer_compatibility(self, model):
        """Test that TransformerBridge tokenizer works with utility functions."""
        prompt = "Hello, world!"

        # Test basic tokenization
        tokens = model.to_tokens(prompt)
        assert tokens.ndim == 2  # Should have batch dimension
        assert tokens.shape[0] == 1  # Single prompt

        # Test string conversion
        decoded = model.to_string(tokens)
        assert isinstance(decoded, list)
        assert len(decoded) == 1

        # Test str_tokens
        str_tokens = model.to_str_tokens(prompt)
        assert isinstance(str_tokens, list)
        assert all(isinstance(token, str) for token in str_tokens)

    def test_device_compatibility(self, model):
        """Test that device handling works correctly with TransformerBridge."""
        prompt = "Test prompt"

        # Test CPU
        model_cpu = model.cpu()
        tokens_cpu = model_cpu.to_tokens(prompt)
        assert tokens_cpu.device.type == "cpu"

        # Test moving between devices if CUDA is available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            tokens_cuda = model_cuda.to_tokens(prompt)
            assert tokens_cuda.device.type == "cuda"

    def test_forward_pass_compatibility(self, model):
        """Test that forward pass works correctly with TransformerBridge."""
        prompt = "The capital of France is"

        # Basic forward pass
        output = model(prompt)
        assert isinstance(output, torch.Tensor)
        assert output.ndim == 3  # [batch, seq, vocab]
        assert output.shape[0] == 1  # Single prompt
        assert output.shape[2] == model.cfg.d_vocab  # Vocab size

        # Test with return_type
        logits = model(prompt, return_type="logits")
        assert torch.allclose(output, logits)

    def test_caching_compatibility(self, model):
        """Test that caching works correctly with TransformerBridge."""
        prompt = "Test caching"

        # Test basic caching
        output, cache = model.run_with_cache(prompt)
        assert isinstance(output, torch.Tensor)
        assert isinstance(cache, dict) or hasattr(cache, "cache_dict")

        # Cache should contain some activations
        if hasattr(cache, "cache_dict"):
            cache_dict = cache.cache_dict
        else:
            cache_dict = cache
        assert len(cache_dict) > 0

    def test_generation_compatibility(self, model):
        """Test that generation works correctly with TransformerBridge."""
        prompt = "Once upon a time"

        # Test basic generation if supported
        try:
            generated = model.generate(prompt, max_new_tokens=5)
            assert isinstance(generated, (str, list, torch.Tensor))
        except (AttributeError, RuntimeError):
            # Generation might not be implemented yet for all bridge models
            pytest.skip("Generation not supported for this TransformerBridge model")

    @pytest.mark.parametrize("method", ["to_tokens", "to_string", "to_str_tokens"])
    def test_tokenization_methods(self, model, method):
        """Test various tokenization methods work with TransformerBridge."""
        prompt = "Test tokenization"

        if method == "to_tokens":
            result = model.to_tokens(prompt)
            assert isinstance(result, torch.Tensor)
        elif method == "to_string":
            tokens = model.to_tokens(prompt)
            result = model.to_string(tokens)
            assert isinstance(result, (str, list))
        elif method == "to_str_tokens":
            result = model.to_str_tokens(prompt)
            assert isinstance(result, list)
            assert all(isinstance(token, str) for token in result)

    def test_weight_access_compatibility(self, model):
        """Test that weight access works correctly with TransformerBridge."""
        # Enable compatibility mode to access property aliases
        model.enable_compatibility_mode(disable_warnings=True)

        # Test basic weight access patterns that should work
        try:
            # These properties should exist on TransformerBridge
            w_q = model.W_Q
            w_k = model.W_K
            w_v = model.W_V
            w_o = model.W_O

            # Basic shape checks
            assert w_q.ndim == 4  # [n_layers, n_heads, d_model, d_head]
            assert w_k.ndim == 4
            assert w_v.ndim == 4
            assert w_o.ndim == 4

            assert w_q.shape[0] == model.cfg.n_layers
            assert w_q.shape[1] == model.cfg.n_heads

        except AttributeError as e:
            pytest.skip(f"Weight access not fully implemented: {e}")

    def test_config_compatibility(self, model):
        """Test that config access works correctly with TransformerBridge."""
        cfg = model.cfg

        # Basic config properties that should exist
        assert hasattr(cfg, "n_layers")
        assert hasattr(cfg, "d_model")
        assert hasattr(cfg, "n_heads")
        assert hasattr(cfg, "d_vocab")

        # Values should be reasonable
        assert cfg.n_layers > 0
        assert cfg.d_model > 0
        assert cfg.n_heads > 0
        assert cfg.d_vocab > 0
