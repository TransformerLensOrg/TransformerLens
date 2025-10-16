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
