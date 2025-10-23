import pytest
import torch

from transformer_lens import utils
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
    @pytest.mark.parametrize("padding_side", ["left", "right"])
    @pytest.mark.parametrize("prepend_bos", [True, False])
    @pytest.mark.parametrize("prompts_with_sep", [True, False])
    def test_get_attention_mask(self, model, padding_side, prepend_bos, prompts_with_sep):
        # setup
        model.tokenizer.padding_side = padding_side
        if hasattr(model.tokenizer, "sep_token_id"):
            model.tokenizer.sep_token_id = model.tokenizer.pad_token_id
        prepend_bos = prepend_bos

        # For TransformerBridge, we need to adapt the prompts format
        prompts = [
            "The quick brown fox jumps over the lazy dog",
            "Hello world, this is a test",
            "Short",
        ]

        if prompts_with_sep:
            # Add separator if model supports it
            if hasattr(model.tokenizer, "sep_token") and model.tokenizer.sep_token:
                prompts = [prompt + model.tokenizer.sep_token for prompt in prompts]

        # Get tokens using TransformerBridge's tokenization method
        tokens = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side=padding_side)

        # Test attention mask utility
        attention_mask = utils.get_attention_mask(model.tokenizer, tokens, prepend_bos)

        # Basic checks
        assert attention_mask.shape == tokens.shape
        # Attention mask should be int64 with values 0/1 for compatibility
        assert attention_mask.dtype == torch.int64

        # Check that non-padding tokens have attention_mask = True
        if hasattr(model.tokenizer, "pad_token_id") and model.tokenizer.pad_token_id is not None:
            non_padding_mask = tokens != model.tokenizer.pad_token_id
            # All non-padding positions should have attention
            assert torch.all(attention_mask >= non_padding_mask)

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
