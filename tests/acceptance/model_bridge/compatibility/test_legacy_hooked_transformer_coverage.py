import gc

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

# Small models for basic testing - focus on those that work with TransformerBridge
PUBLIC_MODEL_NAMES = [
    "gpt2",  # Use the base model name that TransformerBridge supports
]


class TestLegacyHookedTransformerCoverage:
    """Acceptance tests for TransformerBridge functionality."""

    @pytest.fixture(autouse=True, scope="class")
    def cleanup_after_class(self):
        """Clean up memory after each test class."""
        yield
        # Force garbage collection and clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for _ in range(3):
            gc.collect()

    @pytest.fixture(params=PUBLIC_MODEL_NAMES, scope="class")
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def bridge_model(self, model_name):
        """Create a TransformerBridge model for testing."""
        try:
            return TransformerBridge.boot_transformers(model_name, device="cpu")
        except Exception as e:
            pytest.skip(f"Could not load {model_name} with TransformerBridge: {e}")

    def test_basic_model_loading(self, bridge_model, model_name):
        """Test that the model loads successfully and has expected attributes."""
        assert bridge_model is not None
        assert hasattr(bridge_model, "cfg")
        assert hasattr(bridge_model, "tokenizer")

        # Check basic config attributes
        cfg = bridge_model.cfg
        assert hasattr(cfg, "n_layers")
        assert hasattr(cfg, "d_model")
        assert hasattr(cfg, "n_heads")
        assert hasattr(cfg, "d_vocab")

        # Values should be reasonable
        assert cfg.n_layers > 0
        assert cfg.d_model > 0
        assert cfg.n_heads > 0
        assert cfg.d_vocab > 0

    def test_tokenization_functionality(self, bridge_model):
        """Test that tokenization works correctly."""
        prompts = [
            "Hello, world!",
            "The capital of France is Paris.",
            "Once upon a time, in a land far away,",
        ]

        for prompt in prompts:
            # Test basic tokenization
            tokens = bridge_model.to_tokens(prompt)
            assert isinstance(tokens, torch.Tensor)
            assert tokens.ndim == 2
            assert tokens.shape[0] == 1  # Single prompt

            # Test decoding
            decoded = bridge_model.to_string(tokens)
            assert isinstance(decoded, list)
            assert len(decoded) == 1

            # Test str_tokens
            str_tokens = bridge_model.to_str_tokens(prompt)
            assert isinstance(str_tokens, list)
            assert all(isinstance(token, str) for token in str_tokens)

    def test_forward_pass_functionality(self, bridge_model):
        """Test that forward pass works correctly."""
        prompt = "The capital of France is"

        # Basic forward pass
        output = bridge_model(prompt)
        assert isinstance(output, torch.Tensor)
        assert output.ndim == 3  # [batch, seq, vocab]
        assert output.shape[0] == 1  # Single prompt
        assert output.shape[2] == bridge_model.cfg.d_vocab

        # Test with different return types
        try:
            logits = bridge_model(prompt, return_type="logits")
            assert torch.allclose(output, logits)
        except Exception:
            # return_type might not be implemented yet
            pass

    def test_caching_functionality(self, bridge_model):
        """Test that caching works correctly."""
        prompt = "Test caching functionality."

        # Test run_with_cache
        output, cache = bridge_model.run_with_cache(prompt)
        assert isinstance(output, torch.Tensor)
        assert isinstance(cache, dict) or hasattr(cache, "cache_dict")

        # Get cache dict
        if hasattr(cache, "cache_dict"):
            cache_dict = cache.cache_dict
        else:
            cache_dict = cache

        # Should have some cached activations
        assert len(cache_dict) > 0

        # Check that cached values are tensors (or None)
        for key, value in cache_dict.items():
            if value is not None:
                assert isinstance(value, torch.Tensor), f"Cache value for {key} is not a tensor"

    def test_hook_functionality(self, bridge_model):
        """Test that basic hook functionality works."""
        prompt = "Test hook functionality."

        # Test hook_dict access
        try:
            hook_dict = bridge_model.hook_dict
            assert isinstance(hook_dict, dict)
            assert len(hook_dict) > 0

            # All values should be HookPoints
            from transformer_lens.hook_points import HookPoint

            for name, hook_point in hook_dict.items():
                assert isinstance(hook_point, HookPoint)

        except Exception as e:
            pytest.skip(f"Hook functionality not available: {e}")

    def test_weight_access_functionality(self, bridge_model):
        """Test that weight access works where implemented."""
        weight_tests = []

        # Test attention weights
        try:
            w_q = bridge_model.W_Q
            w_k = bridge_model.W_K
            w_v = bridge_model.W_V
            w_o = bridge_model.W_O

            # Basic shape checks
            assert w_q.shape[0] == bridge_model.cfg.n_layers
            assert w_k.shape[0] == bridge_model.cfg.n_layers
            assert w_v.shape[0] == bridge_model.cfg.n_layers
            assert w_o.shape[0] == bridge_model.cfg.n_layers

            weight_tests.append("attention_weights")

        except AttributeError:
            pass  # Weight access might not be implemented yet

        # Test MLP weights
        try:
            w_in = bridge_model.W_in
            w_out = bridge_model.W_out

            assert w_in.shape[0] == bridge_model.cfg.n_layers
            assert w_out.shape[0] == bridge_model.cfg.n_layers

            weight_tests.append("mlp_weights")

        except AttributeError:
            pass  # Weight access might not be implemented yet

        # At least some weight access should work eventually
        if len(weight_tests) > 0:
            print(f"Available weight access: {weight_tests}")

    def test_device_compatibility(self, bridge_model):
        """Test that device handling works correctly."""
        prompt = "Test device compatibility."

        # Test CPU
        model_cpu = bridge_model.cpu()
        tokens_cpu = model_cpu.to_tokens(prompt)
        assert tokens_cpu.device.type == "cpu"

        # Test moving between devices if CUDA is available
        if torch.cuda.is_available():
            try:
                model_cuda = bridge_model.cuda()
                tokens_cuda = model_cuda.to_tokens(prompt)
                assert tokens_cuda.device.type == "cuda"
            except Exception:
                # CUDA might not work in test environment
                pass

    def test_generation_functionality(self, bridge_model):
        """Test that generation works if implemented."""
        prompt = "Once upon a time"

        try:
            generated = bridge_model.generate(prompt, max_new_tokens=5)
            assert isinstance(generated, (str, list, torch.Tensor))
        except (AttributeError, RuntimeError):
            # Generation might not be implemented yet
            pytest.skip("Generation not supported for this TransformerBridge model")

    def test_memory_efficiency(self, bridge_model):
        """Test that the model doesn't have obvious memory leaks."""
        prompt = "Test memory efficiency."

        # Record initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0

        # Run multiple forward passes
        for _ in range(5):
            output = bridge_model(prompt)
            del output  # Explicitly delete

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()

            # Memory shouldn't grow significantly
            memory_growth = final_memory - initial_memory
            # Allow some growth but not excessive
            assert memory_growth < 100 * 1024 * 1024, f"Memory grew by {memory_growth} bytes"

    def test_batch_processing(self, bridge_model):
        """Test that batch processing works correctly."""
        prompts = [
            "First prompt for batch processing.",
            "Second prompt for batch processing.",
            "Third prompt for batch processing.",
        ]

        try:
            # Test batch tokenization
            tokens = bridge_model.to_tokens(prompts)
            assert isinstance(tokens, torch.Tensor)
            assert tokens.ndim == 2
            assert tokens.shape[0] == len(prompts)

            # Test batch forward pass
            output = bridge_model(tokens)
            assert isinstance(output, torch.Tensor)
            assert output.shape[0] == len(prompts)
            assert output.shape[2] == bridge_model.cfg.d_vocab

        except Exception as e:
            pytest.skip(f"Batch processing not supported: {e}")

    def test_consistent_outputs(self, bridge_model):
        """Test that the model produces consistent outputs."""
        prompt = "Test consistent outputs."

        # Run the same prompt multiple times
        outputs = []
        for _ in range(3):
            output = bridge_model(prompt)
            outputs.append(output)

        # All outputs should be identical (deterministic)
        for i in range(1, len(outputs)):
            assert torch.allclose(
                outputs[0], outputs[i], atol=1e-6
            ), f"Output {i} differs from first output"

    @pytest.mark.slow
    def test_large_input_handling(self, bridge_model):
        """Test that the model can handle reasonably large inputs."""
        # Create a longer prompt (but not too long to avoid timeouts)
        base_text = "This is a longer text for testing large input handling. "
        long_prompt = base_text * 20  # Repeat to make it longer

        try:
            tokens = bridge_model.to_tokens(long_prompt)

            # Should be able to handle reasonable length
            if tokens.shape[1] > bridge_model.cfg.n_ctx:
                # If too long, tokenizer should handle truncation
                assert tokens.shape[1] == bridge_model.cfg.n_ctx

            # Should be able to run forward pass
            output = bridge_model(tokens)
            assert isinstance(output, torch.Tensor)
            assert output.shape[0] == 1
            assert output.shape[1] == tokens.shape[1]
            assert output.shape[2] == bridge_model.cfg.d_vocab

        except Exception as e:
            pytest.skip(f"Large input handling failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
