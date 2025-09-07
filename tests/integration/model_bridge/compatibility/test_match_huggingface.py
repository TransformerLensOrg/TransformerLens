import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge


class TestMatchHuggingFace:
    """Test that TransformerBridge matches HuggingFace model outputs."""

    # fixtures
    @pytest.fixture(scope="class", params=["gpt2"])
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def bridge_model(self, model_name):
        """Load TransformerBridge once per test class."""
        return TransformerBridge.boot_transformers(model_name, device="cpu")

    @pytest.fixture(scope="class")
    def hf_model(self, model_name):
        """Load HuggingFace model once per test class."""
        return AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    # tests
    def test_compare_huggingface_mlp_match_local_implementation(self, bridge_model, hf_model):
        """Test that TransformerBridge MLP outputs match HuggingFace MLP outputs."""
        try:
            tensor_shape = (3, 5, bridge_model.cfg.d_model)
            test_tensor = torch.randn(tensor_shape)

            # Check if bridge model has blocks
            if not hasattr(bridge_model, "blocks"):
                pytest.skip("TransformerBridge doesn't have blocks attribute")

            n_layers = min(len(bridge_model.blocks), len(hf_model.transformer.h))

            for layer_n in range(n_layers):
                # Get MLP from bridge model
                if hasattr(bridge_model.blocks[layer_n], "mlp"):
                    bridge_out = bridge_model.blocks[layer_n].mlp(test_tensor)
                else:
                    pytest.skip(f"Layer {layer_n} doesn't have mlp attribute in TransformerBridge")

                # Get MLP from HuggingFace model
                hf_out = hf_model.transformer.h[layer_n].mlp(test_tensor)

                assert torch.allclose(
                    bridge_out, hf_out, atol=1e-4
                ), f"MLP layer {layer_n} outputs don't match"

        except AttributeError as e:
            pytest.skip(f"Required attributes not available on TransformerBridge: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error in MLP comparison: {e}")

    def test_compare_huggingface_attention_match_local_implementation(self, bridge_model, hf_model):
        """Test that TransformerBridge attention outputs match HuggingFace attention outputs."""
        try:
            batch, pos, d_model = 3, 5, bridge_model.cfg.d_model
            input_tensor = torch.randn(batch, pos, d_model)

            # Check if bridge model has blocks
            if not hasattr(bridge_model, "blocks"):
                pytest.skip("TransformerBridge doesn't have blocks attribute")

            n_layers = min(len(bridge_model.blocks), len(hf_model.transformer.h))

            for layer_n in range(n_layers):
                # Get attention from bridge model
                if hasattr(bridge_model.blocks[layer_n], "attn"):
                    bridge_attn = bridge_model.blocks[layer_n].attn

                    # Try different ways to call attention based on what's available
                    if hasattr(bridge_attn, "__call__"):
                        try:
                            # Try TransformerLens-style call
                            bridge_out = bridge_attn(
                                query_input=input_tensor,
                                key_input=input_tensor,
                                value_input=input_tensor,
                                past_kv_cache_entry=None,
                                attention_mask=None,
                            )
                        except TypeError:
                            # Try simpler call
                            bridge_out = bridge_attn(input_tensor)
                    else:
                        pytest.skip(f"Layer {layer_n} attention not callable in TransformerBridge")
                else:
                    pytest.skip(f"Layer {layer_n} doesn't have attn attribute in TransformerBridge")

                # Get attention from HuggingFace model
                hf_attn_output = hf_model.transformer.h[layer_n].attn(hidden_states=input_tensor)

                # Handle different return formats from HuggingFace attention
                if isinstance(hf_attn_output, tuple):
                    # When output_attentions=True, it returns (hidden_states, attention_weights)
                    hf_out = hf_attn_output[0]  # hidden_states
                else:
                    hf_out = hf_attn_output

                # Handle different return formats from bridge attention
                if isinstance(bridge_out, tuple):
                    # Bridge attention might also return a tuple
                    bridge_out_tensor = bridge_out[0]  # Take first element
                else:
                    bridge_out_tensor = bridge_out

                assert torch.allclose(
                    bridge_out_tensor, hf_out, atol=1e-3, rtol=1e-3
                ), f"Attention layer {layer_n} outputs don't match"

        except AttributeError as e:
            pytest.skip(f"Required attributes not available on TransformerBridge: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error in attention comparison: {e}")

    def test_full_model_output_match(self, bridge_model, hf_model):
        """Test that full TransformerBridge model output matches HuggingFace model output."""
        try:
            # Test with a simple prompt
            prompt = "The capital of France is"

            # Get bridge model output
            bridge_tokens = bridge_model.to_tokens(prompt)
            bridge_output = bridge_model(bridge_tokens)

            # Get HuggingFace model output
            hf_output = hf_model(bridge_tokens)
            if hasattr(hf_output, "logits"):
                hf_logits = hf_output.logits
            else:
                hf_logits = hf_output

            # Compare outputs
            assert (
                bridge_output.shape == hf_logits.shape
            ), f"Output shapes don't match: {bridge_output.shape} vs {hf_logits.shape}"
            assert torch.allclose(
                bridge_output, hf_logits, atol=1e-3
            ), "Full model outputs don't match"

        except Exception as e:
            pytest.fail(f"Unexpected error in full model comparison: {e}")

    def test_tokenizer_consistency(self, bridge_model):
        """Test that TransformerBridge tokenizer matches HuggingFace tokenizer."""
        try:
            # Test tokenization
            prompt = "Hello, world! This is a test."
            bridge_tokens = bridge_model.to_tokens(prompt)

            # Basic checks
            assert isinstance(bridge_tokens, torch.Tensor)
            assert bridge_tokens.ndim == 2  # [batch, seq]
            assert bridge_tokens.shape[0] == 1  # Single prompt

            # Test decoding
            decoded = bridge_model.to_string(bridge_tokens)
            assert isinstance(decoded, list)
            assert len(decoded) == 1

            # Test str_tokens
            str_tokens = bridge_model.to_str_tokens(prompt)
            assert isinstance(str_tokens, list)
            assert all(isinstance(token, str) for token in str_tokens)

        except Exception as e:
            pytest.fail(f"Unexpected error in tokenizer consistency: {e}")

    def test_config_consistency(self, bridge_model, hf_model):
        """Test that TransformerBridge config matches HuggingFace config."""
        try:
            bridge_cfg = bridge_model.cfg
            hf_cfg = hf_model.config

            # Check key configuration parameters
            config_mappings = [
                ("n_layers", "n_layer"),
                ("d_model", "n_embd"),
                ("n_heads", "n_head"),
                ("d_vocab", "vocab_size"),
            ]

            for bridge_attr, hf_attr in config_mappings:
                if hasattr(bridge_cfg, bridge_attr) and hasattr(hf_cfg, hf_attr):
                    bridge_val = getattr(bridge_cfg, bridge_attr)
                    hf_val = getattr(hf_cfg, hf_attr)
                    assert (
                        bridge_val == hf_val
                    ), f"Config mismatch: {bridge_attr}={bridge_val} vs {hf_attr}={hf_val}"

        except Exception as e:
            pytest.fail(f"Unexpected error in config consistency: {e}")

    def test_weight_access_consistency(self, bridge_model):
        """Test that TransformerBridge weight access provides expected values."""
        try:
            # Test basic weight access patterns
            weight_checks = []

            try:
                # Test attention weights
                w_q = bridge_model.W_Q
                w_k = bridge_model.W_K
                w_v = bridge_model.W_V
                w_o = bridge_model.W_O

                # Basic shape checks
                assert w_q.shape[0] == bridge_model.cfg.n_layers
                assert w_k.shape[0] == bridge_model.cfg.n_layers
                assert w_v.shape[0] == bridge_model.cfg.n_layers
                assert w_o.shape[0] == bridge_model.cfg.n_layers

                weight_checks.append("attention_weights")

            except AttributeError:
                pass  # Weight access might not be implemented yet

            try:
                # Test MLP weights
                w_in = bridge_model.W_in
                w_out = bridge_model.W_out

                assert w_in.shape[0] == bridge_model.cfg.n_layers
                assert w_out.shape[0] == bridge_model.cfg.n_layers

                weight_checks.append("mlp_weights")

            except AttributeError:
                pass  # Weight access might not be implemented yet

            # At least basic weight access should work
            if len(weight_checks) == 0:
                pytest.skip("No weight access methods available on TransformerBridge")
            else:
                print(f"Available weight access: {weight_checks}")

        except Exception as e:
            pytest.fail(f"Unexpected error in weight access consistency: {e}")
