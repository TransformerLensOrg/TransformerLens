import gc

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer


class TestMatchHuggingFace:
    # fixtures
    @pytest.fixture(scope="class", params=["gpt2"])
    def model_name(self, request):
        return request.param

    @pytest.fixture(autouse=True, scope="class")
    def cleanup_after_class(self):
        """Clean up memory after each test class."""
        yield
        # Force garbage collection and clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for _ in range(3):
            gc.collect()

    @pytest.fixture(scope="class")
    def tl_model(self, model_name):
        """Load TransformerLens model once per class."""
        return HookedTransformer.from_pretrained_no_processing(model_name, device="cpu")

    @pytest.fixture(scope="class")
    def hf_model(self, model_name):
        """Load HuggingFace model once per class."""
        return AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    # tests
    def test_compare_huggingface_mlp_match_local_implementation(
        self, model_name, tl_model, hf_model
    ):
        # Set seed for reproducible results
        torch.manual_seed(42)
        tensor_shape = (3, 5, tl_model.cfg.d_model)
        test_tensor = torch.randn(tensor_shape)

        for layer_n in range(len(tl_model.blocks)):
            tl_out = tl_model.blocks[layer_n].mlp(test_tensor)
            hf_out = hf_model.transformer.h[layer_n].mlp(test_tensor)

            assert torch.allclose(tl_out, hf_out, atol=1e-4)

    def test_compare_huggingface_attention_match_local_implementation(
        self, model_name, tl_model, hf_model
    ):
        # Set seed for reproducible results
        torch.manual_seed(43)
        batch, pos, d_model = 3, 5, tl_model.cfg.d_model
        input = torch.randn(batch, pos, d_model)

        for layer_n in range(len(tl_model.blocks)):
            # Both models should apply layer norm to the input before attention
            # HuggingFace GPT-2 attention expects raw input and applies layer norm internally
            # TransformerLens attention expects pre-normalized input

            # Apply layer norm using the same layer norm (use HF layer norm as reference)
            normalized_input = hf_model.transformer.h[layer_n].ln_1(input)

            tl_out = tl_model.blocks[layer_n].attn(
                query_input=normalized_input,
                key_input=normalized_input,
                value_input=normalized_input,
                past_kv_cache_entry=None,
                attention_mask=None,
            )

            # For HuggingFace, we need to call the attention directly without the layer norm
            # since we already applied it above
            hf_attn = hf_model.transformer.h[layer_n].attn

            # Manually compute HF attention without layer norm
            # This mimics what happens inside the HF attention module
            qkv = torch.nn.functional.linear(
                normalized_input, hf_attn.c_attn.weight.T, hf_attn.c_attn.bias
            )
            q, k, v = qkv.split(d_model, dim=2)

            # Reshape for multi-head attention
            q = q.view(batch, pos, tl_model.cfg.n_heads, tl_model.cfg.d_head).transpose(1, 2)
            k = k.view(batch, pos, tl_model.cfg.n_heads, tl_model.cfg.d_head).transpose(1, 2)
            v = v.view(batch, pos, tl_model.cfg.n_heads, tl_model.cfg.d_head).transpose(1, 2)

            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (tl_model.cfg.d_head**0.5)

            # Apply causal mask
            causal_mask = torch.tril(torch.ones(pos, pos, device=input.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

            # Apply softmax
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)

            # Reshape and apply output projection
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, pos, d_model)
            hf_out = torch.nn.functional.linear(
                attn_output, hf_attn.c_proj.weight.T, hf_attn.c_proj.bias
            )

            assert torch.allclose(tl_out, hf_out, atol=1e-4)
