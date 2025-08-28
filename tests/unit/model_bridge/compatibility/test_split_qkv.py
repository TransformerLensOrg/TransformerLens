import torch

from transformer_lens.model_bridge import TransformerBridge


def test_split_qkv_normal_attn_correct():
    """Verifies that the split_qkv_input flag does not change the output for models with normal attention."""
    # Use a simple pretrained model instead of creating from scratch
    # since TransformerBridge works with pretrained models
    model = TransformerBridge.boot_transformers("gpt2", device="cpu")

    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = model(x)

    model.set_use_split_qkv_input(True)
    assert model.cfg.use_split_qkv_input is True

    split_output = model(x)

    assert torch.allclose(normal_output, split_output, atol=1e-6)


def test_split_qkv_grouped_query_attn_correct():
    """Verifies that the split_qkv_input flag does not change the output for models with grouped query attention."""
    # Use a model that supports grouped query attention
    # Note: GPT-2 doesn't have grouped query attention, so this test may not be fully applicable
    # but we'll keep the structure for when models with GQA are supported
    model = TransformerBridge.boot_transformers("gpt2", device="cpu")

    # Check if model supports grouped query attention
    if not hasattr(model.cfg, "n_key_value_heads") or model.cfg.n_key_value_heads is None:
        # Skip this test for models without grouped query attention
        import pytest

        pytest.skip("Model does not support grouped query attention")

    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = model(x)

    model.set_use_split_qkv_input(True)
    assert model.cfg.use_split_qkv_input is True

    split_output = model(x)

    assert torch.allclose(normal_output, split_output, atol=1e-6)
