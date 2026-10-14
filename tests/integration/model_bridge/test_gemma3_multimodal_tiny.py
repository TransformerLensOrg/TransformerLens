"""Offline image-path regressions for Gemma3's mixed LayerNorm/RMSNorm components."""

import pytest
import torch
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    SiglipVisionConfig,
)

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture
def tiny_image_bridge():
    text = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    config = Gemma3Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        mm_tokens_per_image=4,
        image_token_index=31,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        model = Gemma3ForConditionalGeneration(config).eval()
        with torch.no_grad():
            # A uniform offset error can be canceled by the projector's RMSNorm.
            for module in model.model.vision_tower.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.weight.copy_(torch.linspace(0.7, 1.3, module.weight.numel()))
            # HF initializes this projection to zero, hiding vision-path regressions.
            model.model.multi_modal_projector.mm_input_projection_weight.normal_(0, 0.2)
        pixels = torch.randn(1, 3, 8, 8)
    tokens = torch.tensor([[1, 31, 31, 31, 31, 7, 8]])
    kwargs = {
        "pixel_values": pixels,
        "token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0, 0]]),
    }
    with torch.no_grad():
        reference = model(tokens, **kwargs).logits
    bridge = build_bridge_from_module(
        model, architecture="Gemma3ForConditionalGeneration", hf_config=config
    )
    return bridge, tokens, kwargs, reference


@pytest.mark.parametrize("hook_name", ["hook_scale", "hook_normalized"])
@pytest.mark.parametrize("direction", ["fwd", "bwd"])
@pytest.mark.parametrize(
    "norm_name", ["encoder_layers.0.ln1", "encoder_layers.0.ln2", "post_layernorm"]
)
def test_image_layernorm_identity_hooks_preserve_logits(
    tiny_image_bridge, hook_name, direction, norm_name
):
    bridge, tokens, kwargs, reference = tiny_image_bridge
    norm = bridge.vision_encoder.get_submodule(norm_name)
    assert isinstance(norm.original_component, torch.nn.LayerNorm)
    assert not norm.uses_rms_norm
    assert norm.config.rmsnorm_uses_offset
    assert bridge.blocks[0].ln1.uses_rms_norm
    fired = []

    def identity(value, hook):
        fired.append(True)
        return value.clone()

    with torch.no_grad():
        baseline = bridge(tokens, **kwargs)
    torch.testing.assert_close(baseline, reference, atol=1e-6, rtol=1e-5)
    message = "Backward hooks" if direction == "bwd" else "reconstructed from the hooked values"
    key = f"vision_encoder.{norm_name}.{hook_name}"
    with bridge.hooks(**{f"{direction}_hooks": [(key, identity)]}):
        with pytest.warns(UserWarning, match=message):
            actual = bridge(tokens, **kwargs)
        if direction == "bwd":
            actual.square().sum().backward()
    assert fired
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
