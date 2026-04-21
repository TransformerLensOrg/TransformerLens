"""Unit tests for multimodal support in TransformerBridge."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    NormalizationBridge,
)


def _make_adapter(is_multimodal=False):
    """Create a minimal adapter with optional multimodal flag."""
    cfg = TransformerBridgeConfig(
        d_model=10,
        d_head=5,
        n_layers=1,
        n_ctx=64,
        d_vocab=100,
        architecture="TestArch",
    )
    if is_multimodal:
        cfg.is_multimodal = True

    class MinimalAdapter(ArchitectureAdapter):
        def __init__(self, cfg):
            super().__init__(cfg)
            attn_cfg = SimpleNamespace(n_heads=2)
            self.component_mapping = {
                "ln_final": NormalizationBridge(name="final_norm", config=self.cfg),
                "blocks": BlockBridge(
                    name="encoder.layers",
                    submodules={
                        "ln1": NormalizationBridge(name="norm1", config=self.cfg),
                        "attn": AttentionBridge(name="self_attn", config=attn_cfg),
                    },
                ),
            }

    return MinimalAdapter(cfg)


def _make_model():
    """Create a minimal nn.Module matching the adapter's component mapping paths."""
    model = nn.Module()
    model.final_norm = nn.LayerNorm(10)
    model.encoder = nn.Module()
    model.encoder.layers = nn.ModuleList([nn.Module()])
    model.encoder.layers[0].norm1 = nn.LayerNorm(10)
    model.encoder.layers[0].self_attn = nn.Module()

    # Make model callable — return logits-like output
    def forward(input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, 100)
        return SimpleNamespace(logits=logits, past_key_values=None)

    model.forward = forward
    model.__call__ = forward
    return model


def _make_bridge(is_multimodal=False):
    """Create a TransformerBridge with a minimal model and adapter."""
    adapter = _make_adapter(is_multimodal=is_multimodal)
    model = _make_model()
    bridge = TransformerBridge(model, adapter, tokenizer=MagicMock())
    return bridge


class TestForwardPixelValues:
    """Test pixel_values handling in forward()."""

    def test_forward_rejects_pixel_values_for_text_model(self):
        """Passing pixel_values to a text-only model should raise ValueError."""
        bridge = _make_bridge(is_multimodal=False)
        pixel_values = torch.randn(1, 3, 224, 224)

        with pytest.raises(ValueError, match="pixel_values can only be passed to multimodal"):
            bridge.forward(torch.tensor([[1, 2, 3]]), pixel_values=pixel_values)

    def test_forward_passes_pixel_values_to_original_model(self):
        """pixel_values should be passed through to original_model for multimodal models."""
        bridge = _make_bridge(is_multimodal=True)

        # Track what kwargs are passed to the model
        received_kwargs = {}
        original_forward = bridge.__dict__["original_model"].forward

        def tracking_forward(input_ids, **kwargs):
            received_kwargs.update(kwargs)
            return original_forward(input_ids, **kwargs)

        bridge.__dict__["original_model"].forward = tracking_forward
        bridge.__dict__["original_model"].__call__ = tracking_forward

        pixel_values = torch.randn(1, 3, 224, 224)
        bridge.forward(torch.tensor([[1, 2, 3]]), pixel_values=pixel_values)

        assert "pixel_values" in received_kwargs
        assert torch.equal(received_kwargs["pixel_values"], pixel_values)

    def test_forward_without_pixel_values_no_injection(self):
        """Forward without pixel_values should not inject pixel_values into kwargs."""
        bridge = _make_bridge(is_multimodal=True)

        received_kwargs = {}
        original_forward = bridge.__dict__["original_model"].forward

        def tracking_forward(input_ids, **kwargs):
            received_kwargs.update(kwargs)
            return original_forward(input_ids, **kwargs)

        bridge.__dict__["original_model"].forward = tracking_forward
        bridge.__dict__["original_model"].__call__ = tracking_forward

        bridge.forward(torch.tensor([[1, 2, 3]]))
        assert "pixel_values" not in received_kwargs


class TestGeneratePixelValues:
    """Test pixel_values handling in generate()."""

    def test_generate_pixel_values_first_iteration_only(self):
        """pixel_values should only be passed on the first generation step."""
        bridge = _make_bridge(is_multimodal=True)
        bridge.tokenizer.eos_token_id = 2
        bridge.tokenizer.decode = MagicMock(return_value="test output")

        # Track forward calls and their kwargs
        call_kwargs_list = []
        original_forward = bridge.forward

        def tracking_forward(input, **kwargs):
            call_kwargs_list.append(dict(kwargs))
            # Return random logits matching expected shape
            if isinstance(input, torch.Tensor):
                batch_size, seq_len = input.shape
            else:
                batch_size, seq_len = 1, 3
            return torch.randn(batch_size, seq_len, 100)

        bridge.forward = tracking_forward

        pixel_values = torch.randn(1, 3, 224, 224)
        bridge.generate(
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=3,
            pixel_values=pixel_values,
            return_type="tokens",
            stop_at_eos=False,
        )

        # First call should have pixel_values, subsequent calls should not
        assert len(call_kwargs_list) == 3
        assert "pixel_values" in call_kwargs_list[0]
        assert "pixel_values" not in call_kwargs_list[1]
        assert "pixel_values" not in call_kwargs_list[2]


class TestPrepareMultimodalInputs:
    """Test prepare_multimodal_inputs() method."""

    def test_raises_for_text_only_model(self):
        """Should raise ValueError for non-multimodal models."""
        bridge = _make_bridge(is_multimodal=False)

        with pytest.raises(ValueError, match="requires a multimodal model"):
            bridge.prepare_multimodal_inputs("test text")

    def test_raises_when_no_processor(self):
        """Should raise ValueError when processor is None."""
        bridge = _make_bridge(is_multimodal=True)
        assert bridge.processor is None

        with pytest.raises(ValueError, match="No processor available"):
            bridge.prepare_multimodal_inputs("test text")

    def test_calls_processor_with_inputs(self):
        """Should call processor with text and images, move tensors to device."""
        bridge = _make_bridge(is_multimodal=True)

        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
            "attention_mask": torch.ones(1, 3),
        }
        bridge.processor = mock_processor

        result = bridge.prepare_multimodal_inputs("test text", images="fake_image")

        mock_processor.assert_called_once_with(
            text="test text", images="fake_image", return_tensors="pt"
        )
        assert "input_ids" in result
        assert "pixel_values" in result
        assert "attention_mask" in result


class TestProcessorAttribute:
    """Test that processor attribute is always available."""

    def test_processor_defaults_to_none(self):
        """Bridge should always have processor attribute, defaulting to None."""
        bridge = _make_bridge(is_multimodal=False)
        assert hasattr(bridge, "processor")
        assert bridge.processor is None

    def test_processor_settable(self):
        """processor should be settable after construction."""
        bridge = _make_bridge(is_multimodal=True)
        mock_processor = MagicMock()
        bridge.processor = mock_processor
        assert bridge.processor is mock_processor
