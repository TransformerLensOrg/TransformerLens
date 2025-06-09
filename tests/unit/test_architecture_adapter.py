"""Tests for the architecture adapter."""

import pytest
import torch.nn as nn

from tests.mocks.models import MockGemma3Model
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    LayerNormBridge,
    MLPBridge,
)
from transformer_lens.model_bridge.supported_architectures.gemma3 import (
    Gemma3ArchitectureAdapter,
)


class MockArchitectureAdapter(ArchitectureAdapter):
    """Mock architecture adapter for testing."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.component_mapping = {
            "ln_final": ("ln_final", LayerNormBridge),
            "blocks": (
                "blocks",
                BlockBridge,
                {
                    "ln1": ("ln1", LayerNormBridge),
                    "ln2": ("ln2", LayerNormBridge),
                    "attn": ("attn", AttentionBridge),
                    "mlp": ("mlp", MLPBridge),
                },
            ),
        }


@pytest.fixture
def mock_adapter() -> MockArchitectureAdapter:
    """Create a mock adapter."""
    return MockArchitectureAdapter()


@pytest.fixture
def mock_model() -> nn.Module:
    """Create a mock model for testing."""
    model = nn.Module()
    model.ln_final = nn.LayerNorm(10)
    model.blocks = nn.ModuleList()
    block = nn.Module()
    block.ln1 = nn.LayerNorm(10)
    block.ln2 = nn.LayerNorm(10)
    block.attn = nn.Module()
    block.mlp = nn.Module()
    model.blocks.append(block)
    return model


def test_get_remote_component_with_mock(
    mock_adapter: MockArchitectureAdapter, mock_model: nn.Module
):
    """Test get_remote_component with the mock adapter."""
    # Test direct mapping
    ln_final = mock_adapter.get_component(mock_model, "ln_final")
    assert isinstance(ln_final, nn.LayerNorm)

    # Test block mapping
    block = mock_adapter.get_component(mock_model, "blocks.0")
    assert isinstance(block, nn.Module)

    # Test block subcomponent mapping
    ln1 = mock_adapter.get_component(mock_model, "blocks.0.ln1")
    assert isinstance(ln1, nn.LayerNorm)

    attn = mock_adapter.get_component(mock_model, "blocks.0.attn")
    assert isinstance(attn, nn.Module)

    mlp = mock_adapter.get_component(mock_model, "blocks.0.mlp")
    assert isinstance(mlp, nn.Module)


class DummyHFConfig:
    def __init__(self):
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.hidden_size = 128
        # Add any other attributes needed by the adapter here


@pytest.fixture
def cfg():
    return DummyHFConfig()


@pytest.fixture
def adapter(cfg) -> Gemma3ArchitectureAdapter:
    """Create a Gemma3 adapter."""
    return Gemma3ArchitectureAdapter(cfg)


@pytest.fixture
def model() -> MockGemma3Model:
    """Create a mock Gemma 3 model."""
    return MockGemma3Model()


def test_translate_transformer_lens_path(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test path translation from TransformerLens to Remote paths."""
    # Test direct mapping
    assert adapter.translate_transformer_lens_path("embed") == "model.embed_tokens"
    assert adapter.translate_transformer_lens_path("ln_final") == "model.norm"
    assert adapter.translate_transformer_lens_path("unembed") == "lm_head"

    # Test block mapping
    assert adapter.translate_transformer_lens_path("blocks") == "model.layers"
    assert adapter.translate_transformer_lens_path("blocks.0") == "model.layers.0"
    assert adapter.translate_transformer_lens_path("blocks.1") == "model.layers.1"

    # Test block subcomponent mapping
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1") == "model.layers.0.input_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2")
        == "model.layers.0.post_attention_layernorm"
    )
    assert adapter.translate_transformer_lens_path("blocks.0.attn") == "model.layers.0.self_attn"
    assert adapter.translate_transformer_lens_path("blocks.0.mlp") == "model.layers.0.mlp"


def test_translate_transformer_lens_path_last_component(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test path translation with last_component_only=True."""
    # Test direct mapping
    assert (
        adapter.translate_transformer_lens_path("embed", last_component_only=True) == "embed_tokens"
    )
    assert adapter.translate_transformer_lens_path("ln_final", last_component_only=True) == "norm"
    assert (
        adapter.translate_transformer_lens_path("unembed", last_component_only=True)
        == "lm_head"
    )

    # Test block mapping
    assert adapter.translate_transformer_lens_path("blocks", last_component_only=True) == "layers"
    assert adapter.translate_transformer_lens_path("blocks.0", last_component_only=True) == "0"
    assert adapter.translate_transformer_lens_path("blocks.1", last_component_only=True) == "1"

    # Test block subcomponent mapping
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1", last_component_only=True)
        == "input_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2", last_component_only=True)
        == "post_attention_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn", last_component_only=True)
        == "self_attn"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp", last_component_only=True) == "mlp"
    )


def test_get_component(adapter: Gemma3ArchitectureAdapter, model: MockGemma3Model) -> None:
    """Test getting components from the model."""
    # Test direct mapping
    assert isinstance(adapter.get_component(model, "embed"), nn.Embedding)

    # Test block mapping
    block = adapter.get_component(model, "blocks.0")
    assert isinstance(block, nn.Module)

    # Test block subcomponent mapping
    ln1 = adapter.get_component(model, "blocks.0.ln1")
    assert isinstance(ln1, nn.LayerNorm)

    attn = adapter.get_component(model, "blocks.0.attn")
    assert isinstance(attn, nn.Module)

    mlp = adapter.get_component(model, "blocks.0.mlp")
    assert isinstance(mlp, nn.Module)


def test_invalid_paths(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test handling of invalid paths."""
    with pytest.raises(ValueError, match="Component not_found not found in component mapping"):
        adapter.translate_transformer_lens_path("not_found")

    with pytest.raises(ValueError, match="Expected index, got invalid"):
        adapter.translate_transformer_lens_path("blocks.invalid")

    with pytest.raises(ValueError, match="Component not_found not found in blocks components"):
        adapter.translate_transformer_lens_path("blocks.0.not_found")
