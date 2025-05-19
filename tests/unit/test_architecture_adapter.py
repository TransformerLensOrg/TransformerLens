"""Tests for the architecture adapter."""

import pytest
import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.supported_architectures.gemma3 import (
    Gemma3ArchitectureAdapter,
)
from transformer_lens.TransformerLensConfig import TransformerLensConfig


class MockGemma3Model(nn.Module):
    """A mock implementation of the Gemma 3 model architecture for testing purposes.
    
    This mock model replicates the key architectural components of Gemma 3:
    - Embedding layer (embed_tokens)
    - Multiple transformer layers with:
        - Input and post-attention layer norms
        - Self-attention with Q, K, V, O projections
        - MLP with up, gate, and down projections
    - Final layer norm
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(1000, 512)
        self.model.layers = nn.ModuleList([
            nn.Module() for _ in range(2)
        ])
        for layer in self.model.layers:
            layer.input_layernorm = nn.LayerNorm(512)
            layer.post_attention_layernorm = nn.LayerNorm(512)
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(512, 512)
            layer.self_attn.k_proj = nn.Linear(512, 512)
            layer.self_attn.v_proj = nn.Linear(512, 512)
            layer.self_attn.o_proj = nn.Linear(512, 512)
            layer.mlp = nn.Module()
            layer.mlp.up_proj = nn.Linear(512, 2048)
            layer.mlp.gate_proj = nn.Linear(512, 2048)
            layer.mlp.down_proj = nn.Linear(2048, 512)
        self.model.norm = nn.LayerNorm(512)
        self.embed_tokens = self.model.embed_tokens  # For shared embedding/unembedding


@pytest.fixture
def cfg() -> TransformerLensConfig:
    """Create a test config."""
    return TransformerLensConfig(
        d_model=512,
        d_head=64,
        n_layers=2,
        n_heads=8,
        n_ctx=1024,
        d_vocab=1000,
        act_fn="silu",
    )


@pytest.fixture
def adapter(cfg: TransformerLensConfig) -> Gemma3ArchitectureAdapter:
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
    assert adapter.translate_transformer_lens_path("unembed") == "model.embed_tokens"
    
    # Test block mapping
    assert adapter.translate_transformer_lens_path("blocks") == "model.layers"
    assert adapter.translate_transformer_lens_path("blocks.0") == "model.layers.0"
    assert adapter.translate_transformer_lens_path("blocks.1") == "model.layers.1"
    
    # Test block subcomponent mapping
    assert adapter.translate_transformer_lens_path("blocks.0.ln1") == "model.layers.0.input_layernorm"
    assert adapter.translate_transformer_lens_path("blocks.0.ln2") == "model.layers.0.post_attention_layernorm"
    assert adapter.translate_transformer_lens_path("blocks.0.attn") == "model.layers.0.self_attn"
    assert adapter.translate_transformer_lens_path("blocks.0.mlp") == "model.layers.0.mlp"


def test_translate_transformer_lens_path_last_component(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test path translation with last_component_only=True."""
    # Test direct mapping
    assert adapter.translate_transformer_lens_path("embed", last_component_only=True) == "embed_tokens"
    assert adapter.translate_transformer_lens_path("ln_final", last_component_only=True) == "norm"
    assert adapter.translate_transformer_lens_path("unembed", last_component_only=True) == "embed_tokens"
    
    # Test block mapping
    assert adapter.translate_transformer_lens_path("blocks", last_component_only=True) == "layers"
    assert adapter.translate_transformer_lens_path("blocks.0", last_component_only=True) == "0"
    assert adapter.translate_transformer_lens_path("blocks.1", last_component_only=True) == "1"
    
    # Test block subcomponent mapping
    assert adapter.translate_transformer_lens_path("blocks.0.ln1", last_component_only=True) == "input_layernorm"
    assert adapter.translate_transformer_lens_path("blocks.0.ln2", last_component_only=True) == "post_attention_layernorm"
    assert adapter.translate_transformer_lens_path("blocks.0.attn", last_component_only=True) == "self_attn"
    assert adapter.translate_transformer_lens_path("blocks.0.mlp", last_component_only=True) == "mlp"


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