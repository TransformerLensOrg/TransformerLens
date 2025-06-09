"""Mock architecture adapter for testing."""
import pytest
import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
)


class MockArchitectureAdapter(ArchitectureAdapter):
    """Mock architecture adapter for testing."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.component_mapping = {
            "embed": ("embed", EmbeddingBridge),
            "unembed": ("unembed", EmbeddingBridge),
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
            "outer_blocks": (
                "outer_blocks",
                BlockBridge,
                {
                    "inner_blocks": (
                        "inner_blocks",
                        BlockBridge,
                        {"ln": ("ln", LayerNormBridge)},
                    )
                },
            ),
        }


@pytest.fixture
def mock_adapter() -> MockArchitectureAdapter:
    """Create a mock adapter."""
    return MockArchitectureAdapter()


@pytest.fixture
def mock_model_adapter() -> nn.Module:
    """Create a mock model for testing."""
    model = nn.Module()

    # For embed/unembed
    model.embed = nn.Embedding(100, 10)
    model.unembed = nn.Linear(10, 100)

    model.ln_final = nn.LayerNorm(10)
    model.blocks = nn.ModuleList()
    block = nn.Module()
    block.ln1 = nn.LayerNorm(10)
    block.ln2 = nn.LayerNorm(10)
    block.attn = nn.Module()
    block.mlp = nn.Module()
    model.blocks.append(block)

    # For nested blocks
    model.outer_blocks = nn.ModuleList()
    outer_block = nn.Module()
    outer_block.inner_blocks = nn.ModuleList()
    inner_block = nn.Module()
    inner_block.ln = nn.LayerNorm(10)
    outer_block.inner_blocks.append(inner_block)
    model.outer_blocks.append(outer_block)

    return model
