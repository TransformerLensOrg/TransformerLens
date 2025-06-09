"""Mock architecture adapter for testing."""
import pytest
import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    LayerNormBridge,
    MLPBridge,
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