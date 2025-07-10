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
        # Use actual bridge instances instead of tuples
        self.component_mapping = {
            "embed": EmbeddingBridge(name="embed"),
            "unembed": EmbeddingBridge(name="unembed"),
            "ln_final": LayerNormBridge(name="ln_final"),
            "blocks": BlockBridge(
                name="blocks",
                submodules={
                    "ln1": LayerNormBridge(name="ln1"),
                    "ln2": LayerNormBridge(name="ln2"),
                    "attn": AttentionBridge(name="attn"),
                    "mlp": MLPBridge(name="mlp"),
                },
            ),
            "outer_blocks": BlockBridge(
                name="outer_blocks",
                submodules={
                    "inner_blocks": BlockBridge(
                        name="inner_blocks",
                        submodules={"ln": LayerNormBridge(name="ln")},
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
