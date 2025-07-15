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
        if cfg is None:
            # Create a minimal config for testing
            cfg = type(
                "MockConfig",
                (),
                {"d_mlp": 512, "intermediate_size": 512, "default_prepend_bos": True},
            )()
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

        # Set up the submodules properly by registering them as PyTorch modules
        self._setup_mock_submodules()

    def _setup_mock_submodules(self):
        """Set up submodules for testing by registering them as PyTorch modules."""
        for component_name, component in self.component_mapping.items():
            self._register_submodules(component)

    def _register_submodules(self, component):
        """Recursively register submodules for a component."""
        if component.submodules:
            for submodule_name, submodule in component.submodules.items():
                component.add_module(submodule_name, submodule)
                # Recursively register nested submodules
                self._register_submodules(submodule)


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
