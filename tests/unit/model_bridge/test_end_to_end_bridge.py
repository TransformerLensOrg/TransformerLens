"""End-to-end tests for the TransformerBridge."""
from unittest.mock import MagicMock

import torch.nn as nn

from tests.mocks.architecture_adapter import MockArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    LayerNormBridge,
)
from transformer_lens.model_bridge.types import ComponentMapping


class MockBlock(BlockBridge):
    """Mock block for testing."""

    pass


class TestEndToEndBridge:
    """Test suite for the end-to-end functionality of the TransformerBridge."""

    def test_bridge_creation_and_component_access(self):
        """Test the creation of a TransformerBridge and accessing its components."""
        # Create a mock model and adapter
        model = nn.Module()
        model.h = nn.ModuleList([nn.Module() for _ in range(2)])
        model.h[0].ln_1 = nn.LayerNorm(10)
        model.h[0].attn = nn.Module()
        model.h[1].ln_1 = nn.LayerNorm(10)
        model.h[1].attn = nn.Module()
        model.ln_final = nn.LayerNorm(10)

        adapter = MockArchitectureAdapter(model)
        adapter.cfg = MagicMock()
        adapter.cfg.n_layers = 2
        adapter.component_mapping = {
            "ln_final": ("ln_final", LayerNormBridge),
            "blocks": (
                "h",
                MockBlock,  # Mock block bridge
                {
                    "ln1": ("ln_1", LayerNormBridge),
                    "attn": ("attn", AttentionBridge),
                },
            ),
        }

        # Create the bridge
        bridge = TransformerBridge(model, adapter, tokenizer=MagicMock())

        # Check that the components are correctly bridged
        assert isinstance(bridge.ln_final, LayerNormBridge)
        assert isinstance(bridge.blocks, nn.ModuleList)
        assert len(bridge.blocks) == 2
        assert isinstance(bridge.blocks[0], MockBlock)
        assert isinstance(bridge.blocks[0].ln1, LayerNormBridge)
        assert isinstance(bridge.blocks[0].attn, AttentionBridge)
        assert bridge.blocks[0].ln1.name == "blocks.0.ln1"
        assert bridge.blocks[0].attn.name == "blocks.0.attn"
