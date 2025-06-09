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


class TestEndToEndBridge:
    """Test suite for the end-to-end functionality of the TransformerBridge."""

    def test_bridge_creation_and_component_access(self):
        """Test the creation of a TransformerBridge and accessing its components."""
        # Create a mock model and adapter
        model = nn.Module()
        model.ln_final = nn.LayerNorm(10)
        model.blocks = nn.ModuleList([nn.Module() for _ in range(2)])
        model.blocks[0].ln1 = nn.LayerNorm(10)
        model.blocks[0].attn = nn.Module()
        model.blocks[1].ln1 = nn.LayerNorm(10)
        model.blocks[1].attn = nn.Module()

        adapter = MockArchitectureAdapter()
        adapter.component_mapping = {
            "ln_final": ("ln_final", LayerNormBridge),
            "blocks": (
                "blocks",
                BlockBridge,
                {
                    "ln1": ("ln1", LayerNormBridge),
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
