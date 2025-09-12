"""End-to-end tests for the TransformerBridge."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch.nn as nn

from tests.mocks.architecture_adapter import MockArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    NormalizationBridge,
)


class TestEndToEndBridge:
    """Test suite for the end-to-end functionality of the TransformerBridge."""

    def test_bridge_creation_and_component_access(self):
        """Test the creation of a TransformerBridge and accessing its components."""
        # Create a mock model with different naming conventions
        model = nn.Module()
        model.final_norm = nn.LayerNorm(10)
        model.encoder = nn.Module()
        model.encoder.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        model.encoder.layers[0].norm1 = nn.LayerNorm(10)
        model.encoder.layers[0].self_attn = nn.Module()
        model.encoder.layers[1].norm1 = nn.LayerNorm(10)
        model.encoder.layers[1].self_attn = nn.Module()

        adapter = MockArchitectureAdapter()
        # The mapping should now reflect the different names in the remote model
        adapter.component_mapping = {
            "ln_final": NormalizationBridge(name="final_norm", config={}),
            "blocks": BlockBridge(
                name="encoder.layers",
                submodules={
                    "ln1": NormalizationBridge(name="norm1", config={}),
                    "attn": AttentionBridge(name="self_attn", config=SimpleNamespace(n_heads=1)),
                },
            ),
        }

        # Create the bridge
        bridge = TransformerBridge(model, adapter, tokenizer=MagicMock())

        # Check that the components are correctly bridged
        assert isinstance(bridge.ln_final, NormalizationBridge)
        assert isinstance(bridge.blocks, nn.ModuleList)
        assert len(bridge.blocks) == 2
