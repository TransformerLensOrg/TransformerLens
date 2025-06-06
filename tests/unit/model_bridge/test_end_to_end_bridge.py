"""End-to-end tests for the TransformerBridge."""
from unittest.mock import MagicMock

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
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


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_final = nn.LayerNorm(10)
        self.h = nn.ModuleList([nn.Module() for _ in range(2)])
        for i in range(2):
            self.h[i].ln_1 = nn.LayerNorm(10)
            # This needs to be a full module for replacement to work
            attn_module = nn.Module()
            attn_module.q_proj = nn.Linear(10, 10)
            attn_module.k_proj = nn.Linear(10, 10)
            attn_module.v_proj = nn.Linear(10, 10)
            attn_module.out_proj = nn.Linear(10, 10)
            self.h[i].attn = attn_module


class MockAdapter(ArchitectureAdapter):
    def __init__(self, model):
        super().__init__(model)
        self.cfg = MagicMock()
        self.cfg.n_layers = 2
        self.component_mapping: ComponentMapping = {
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


def test_end_to_end_bridge_creation():
    """Test the full process of creating a bridged model."""
    remote_model = MockModel()
    adapter = MockAdapter(remote_model)
    tokenizer = MagicMock()

    bridge = TransformerBridge(model=remote_model, adapter=adapter, tokenizer=tokenizer)

    # Check that components are replaced on the remote model
    assert isinstance(remote_model.ln_final, LayerNormBridge)
    assert isinstance(remote_model.h[0], MockBlock)
    assert isinstance(remote_model.h[0].ln_1, LayerNormBridge)
    assert isinstance(remote_model.h[0].attn, AttentionBridge)
    assert isinstance(remote_model.h[1].ln_1, LayerNormBridge)
    assert isinstance(remote_model.h[1].attn, AttentionBridge)

    # Check that components are set on the bridge
    assert hasattr(bridge, "ln_final")
    assert hasattr(bridge, "blocks")
    assert not hasattr(bridge, "blocks.0")  # Should not be set individually
    assert isinstance(bridge.ln_final, LayerNormBridge)
    assert isinstance(bridge.blocks, nn.ModuleList)
    assert len(bridge.blocks) == 2
    assert isinstance(bridge.blocks[0], MockBlock)

    # After bridging, the sub-components should now be on the bridged block
    assert isinstance(bridge.blocks[0].ln_1, LayerNormBridge)
    assert isinstance(bridge.blocks[0].attn, AttentionBridge)
