"""Tests for component setup utilities."""


from types import SimpleNamespace

import pytest
import torch.nn as nn

from tests.mocks.architecture_adapter import MockArchitectureAdapter, mock_model_adapter
from transformer_lens.model_bridge.component_setup import (
    replace_remote_component,
    set_original_components,
    setup_blocks_bridge,
    setup_components,
    setup_submodules,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
)


class TestComponentSetup:
    """Test suite for component setup utilities."""

    def test_replace_remote_component(self, mock_model_adapter):
        """Test replacing a remote component."""
        new_ln = nn.LayerNorm(10)
        replace_remote_component(new_ln, "ln_final", mock_model_adapter)
        assert mock_model_adapter.ln_final is new_ln

    def test_replace_remote_component_nested_path(self, mock_model_adapter):
        """Test replacing a nested remote component."""
        new_ln1 = nn.LayerNorm(10)
        replace_remote_component(new_ln1, "blocks.0.ln1", mock_model_adapter)
        assert mock_model_adapter.blocks[0].ln1 is new_ln1

    def test_replace_remote_component_invalid_path(self, mock_model_adapter):
        """Test replacing with an invalid path raises ValueError."""
        new_component = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Path nonexistent.path not found in model"):
            replace_remote_component(new_component, "nonexistent.path", mock_model_adapter)

    def test_replace_remote_component_invalid_attribute(self, mock_model_adapter):
        """Test replacing with an invalid attribute raises ValueError."""
        new_component = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Attribute nonexistent not found"):
            replace_remote_component(new_component, "nonexistent", mock_model_adapter)

    def test_setup_submodules_basic(self, mock_model_adapter):
        """Test setting up submodules for a component."""
        adapter = MockArchitectureAdapter()

        # Create a component with submodules
        component = AttentionBridge(
            name="self_attn",
            config=SimpleNamespace(n_heads=1),
            submodules={
                "q_proj": EmbeddingBridge(name="q_proj"),
                "k_proj": EmbeddingBridge(name="k_proj"),
            },
        )

        # Mock the original component with the expected attributes
        original_attn = nn.Module()
        original_q_proj = nn.Linear(10, 10)
        original_k_proj = nn.Linear(10, 10)
        original_attn.q_proj = original_q_proj
        original_attn.k_proj = original_k_proj

        setup_submodules(component, adapter, original_attn)

        # Check that submodules were registered and have original components set
        assert hasattr(component, "q_proj")
        assert hasattr(component, "k_proj")
        assert component.q_proj.original_component is original_q_proj
        assert component.k_proj.original_component is original_k_proj

    def test_setup_submodules_nested(self):
        """Test setting up nested submodules."""
        adapter = MockArchitectureAdapter()

        # Create a component with nested submodules
        inner_component = AttentionBridge(
            name="q_proj",
            config=SimpleNamespace(n_heads=1),
            submodules={},  # This should match a real path
        )
        component = AttentionBridge(
            name="attn",
            config=SimpleNamespace(n_heads=1),
            submodules={
                "q_proj": inner_component,
            },
        )

        # Mock the original nested structure
        original_attn = nn.Module()
        original_q_proj = nn.Linear(10, 10)
        original_attn.q_proj = original_q_proj

        setup_submodules(component, adapter, original_attn)

        # Check that nested submodules were set up correctly
        assert hasattr(component, "q_proj")
        assert component.q_proj.original_component is original_q_proj

    def test_setup_submodules_empty(self):
        """Test setting up submodules when there are none."""
        adapter = MockArchitectureAdapter()
        component = NormalizationBridge(name="ln1", config={})  # No submodules
        original_ln = nn.LayerNorm(10)

        # Should not raise any errors
        setup_submodules(component, adapter, original_ln)

    def test_setup_components_regular_component(self):
        """Test setting up regular (non-list) components."""
        adapter = MockArchitectureAdapter()
        bridge_module = nn.Module()
        mock_model = self._create_fresh_mock_model()

        components = {
            "embed": EmbeddingBridge(name="embed"),
            "ln_final": NormalizationBridge(name="ln_final", config={}),
        }

        # Store original components before setup
        original_embed = mock_model.embed
        original_ln_final = mock_model.ln_final

        setup_components(components, bridge_module, adapter, mock_model)

        # Check that components were added to bridge module and have original components set
        assert hasattr(bridge_module, "embed")
        assert hasattr(bridge_module, "ln_final")
        assert bridge_module.embed.original_component is original_embed
        assert bridge_module.ln_final.original_component is original_ln_final

    def test_setup_components_with_submodules(self):
        """Test setting up components that have submodules."""
        adapter = MockArchitectureAdapter()
        bridge_module = nn.Module()
        mock_model = self._create_fresh_mock_model()

        components = {
            "embed": EmbeddingBridge(
                name="embed", submodules={"norm": NormalizationBridge(name="norm", config={})}
            ),
        }

        # Mock the embed to have a norm attribute (a submodule, not a parameter)
        mock_model.embed.norm = nn.LayerNorm(10)
        original_embed = mock_model.embed
        original_norm = mock_model.embed.norm

        setup_components(components, bridge_module, adapter, mock_model)

        # Check that component and its submodules were set up
        assert hasattr(bridge_module, "embed")
        assert hasattr(bridge_module.embed, "norm")
        assert bridge_module.embed.original_component is original_embed
        assert bridge_module.embed.norm.original_component is original_norm

    def test_setup_blocks_bridge(self):
        """Test setting up blocks bridge with ModuleList structure."""
        adapter = MockArchitectureAdapter()
        mock_model = self._create_fresh_mock_model()

        blocks_template = BlockBridge(
            name="blocks",
            submodules={
                "ln1": NormalizationBridge(name="ln1", config={}),
                "ln2": NormalizationBridge(name="ln2", config={}),
                "attn": AttentionBridge(name="attn", config=SimpleNamespace(n_heads=1)),
                "mlp": MLPBridge(name="mlp"),
            },
        )

        # Store original blocks before setup
        original_blocks = mock_model.blocks

        bridged_blocks = setup_blocks_bridge(blocks_template, adapter, mock_model)

        # Check that we got a ModuleList with the right number of blocks
        assert isinstance(bridged_blocks, nn.ModuleList)
        assert len(bridged_blocks) == len(original_blocks)

        # Check that each block has the correct original component and basic setup
        for i, block_bridge in enumerate(bridged_blocks):
            assert block_bridge.original_component is original_blocks[i]
            assert block_bridge.name == f"blocks.{i}"

            # Check that submodules were set up (they exist as attributes)
            assert hasattr(block_bridge, "ln1")
            assert hasattr(block_bridge, "ln2")
            assert hasattr(block_bridge, "attn")
            assert hasattr(block_bridge, "mlp")

            # The submodules should be accessible and functional
            # (Whether they're bridge components or original components depends on implementation)
            assert block_bridge.ln1 is not None
            assert block_bridge.ln2 is not None
            assert block_bridge.attn is not None
            assert block_bridge.mlp is not None

    def test_setup_blocks_bridge_template_isolation(self):
        """Test that blocks bridge templates are properly copied (no shared state)."""
        adapter = MockArchitectureAdapter()
        mock_model = self._create_fresh_mock_model()

        blocks_template = BlockBridge(
            name="blocks",
            submodules={
                "ln1": NormalizationBridge(name="ln1", config={}),
            },
        )

        bridged_blocks = setup_blocks_bridge(blocks_template, adapter, mock_model)

        # Verify that each block is a separate instance
        assert bridged_blocks[0] is not bridged_blocks[1]
        assert bridged_blocks[0].ln1 is not bridged_blocks[1].ln1

        # Verify that the original template wasn't modified
        assert blocks_template.name == "blocks"
        assert not hasattr(blocks_template, "ln1")  # Submodules not registered on template

    def test_set_original_components_integration(self):
        """Test the full set_original_components integration."""

        # Create a simpler adapter without outer_blocks for this test
        class SimpleAdapter(MockArchitectureAdapter):
            def __init__(self):
                super().__init__()
                # Remove outer_blocks that cause issues in this test
                self.component_mapping = {
                    "embed": EmbeddingBridge(name="embed"),
                    "unembed": EmbeddingBridge(name="unembed"),
                    "ln_final": NormalizationBridge(name="ln_final", config={}),
                    "blocks": BlockBridge(
                        name="blocks",
                        submodules={
                            "ln1": NormalizationBridge(name="ln1", config={}),
                            "ln2": NormalizationBridge(name="ln2", config={}),
                            "attn": AttentionBridge(name="attn", config=SimpleNamespace(n_heads=1)),
                            "mlp": MLPBridge(name="mlp"),
                        },
                    ),
                }

        adapter = SimpleAdapter()
        bridge_module = nn.Module()
        mock_model = self._create_fresh_mock_model()

        # This should set up all components from the adapter's component mapping
        set_original_components(bridge_module, adapter, mock_model)

        # Check that the components are correctly bridged
        assert isinstance(bridge_module.ln_final, NormalizationBridge)
        assert isinstance(bridge_module.blocks, nn.ModuleList)
        assert len(bridge_module.blocks) == 2

    def _create_fresh_mock_model(self):
        """Create a fresh mock model for testing."""
        model = nn.Module()

        # For embed/unembed
        model.embed = nn.Embedding(100, 10)
        model.unembed = nn.Linear(10, 100)

        model.ln_final = nn.LayerNorm(10)
        model.blocks = nn.ModuleList()

        # Create two blocks for testing
        for i in range(2):
            block = nn.Module()
            block.ln1 = nn.LayerNorm(10)
            block.ln2 = nn.LayerNorm(10)
            block.attn = nn.Module()
            block.mlp = nn.Module()
            model.blocks.append(block)

        return model

    @pytest.fixture
    def mock_model_adapter(self):
        """Create a mock model for testing."""
        return self._create_fresh_mock_model()
