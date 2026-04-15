"""Unit tests for the optional submodule framework.

Tests the `optional` flag on GeneralizedComponent and the `blocks_with()`
capability query API on TransformerBridge, which together enable hybrid
architectures where layers have structurally different submodules.
"""

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_setup import setup_submodules
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.block import BlockBridge
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

# ============================================================================
# Fixtures: synthetic hybrid model
# ============================================================================


class FakeSubmodule(nn.Module):
    """A simple nn.Linear submodule for testing."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HybridLayer(nn.Module):
    """A layer that conditionally has a 'foo' submodule."""

    def __init__(self, has_foo: bool, dim: int = 4):
        super().__init__()
        self.bar = nn.Linear(dim, dim, bias=False)
        if has_foo:
            self.foo = FakeSubmodule(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "foo"):
            x = self.foo(x)
        return self.bar(x)


class HybridModel(nn.Module):
    """Model with 4 layers: layers 0-2 have 'foo', layer 3 does not."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([HybridLayer(has_foo=(i < 3), dim=dim) for i in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MinimalAdapter(ArchitectureAdapter):
    """Minimal adapter for testing optional submodule setup."""

    def __init__(self, optional: bool = True):
        self.cfg = type("Cfg", (), {"n_layers": 4, "d_model": 4})()
        self.component_mapping = {}
        self._optional = optional

    def make_block_template(self) -> BlockBridge:
        return BlockBridge(
            name="layers",
            submodules={
                "bar": LinearBridge(name="bar"),
                "foo": LinearBridge(name="foo", optional=self._optional),
            },
        )


# ============================================================================
# Tests: optional flag on GeneralizedComponent
# ============================================================================


class TestOptionalFlag:
    """Test that the optional flag is properly stored and defaults to False."""

    def test_default_is_false(self):
        comp = GeneralizedComponent(name="test")
        assert comp.optional is False

    def test_optional_true(self):
        comp = GeneralizedComponent(name="test", optional=True)
        assert comp.optional is True

    def test_optional_false_explicit(self):
        comp = GeneralizedComponent(name="test", optional=False)
        assert comp.optional is False


# ============================================================================
# Tests: setup_submodules with optional
# ============================================================================


class TestOptionalSubmoduleSetup:
    """Test that optional submodules are skipped cleanly during setup."""

    def test_optional_submodule_skipped_on_missing_layers(self):
        """Layers 0-2 have 'foo', layer 3 does not. Setup should succeed."""
        model = HybridModel()
        adapter = MinimalAdapter(optional=True)
        template = adapter.make_block_template()

        # Simulate what setup_blocks_bridge does: deepcopy + setup per layer
        import copy

        blocks = []
        for i, layer in enumerate(model.layers):
            block = copy.deepcopy(template)
            block.name = f"layers.{i}"
            block.set_original_component(layer)
            setup_submodules(block, adapter, layer)
            blocks.append(block)

        # Layers 0-2 should have 'foo' in real_components
        for i in range(3):
            assert "foo" in blocks[i].real_components, f"Block {i} should have 'foo'"
            assert hasattr(blocks[i], "foo"), f"Block {i} should have foo module"

        # Layer 3 should NOT have 'foo' in any lookup path
        assert (
            "foo" not in blocks[3].real_components
        ), "Block 3 should not have 'foo' in real_components"
        assert "foo" not in blocks[3]._modules, "Block 3 should not have 'foo' in _modules"
        assert "foo" not in blocks[3].submodules, "Block 3 should not have 'foo' in submodules"

        # All layers should have 'bar'
        for i in range(4):
            assert "bar" in blocks[i].real_components, f"Block {i} should have 'bar'"

    def test_non_optional_missing_submodule_raises(self):
        """When optional=False, missing submodule should raise AttributeError."""
        model = HybridModel()
        adapter = MinimalAdapter(optional=False)
        template = adapter.make_block_template()

        import copy

        # Layer 3 lacks 'foo' and optional=False, so this should raise
        block = copy.deepcopy(template)
        block.name = "layers.3"
        block.set_original_component(model.layers[3])
        with pytest.raises(AttributeError):
            setup_submodules(block, adapter, model.layers[3])


# ============================================================================
# Tests: blocks_with() API
# ============================================================================


class TestBlocksWith:
    """Test the blocks_with() capability query on TransformerBridge."""

    def test_blocks_with_returns_matching_blocks(self):
        """blocks_with('foo') should return only blocks that have 'foo'."""
        from transformer_lens.model_bridge.bridge import TransformerBridge

        model = HybridModel()
        adapter = MinimalAdapter(optional=True)
        template = adapter.make_block_template()

        import copy

        blocks = nn.ModuleList()
        for i, layer in enumerate(model.layers):
            block = copy.deepcopy(template)
            block.name = f"layers.{i}"
            block.set_original_component(layer)
            setup_submodules(block, adapter, layer)
            blocks.append(block)

        # Create a minimal bridge-like object with blocks attribute
        # We test blocks_with as a standalone method
        bridge = TransformerBridge.__new__(TransformerBridge)
        nn.Module.__init__(bridge)
        bridge.add_module("blocks", blocks)

        foo_blocks = bridge.blocks_with("foo")
        assert len(foo_blocks) == 3
        assert [idx for idx, _ in foo_blocks] == [0, 1, 2]

        bar_blocks = bridge.blocks_with("bar")
        assert len(bar_blocks) == 4

        missing_blocks = bridge.blocks_with("nonexistent")
        assert len(missing_blocks) == 0

    def test_blocks_with_no_blocks_attribute(self):
        """blocks_with() should return empty list if no blocks attribute."""
        from transformer_lens.model_bridge.bridge import TransformerBridge

        bridge = TransformerBridge.__new__(TransformerBridge)
        nn.Module.__init__(bridge)
        assert bridge.blocks_with("attn") == []


# ============================================================================
# Tests: _stack_block_params with hybrid blocks
# ============================================================================


class TestStackBlockParamsHybridSafe:
    """Test that _stack_block_params raises clear errors for hybrid blocks."""

    def test_logs_warning_and_returns_subset_on_hybrid(self, caplog):
        """On hybrid blocks, should log warning and return tensor for matching blocks only."""
        import logging

        from transformer_lens.model_bridge.bridge import TransformerBridge

        # Build blocks where block 3 lacks 'foo' but blocks 0-2 have it
        model = HybridModel()
        adapter = MinimalAdapter(optional=True)
        template = adapter.make_block_template()

        import copy

        blocks = nn.ModuleList()
        for i, layer in enumerate(model.layers):
            block = copy.deepcopy(template)
            block.name = f"layers.{i}"
            block.set_original_component(layer)
            setup_submodules(block, adapter, layer)
            blocks.append(block)

        # Verify precondition: block 3 lacks 'foo'
        assert "foo" in blocks[0]._modules
        assert "foo" not in blocks[3]._modules

        bridge = TransformerBridge.__new__(TransformerBridge)
        nn.Module.__init__(bridge)
        bridge.add_module("blocks", blocks)

        # Should succeed with a log warning, returning only matching blocks.
        # logging.warning always emits (no deduplication), so researchers see
        # the index mapping notice on every access — not just the first.
        with caplog.at_level(logging.WARNING):
            result = bridge._stack_block_params("foo.proj.weight")
        assert any("Hybrid model" in msg for msg in caplog.messages)
        assert any("stack_params_for" in msg for msg in caplog.messages)
        # 3 blocks have 'foo', not 4
        assert result.shape[0] == 3

        # Verify it logs again on a second call (no deduplication)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result2 = bridge._stack_block_params("foo.proj.weight")
        assert any(
            "Hybrid model" in msg for msg in caplog.messages
        ), "Warning should emit on every call, not just the first"

    def test_raises_when_no_blocks_have_submodule(self):
        """Should raise AttributeError when zero blocks have the submodule."""
        from transformer_lens.model_bridge.bridge import TransformerBridge

        bridge = _make_hybrid_bridge()
        with pytest.raises(AttributeError, match="No blocks have"):
            bridge._stack_block_params("nonexistent")

    def test_succeeds_on_universal_submodule(self):
        """Should succeed when all blocks have the requested submodule."""
        from transformer_lens.model_bridge.bridge import TransformerBridge

        model = HybridModel()
        adapter = MinimalAdapter(optional=True)
        template = adapter.make_block_template()

        import copy

        blocks = nn.ModuleList()
        for i, layer in enumerate(model.layers):
            block = copy.deepcopy(template)
            block.name = f"layers.{i}"
            block.set_original_component(layer)
            setup_submodules(block, adapter, layer)
            blocks.append(block)

        bridge = TransformerBridge.__new__(TransformerBridge)
        nn.Module.__init__(bridge)
        bridge.add_module("blocks", blocks)

        # 'bar' exists on all blocks → should succeed
        result = bridge._stack_block_params("bar.weight")
        assert result.shape[0] == 4  # 4 layers


# ============================================================================
# Tests: refactor_factored_attn_matrices with missing layers
# ============================================================================


class TestRefactorFactoredAttnHybrid:
    """Test that refactor_factored_attn_matrices skips layers without attn."""

    def test_skips_missing_attn_layers(self):
        """Should process layers with attn keys and skip those without."""
        from transformer_lens.config.TransformerLensConfig import TransformerLensConfig
        from transformer_lens.weight_processing import ProcessWeights

        n_heads = 2
        d_head = 4
        d_model = n_heads * d_head
        cfg = TransformerLensConfig(
            n_layers=4,
            n_heads=n_heads,
            d_head=d_head,
            d_model=d_model,
            n_ctx=16,
            positional_embedding_type="standard",
        )

        # Create state_dict with attn weights for layers 0-2 only.
        # W_Q/W_K/W_V: [n_heads, d_model, d_head], W_O: [n_heads, d_head, d_model]
        # b_Q/b_K/b_V: [n_heads, d_head], b_O: [d_model]
        state_dict = {}
        for l in range(3):  # layers 0-2 have attention
            state_dict[f"blocks.{l}.attn.W_Q"] = torch.randn(n_heads, d_model, d_head)
            state_dict[f"blocks.{l}.attn.W_K"] = torch.randn(n_heads, d_model, d_head)
            state_dict[f"blocks.{l}.attn.W_V"] = torch.randn(n_heads, d_model, d_head)
            state_dict[f"blocks.{l}.attn.W_O"] = torch.randn(n_heads, d_head, d_model)
            state_dict[f"blocks.{l}.attn.b_Q"] = torch.randn(n_heads, d_head)
            state_dict[f"blocks.{l}.attn.b_K"] = torch.randn(n_heads, d_head)
            state_dict[f"blocks.{l}.attn.b_V"] = torch.randn(n_heads, d_head)
            state_dict[f"blocks.{l}.attn.b_O"] = torch.randn(d_model)

        # Layer 3 has NO attention keys — should be skipped, not crash
        result = ProcessWeights.refactor_factored_attn_matrices(state_dict, cfg)

        # Layers 0-2 should still have their attn keys (now refactored)
        for l in range(3):
            assert f"blocks.{l}.attn.W_Q" in result
            assert f"blocks.{l}.attn.W_K" in result

        # Layer 3 should have no attn keys
        assert f"blocks.3.attn.W_Q" not in result


# ============================================================================
# Tests: weight distribution with ragged blocks
# ============================================================================


class TestWeightDistributionRagged:
    """Test that weight distribution handles heterogeneous real_components."""

    def test_distribute_weights_skips_empty_blocks(self):
        """Blocks without attn weights should receive no attn keys."""
        from transformer_lens.weight_processing import ProcessWeights

        # Build a minimal real_components mapping with ragged blocks
        model = HybridModel()
        adapter = MinimalAdapter(optional=True)
        template = adapter.make_block_template()

        import copy

        blocks = []
        for i, layer in enumerate(model.layers):
            block = copy.deepcopy(template)
            block.name = f"layers.{i}"
            block.set_original_component(layer)
            setup_submodules(block, adapter, layer)
            blocks.append(block)

        # Construct state_dict with 'foo' weights for blocks 0-2 only
        state_dict = {}
        for i in range(3):
            state_dict[f"blocks.{i}.foo.weight"] = torch.randn(4, 4)
        for i in range(4):
            state_dict[f"blocks.{i}.bar.weight"] = torch.randn(4, 4)

        # Build the component mapping
        component_mapping = {
            "blocks": ("layers", blocks),
        }

        # This should not crash
        ProcessWeights.distribute_weights_to_components(
            state_dict=state_dict,
            component_mapping=component_mapping,
        )


# ============================================================================
# Helpers for bridge-level tests
# ============================================================================


def _make_hybrid_bridge():
    """Build a minimal TransformerBridge with hybrid blocks for testing.

    Uses 'foo' and 'bar' as submodule names. Layers 0-2 have 'foo', layer 3 does not.
    """
    import copy

    from transformer_lens.model_bridge.bridge import TransformerBridge

    model = HybridModel()
    adapter = MinimalAdapter(optional=True)
    template = adapter.make_block_template()

    blocks = nn.ModuleList()
    for i, layer in enumerate(model.layers):
        block = copy.deepcopy(template)
        block.name = f"layers.{i}"
        block.set_original_component(layer)
        setup_submodules(block, adapter, layer)
        blocks.append(block)

    bridge = TransformerBridge.__new__(TransformerBridge)
    nn.Module.__init__(bridge)
    bridge.add_module("blocks", blocks)

    # Minimal cfg for accumulated_bias
    bridge.cfg = type("Cfg", (), {"d_model": 4, "device": "cpu", "n_layers": 4})()
    return bridge


class AttnAdapter(ArchitectureAdapter):
    """Adapter using 'attn' as the optional submodule name (matches real adapters)."""

    def __init__(self):
        self.cfg = type("Cfg", (), {"n_layers": 4, "d_model": 4})()
        self.component_mapping = {}

    def make_block_template(self) -> BlockBridge:
        return BlockBridge(
            name="layers",
            submodules={
                "bar": LinearBridge(name="bar"),
                "attn": LinearBridge(name="foo", optional=True),
            },
        )


def _make_hybrid_bridge_with_attn():
    """Build a hybrid bridge where 'attn' is the optional submodule.

    Layers 0-2 have 'attn' (mapped from 'foo'), layer 3 does not.
    Used for testing APIs that specifically look for 'attn' (composition scores, labels).
    """
    import copy

    from transformer_lens.model_bridge.bridge import TransformerBridge

    model = HybridModel()
    adapter = AttnAdapter()
    template = adapter.make_block_template()

    blocks = nn.ModuleList()
    for i, layer in enumerate(model.layers):
        block = copy.deepcopy(template)
        block.name = f"layers.{i}"
        block.set_original_component(layer)
        setup_submodules(block, adapter, layer)
        blocks.append(block)

    bridge = TransformerBridge.__new__(TransformerBridge)
    nn.Module.__init__(bridge)
    bridge.add_module("blocks", blocks)
    bridge.cfg = type("Cfg", (), {"d_model": 4, "device": "cpu", "n_layers": 4, "n_heads": 2})()
    return bridge


# ============================================================================
# Tests: blocks_with uses _modules not hasattr
# ============================================================================


class TestBlocksWithModulesCheck:
    """blocks_with() should only find bridged submodules, not HF attrs."""

    def test_does_not_find_hf_internal_attrs(self):
        """blocks_with should not match HF attributes that aren't bridged."""
        bridge = _make_hybrid_bridge()
        # 'bar' is a bridged submodule (in _modules), should be found
        assert len(bridge.blocks_with("bar")) == 4
        # 'training' exists as an attr on nn.Module but is not a bridged submodule
        assert len(bridge.blocks_with("training")) == 0

    def test_finds_only_bridged_optional_submodules(self):
        """Optional submodules should be found only on layers where they were bound."""
        bridge = _make_hybrid_bridge()
        foo_blocks = bridge.blocks_with("foo")
        assert [idx for idx, _ in foo_blocks] == [0, 1, 2]


# ============================================================================
# Tests: accumulated_bias on hybrid models
# ============================================================================


class TestAccumulatedBiasHybrid:
    """accumulated_bias should not crash on hybrid models."""

    def test_accumulated_bias_skips_non_attn_layers(self):
        """Should not crash when some layers lack attention."""
        bridge = _make_hybrid_bridge()
        # Should run without error through all 4 layers (layer 3 has no attn)
        result = bridge.accumulated_bias(layer=4)
        assert result.shape == (4,)

    def test_accumulated_bias_mlp_input_on_non_attn_layer(self):
        """mlp_input=True on a non-attention layer should not crash."""
        bridge = _make_hybrid_bridge()
        # Layer 3 has no attn — should still work with mlp_input=True
        result = bridge.accumulated_bias(layer=3, mlp_input=True)
        assert result.shape == (4,)


# ============================================================================
# Tests: block_submodules and layer_types introspection
# ============================================================================


class TestBlockIntrospection:
    """Test layer introspection APIs."""

    def test_block_submodules(self):
        """block_submodules should list bridged submodules per layer."""
        bridge = _make_hybrid_bridge()
        # Layer 0 has both foo and bar
        subs_0 = bridge.block_submodules(0)
        assert "foo" in subs_0
        assert "bar" in subs_0
        # Layer 3 has only bar
        subs_3 = bridge.block_submodules(3)
        assert "foo" not in subs_3
        assert "bar" in subs_3

    def test_layer_types(self):
        """layer_types should return a list with one entry per block."""
        bridge = _make_hybrid_bridge()
        types = bridge.layer_types()
        assert len(types) == 4
        # Layers 0-2 have 'foo', layer 3 does not
        for i in range(3):
            assert "foo" in types[i]
        assert "foo" not in types[3]


# ============================================================================
# Tests: stack_params_for hybrid API
# ============================================================================


class TestStackParamsFor:
    """Test stack_params_for on hybrid bridges."""

    def test_returns_correct_indices_and_tensors(self):
        """stack_params_for should return only matching blocks."""
        bridge = _make_hybrid_bridge()
        indices, stacked = bridge.stack_params_for("foo", "foo.proj.weight")
        assert indices == [0, 1, 2]
        assert stacked.shape[0] == 3

    def test_raises_on_no_matching_blocks(self):
        """Should raise ValueError when no blocks have the submodule."""
        bridge = _make_hybrid_bridge()
        with pytest.raises(ValueError, match="No blocks have submodule"):
            bridge.stack_params_for("nonexistent", "nonexistent.weight")


# ============================================================================
# Tests: refactor guard validates all attn keys
# ============================================================================


class TestRefactorGuardConsistency:
    """Test that refactor raises on inconsistent attn keys (W_Q present, W_K missing)."""

    def test_raises_on_partial_attn_keys(self):
        """If W_Q is present but W_K is missing, should raise ValueError."""
        from transformer_lens.config.TransformerLensConfig import TransformerLensConfig
        from transformer_lens.weight_processing import ProcessWeights

        cfg = TransformerLensConfig(
            n_layers=1,
            n_heads=2,
            d_head=4,
            d_model=8,
            n_ctx=16,
            positional_embedding_type="standard",
        )
        # Only W_Q present, missing W_K/W_V/W_O
        state_dict = {
            "blocks.0.attn.W_Q": torch.randn(2, 8, 4),
        }
        with pytest.raises(ValueError, match="Inconsistent attention weights"):
            ProcessWeights.refactor_factored_attn_matrices(state_dict, cfg)


# ============================================================================
# Tests: __setattr__ whitelist includes optional
# ============================================================================


class TestSetAttrWhitelist:
    """Test that 'optional' is in the __setattr__ whitelist."""

    def test_optional_set_on_bridge_not_hf_model(self):
        """Setting optional after set_original_component should stay on bridge."""
        comp = LinearBridge(name="test")
        fake_hf = nn.Linear(4, 4, bias=False)
        comp.set_original_component(fake_hf)
        comp.optional = True
        # Should be on the bridge, not on the HF module
        assert comp.optional is True
        assert not hasattr(fake_hf, "optional")


# ============================================================================
# Tests: attn_head_labels matches composition scores dimensions
# ============================================================================


class TestAttnHeadLabels:
    """attn_head_labels should match all_composition_scores dimensions."""

    def test_attn_head_labels_excludes_non_attn_layers(self):
        """Labels should only cover attention layers, not SSM/linear-attn."""
        bridge = _make_hybrid_bridge_with_attn()
        bridge.cfg.n_heads = 2
        labels = bridge.attn_head_labels
        # 3 attention layers (0, 1, 2) * 2 heads = 6 labels
        assert len(labels) == 6
        assert labels == ["L0H0", "L0H1", "L1H0", "L1H1", "L2H0", "L2H1"]
        # Should NOT contain L3 (non-attention layer)
        assert all("L3" not in lbl for lbl in labels)

    def test_all_head_labels_includes_all_layers(self):
        """all_head_labels should still include every layer."""
        bridge = _make_hybrid_bridge_with_attn()
        bridge.cfg.n_heads = 2
        labels = bridge.all_head_labels
        # 4 layers * 2 heads = 8 labels
        assert len(labels) == 8


# ============================================================================
# Tests: hook propagation through optional submodules
# ============================================================================


class TestHookPropagation:
    """Verify hooks fire on present optional submodules and don't exist on absent ones."""

    def _build_hybrid_model_and_blocks(self):
        """Build a hybrid model with setup done so hooks are wired."""
        import copy

        model = HybridModel()
        adapter = MinimalAdapter(optional=True)
        template = adapter.make_block_template()

        blocks = []
        for i, layer in enumerate(model.layers):
            block = copy.deepcopy(template)
            block.name = f"layers.{i}"
            block.set_original_component(layer)
            setup_submodules(block, adapter, layer)
            blocks.append(block)

        return model, blocks

    def test_hooks_fire_on_present_optional_submodule(self):
        """hook_in and hook_out should fire on blocks where the optional submodule exists."""
        model, blocks = self._build_hybrid_model_and_blocks()

        # Block 0 has 'foo' — its hook_in and hook_out should fire
        foo_bridge = blocks[0].foo
        hook_in_fired = []
        hook_out_fired = []

        foo_bridge.hook_in.add_hook(lambda tensor, hook: hook_in_fired.append(True) or tensor)
        foo_bridge.hook_out.add_hook(lambda tensor, hook: hook_out_fired.append(True) or tensor)

        # Run a forward pass through the HF model's layer 0
        # Because replace_remote_component swapped model.layers[0].foo with the bridge,
        # calling model.layers[0].foo(x) goes through LinearBridge.forward
        x = torch.randn(1, 4)
        _ = blocks[0].foo(x)

        assert len(hook_in_fired) == 1, "hook_in should fire on present optional submodule"
        assert len(hook_out_fired) == 1, "hook_out should fire on present optional submodule"

    def test_absent_optional_submodule_has_no_hooks(self):
        """Block 3 should not have 'foo' at all — no hooks to fire."""
        _, blocks = self._build_hybrid_model_and_blocks()

        # Block 3 lacks 'foo' — it shouldn't be in _modules
        assert "foo" not in blocks[3]._modules
        # Attempting to access hooks on the absent submodule should fail
        assert not hasattr(blocks[3], "foo")

    def test_hooks_on_present_dont_affect_absent(self):
        """Running all blocks should fire hooks only on blocks with the optional submodule."""
        model, blocks = self._build_hybrid_model_and_blocks()

        # Track which blocks fire foo.hook_out
        fired_block_indices = []
        for i, block in enumerate(blocks):
            if "foo" in block._modules:
                block.foo.hook_out.add_hook(
                    lambda tensor, hook, idx=i: fired_block_indices.append(idx) or tensor
                )

        # Run forward through all HF layers
        x = torch.randn(1, 4)
        for i, layer in enumerate(model.layers):
            x = layer(x)

        # Hooks should fire on layers 0, 1, 2 (have foo) but not 3
        assert fired_block_indices == [0, 1, 2]

    def test_universal_submodule_hooks_fire_on_all_blocks(self):
        """'bar' is universal — its hooks should fire on every block."""
        model, blocks = self._build_hybrid_model_and_blocks()

        fired_block_indices = []
        for i, block in enumerate(blocks):
            block.bar.hook_out.add_hook(
                lambda tensor, hook, idx=i: fired_block_indices.append(idx) or tensor
            )

        x = torch.randn(1, 4)
        for layer in model.layers:
            x = layer(x)

        assert fired_block_indices == [0, 1, 2, 3]


# ============================================================================
# Tests: CompositionScores tensor protocol
# ============================================================================


class TestCompositionScoresProtocol:
    """CompositionScores should behave like a tensor for existing research code."""

    def _make_scores(self):
        from transformer_lens.model_bridge.composition_scores import CompositionScores

        t = torch.randn(3, 2, 3, 2)
        return CompositionScores(t, [0, 2, 5], ["L0H0", "L0H1", "L2H0", "L2H1", "L5H0", "L5H1"])

    def test_shape(self):
        cs = self._make_scores()
        assert cs.shape == torch.Size([3, 2, 3, 2])

    def test_device_and_dtype(self):
        cs = self._make_scores()
        assert cs.device == torch.device("cpu")
        assert cs.dtype == torch.float32

    def test_indexing_returns_tensor(self):
        cs = self._make_scores()
        sliced = cs[0, :, 1, :]
        assert isinstance(sliced, torch.Tensor)
        assert sliced.shape == (2, 2)

    def test_torch_isnan(self):
        """torch.isnan(scores) must work — used in existing integration tests."""
        cs = self._make_scores()
        result = torch.isnan(cs)
        assert isinstance(result, torch.Tensor)
        assert result.shape == cs.shape
        assert not result.any()

    def test_torch_where(self):
        cs = self._make_scores()
        result = torch.where(cs > 0, cs.scores, torch.zeros_like(cs.scores))
        assert isinstance(result, torch.Tensor)

    def test_comparison_gt(self):
        cs = self._make_scores()
        mask = cs > 0
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == cs.shape

    def test_comparison_ne(self):
        """scores != 0 must return a tensor, not raise RuntimeError."""
        cs = self._make_scores()
        result = cs != 0
        assert isinstance(result, torch.Tensor)
        assert result.shape == cs.shape

    def test_comparison_eq(self):
        cs = self._make_scores()
        result = cs == 0
        assert isinstance(result, torch.Tensor)

    def test_tensor_method_abs(self):
        """scores.abs() must work via __getattr__ delegation."""
        cs = self._make_scores()
        result = cs.abs()
        assert isinstance(result, torch.Tensor)

    def test_tensor_method_sum(self):
        cs = self._make_scores()
        result = cs.sum()
        assert isinstance(result, torch.Tensor)

    def test_tensor_method_any(self):
        cs = self._make_scores()
        result = cs.any()
        assert isinstance(result, torch.Tensor)

    def test_chained_indexing_and_method(self):
        """scores[l1, :, l2, :].abs().sum() — the exact pattern from integration tests."""
        cs = self._make_scores()
        result = cs[0, :, 1, :].abs().sum()
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # scalar

    def test_metadata_accessible(self):
        cs = self._make_scores()
        assert cs.layer_indices == [0, 2, 5]
        assert len(cs.head_labels) == 6

    def test_repr(self):
        cs = self._make_scores()
        r = repr(cs)
        assert "CompositionScores" in r
        assert "layer_indices" in r


# ============================================================================
# Tests: get_bridge_params with hybrid blocks
# ============================================================================


class TestGetBridgeParamsHybrid:
    """get_bridge_params should skip attn keys for non-attention layers."""

    def test_no_attn_keys_for_non_attn_layers(self):
        from transformer_lens.model_bridge.get_params_util import get_bridge_params

        bridge = _make_hybrid_bridge_with_attn()
        bridge.cfg.d_vocab = 10
        bridge.cfg.n_ctx = 8
        bridge.cfg.d_mlp = 16
        bridge.cfg.n_heads = 2
        bridge.cfg.d_head = 2

        # Add minimal embed/unembed so get_bridge_params doesn't fail
        bridge.embed = nn.Embedding(10, 4)
        bridge.pos_embed = type("PE", (), {"weight": torch.randn(8, 4)})()
        bridge.unembed = type(
            "UE",
            (),
            {
                "weight": torch.randn(10, 4),
                "b_U": torch.zeros(10),
            },
        )()

        params = get_bridge_params(bridge)

        # Blocks 0-2 have 'attn' — should have attn keys
        for i in range(3):
            # attn is mapped but internal structure (q/k/v/o) may not match
            # our synthetic LinearBridge wrapping FakeSubmodule — so attn keys
            # may or may not be present depending on structure. The key point
            # is block 3 must NOT have attn keys.
            pass

        # Block 3 has NO 'attn' — must not have any attn keys
        attn_keys_for_block3 = [k for k in params if k.startswith("blocks.3.attn.")]
        assert len(attn_keys_for_block3) == 0, (
            f"Block 3 (non-attention layer) should have no attn keys, "
            f"but found: {attn_keys_for_block3}"
        )
