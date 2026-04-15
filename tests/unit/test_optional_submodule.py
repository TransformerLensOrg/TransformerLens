"""Tests for optional submodule support in hybrid architectures."""

import copy
import logging

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

# -- Synthetic hybrid model fixtures ------------------------------------------


class FakeSubmodule(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HybridLayer(nn.Module):
    """Layer that conditionally has a 'foo' submodule."""

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
    """4 layers: 0-2 have 'foo', layer 3 does not."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([HybridLayer(has_foo=(i < 3), dim=dim) for i in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MinimalAdapter(ArchitectureAdapter):
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


class AttnAdapter(ArchitectureAdapter):
    """Uses 'attn' as the optional submodule name (matches real adapters)."""

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


# -- Bridge construction helpers ----------------------------------------------


def _setup_blocks(model, adapter):
    """Deepcopy template per layer and run setup_submodules."""
    template = adapter.make_block_template()
    blocks = []
    for i, layer in enumerate(model.layers):
        block = copy.deepcopy(template)
        block.name = f"layers.{i}"
        block.set_original_component(layer)
        setup_submodules(block, adapter, layer)
        blocks.append(block)
    return blocks


def _make_bridge(blocks, **cfg_attrs):
    """Wrap blocks in a minimal TransformerBridge shell."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    bridge = TransformerBridge.__new__(TransformerBridge)
    nn.Module.__init__(bridge)
    bridge.add_module("blocks", nn.ModuleList(blocks))
    defaults = {"d_model": 4, "device": "cpu", "n_layers": 4}
    defaults.update(cfg_attrs)
    bridge.cfg = type("Cfg", (), defaults)()
    return bridge


def _make_hybrid_bridge():
    """Hybrid bridge with 'foo' (optional) and 'bar' (universal)."""
    return _make_bridge(_setup_blocks(HybridModel(), MinimalAdapter(optional=True)))


def _make_hybrid_bridge_with_attn():
    """Hybrid bridge where 'attn' is the optional submodule."""
    return _make_bridge(
        _setup_blocks(HybridModel(), AttnAdapter()),
        n_heads=2,
    )


# -- Tests: optional flag -----------------------------------------------------


class TestOptionalFlag:
    def test_default_is_false(self):
        assert GeneralizedComponent(name="test").optional is False

    def test_optional_true(self):
        assert GeneralizedComponent(name="test", optional=True).optional is True

    def test_optional_false_explicit(self):
        assert GeneralizedComponent(name="test", optional=False).optional is False


# -- Tests: setup_submodules --------------------------------------------------


class TestOptionalSubmoduleSetup:
    def test_skipped_on_missing_layers(self):
        blocks = _setup_blocks(HybridModel(), MinimalAdapter(optional=True))

        for i in range(3):
            assert "foo" in blocks[i].real_components
            assert hasattr(blocks[i], "foo")

        assert "foo" not in blocks[3].real_components
        assert "foo" not in blocks[3]._modules
        assert "foo" not in blocks[3].submodules

        for i in range(4):
            assert "bar" in blocks[i].real_components

    def test_non_optional_raises(self):
        model = HybridModel()
        adapter = MinimalAdapter(optional=False)
        block = copy.deepcopy(adapter.make_block_template())
        block.name = "layers.3"
        block.set_original_component(model.layers[3])
        with pytest.raises(AttributeError):
            setup_submodules(block, adapter, model.layers[3])


# -- Tests: blocks_with() -----------------------------------------------------


class TestBlocksWith:
    def test_returns_matching_blocks(self):
        bridge = _make_hybrid_bridge()
        assert [idx for idx, _ in bridge.blocks_with("foo")] == [0, 1, 2]
        assert len(bridge.blocks_with("bar")) == 4
        assert bridge.blocks_with("nonexistent") == []

    def test_no_blocks_attribute(self):
        from transformer_lens.model_bridge.bridge import TransformerBridge

        bridge = TransformerBridge.__new__(TransformerBridge)
        nn.Module.__init__(bridge)
        assert bridge.blocks_with("attn") == []

    def test_checks_modules_not_hasattr(self):
        bridge = _make_hybrid_bridge()
        assert len(bridge.blocks_with("training")) == 0


# -- Tests: _stack_block_params -----------------------------------------------


class TestStackBlockParams:
    def test_logs_warning_and_returns_subset(self, caplog):
        bridge = _make_hybrid_bridge()
        with caplog.at_level(logging.WARNING):
            result = bridge._stack_block_params("foo.proj.weight")
        assert any("Hybrid model" in msg for msg in caplog.messages)
        assert result.shape[0] == 3

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            bridge._stack_block_params("foo.proj.weight")
        assert any("Hybrid model" in msg for msg in caplog.messages)

    def test_raises_when_no_blocks_match(self):
        bridge = _make_hybrid_bridge()
        with pytest.raises(AttributeError, match="No blocks have"):
            bridge._stack_block_params("nonexistent")

    def test_succeeds_on_universal_submodule(self):
        bridge = _make_hybrid_bridge()
        result = bridge._stack_block_params("bar.weight")
        assert result.shape[0] == 4


# -- Tests: refactor_factored_attn_matrices ------------------------------------


class TestRefactorFactoredAttnHybrid:
    def test_skips_missing_attn_layers(self):
        from transformer_lens.config.TransformerLensConfig import TransformerLensConfig
        from transformer_lens.weight_processing import ProcessWeights

        cfg = TransformerLensConfig(
            n_layers=4,
            n_heads=2,
            d_head=4,
            d_model=8,
            n_ctx=16,
            positional_embedding_type="standard",
        )
        state_dict = {}
        for l in range(3):
            state_dict[f"blocks.{l}.attn.W_Q"] = torch.randn(2, 8, 4)
            state_dict[f"blocks.{l}.attn.W_K"] = torch.randn(2, 8, 4)
            state_dict[f"blocks.{l}.attn.W_V"] = torch.randn(2, 8, 4)
            state_dict[f"blocks.{l}.attn.W_O"] = torch.randn(2, 4, 8)
            state_dict[f"blocks.{l}.attn.b_Q"] = torch.randn(2, 4)
            state_dict[f"blocks.{l}.attn.b_K"] = torch.randn(2, 4)
            state_dict[f"blocks.{l}.attn.b_V"] = torch.randn(2, 4)
            state_dict[f"blocks.{l}.attn.b_O"] = torch.randn(8)

        result = ProcessWeights.refactor_factored_attn_matrices(state_dict, cfg)

        for l in range(3):
            assert f"blocks.{l}.attn.W_Q" in result
        assert "blocks.3.attn.W_Q" not in result

    def test_raises_on_partial_attn_keys(self):
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
        state_dict = {"blocks.0.attn.W_Q": torch.randn(2, 8, 4)}
        with pytest.raises(ValueError, match="Inconsistent attention weights"):
            ProcessWeights.refactor_factored_attn_matrices(state_dict, cfg)


# -- Tests: weight distribution ------------------------------------------------


class TestWeightDistributionRagged:
    def test_distribute_weights_skips_empty_blocks(self):
        from transformer_lens.weight_processing import ProcessWeights

        blocks = _setup_blocks(HybridModel(), MinimalAdapter(optional=True))
        state_dict = {}
        for i in range(3):
            state_dict[f"blocks.{i}.foo.weight"] = torch.randn(4, 4)
        for i in range(4):
            state_dict[f"blocks.{i}.bar.weight"] = torch.randn(4, 4)

        ProcessWeights.distribute_weights_to_components(
            state_dict=state_dict,
            component_mapping={"blocks": ("layers", blocks)},
        )


# -- Tests: __setattr__ whitelist ----------------------------------------------


class TestSetAttrWhitelist:
    def test_optional_stays_on_bridge(self):
        comp = LinearBridge(name="test")
        fake_hf = nn.Linear(4, 4, bias=False)
        comp.set_original_component(fake_hf)
        comp.optional = True
        assert comp.optional is True
        assert not hasattr(fake_hf, "optional")


# -- Tests: accumulated_bias --------------------------------------------------


class TestAccumulatedBiasHybrid:
    def test_skips_non_attn_layers(self):
        bridge = _make_hybrid_bridge()
        result = bridge.accumulated_bias(layer=4)
        assert result.shape == (4,)

    def test_mlp_input_on_non_attn_layer(self):
        bridge = _make_hybrid_bridge()
        result = bridge.accumulated_bias(layer=3, mlp_input=True)
        assert result.shape == (4,)


# -- Tests: block introspection ------------------------------------------------


class TestBlockIntrospection:
    def test_block_submodules(self):
        bridge = _make_hybrid_bridge()
        assert "foo" in bridge.block_submodules(0)
        assert "bar" in bridge.block_submodules(0)
        assert "foo" not in bridge.block_submodules(3)
        assert "bar" in bridge.block_submodules(3)

    def test_layer_types(self):
        bridge = _make_hybrid_bridge()
        types = bridge.layer_types()
        assert len(types) == 4
        for i in range(3):
            assert "foo" in types[i]
        assert "foo" not in types[3]


# -- Tests: stack_params_for --------------------------------------------------


class TestStackParamsFor:
    def test_returns_correct_indices_and_tensors(self):
        bridge = _make_hybrid_bridge()
        indices, stacked = bridge.stack_params_for("foo", "foo.proj.weight")
        assert indices == [0, 1, 2]
        assert stacked.shape[0] == 3

    def test_raises_on_no_matching_blocks(self):
        bridge = _make_hybrid_bridge()
        with pytest.raises(ValueError, match="No blocks have submodule"):
            bridge.stack_params_for("nonexistent", "nonexistent.weight")


# -- Tests: attn_head_labels --------------------------------------------------


class TestAttnHeadLabels:
    def test_excludes_non_attn_layers(self):
        bridge = _make_hybrid_bridge_with_attn()
        labels = bridge.attn_head_labels
        assert len(labels) == 6
        assert labels == ["L0H0", "L0H1", "L1H0", "L1H1", "L2H0", "L2H1"]

    def test_all_head_labels_includes_all(self):
        bridge = _make_hybrid_bridge_with_attn()
        assert len(bridge.all_head_labels) == 8


# -- Tests: hook propagation --------------------------------------------------


class TestHookPropagation:
    def test_hooks_fire_on_present_optional(self):
        blocks = _setup_blocks(HybridModel(), MinimalAdapter(optional=True))
        fired = []
        blocks[0].foo.hook_out.add_hook(lambda t, hook: fired.append(True) or t)

        blocks[0].foo(torch.randn(1, 4))
        assert len(fired) == 1

    def test_absent_optional_has_no_module(self):
        blocks = _setup_blocks(HybridModel(), MinimalAdapter(optional=True))
        assert "foo" not in blocks[3]._modules

    def test_hooks_fire_only_on_present(self):
        model = HybridModel()
        blocks = _setup_blocks(model, MinimalAdapter(optional=True))
        fired_indices = []
        for i, block in enumerate(blocks):
            if "foo" in block._modules:
                block.foo.hook_out.add_hook(lambda t, hook, idx=i: fired_indices.append(idx) or t)

        x = torch.randn(1, 4)
        for layer in model.layers:
            x = layer(x)
        assert fired_indices == [0, 1, 2]

    def test_universal_hooks_fire_on_all(self):
        model = HybridModel()
        blocks = _setup_blocks(model, MinimalAdapter(optional=True))
        fired_indices = []
        for i, block in enumerate(blocks):
            block.bar.hook_out.add_hook(lambda t, hook, idx=i: fired_indices.append(idx) or t)

        x = torch.randn(1, 4)
        for layer in model.layers:
            x = layer(x)
        assert fired_indices == [0, 1, 2, 3]


# -- Tests: CompositionScores tensor protocol ----------------------------------


class TestCompositionScoresProtocol:
    def _make_scores(self):
        from transformer_lens.model_bridge.composition_scores import CompositionScores

        t = torch.randn(3, 2, 3, 2)
        return CompositionScores(t, [0, 2, 5], ["L0H0", "L0H1", "L2H0", "L2H1", "L5H0", "L5H1"])

    def test_shape_device_dtype(self):
        cs = self._make_scores()
        assert cs.shape == torch.Size([3, 2, 3, 2])
        assert cs.device == torch.device("cpu")
        assert cs.dtype == torch.float32

    def test_indexing(self):
        cs = self._make_scores()
        assert isinstance(cs[0, :, 1, :], torch.Tensor)
        assert cs[0, :, 1, :].shape == (2, 2)

    def test_torch_isnan(self):
        cs = self._make_scores()
        result = torch.isnan(cs)
        assert isinstance(result, torch.Tensor)
        assert not result.any()

    def test_torch_where(self):
        cs = self._make_scores()
        result = torch.where(cs > 0, cs.scores, torch.zeros_like(cs.scores))
        assert isinstance(result, torch.Tensor)

    def test_comparisons(self):
        cs = self._make_scores()
        assert isinstance(cs > 0, torch.Tensor)
        assert isinstance(cs != 0, torch.Tensor)
        assert isinstance(cs == 0, torch.Tensor)

    def test_tensor_methods(self):
        cs = self._make_scores()
        assert isinstance(cs.abs(), torch.Tensor)
        assert isinstance(cs.sum(), torch.Tensor)
        assert isinstance(cs.any(), torch.Tensor)

    def test_chained_indexing_and_method(self):
        cs = self._make_scores()
        result = cs[0, :, 1, :].abs().sum()
        assert result.ndim == 0

    def test_metadata(self):
        cs = self._make_scores()
        assert cs.layer_indices == [0, 2, 5]
        assert len(cs.head_labels) == 6
        assert "CompositionScores" in repr(cs)


# -- Tests: get_bridge_params with hybrid blocks ------------------------------


class TestGetBridgeParamsHybrid:
    def test_no_attn_keys_for_non_attn_layers(self):
        from transformer_lens.model_bridge.get_params_util import get_bridge_params

        bridge = _make_hybrid_bridge_with_attn()
        bridge.cfg.d_vocab = 10
        bridge.cfg.n_ctx = 8
        bridge.cfg.d_mlp = 16
        bridge.cfg.d_head = 2

        bridge.embed = nn.Embedding(10, 4)
        bridge.pos_embed = type("PE", (), {"weight": torch.randn(8, 4)})()
        bridge.unembed = type("UE", (), {"weight": torch.randn(10, 4), "b_U": torch.zeros(10)})()

        params = get_bridge_params(bridge)
        attn_keys_block3 = [k for k in params if k.startswith("blocks.3.attn.")]
        assert (
            len(attn_keys_block3) == 0
        ), f"Non-attn layer should have no attn keys: {attn_keys_block3}"
