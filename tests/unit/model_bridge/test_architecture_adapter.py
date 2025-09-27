"""Tests for the architecture adapter."""

import pytest
import torch
import torch.nn as nn

from tests.mocks.architecture_adapter import (
    MockArchitectureAdapter,
    mock_adapter,
    mock_model_adapter,
)
from tests.mocks.models import MockGemma3Model
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.gemma3 import (
    Gemma3ArchitectureAdapter,
)


def test_get_remote_component_with_mock(
    mock_adapter: MockArchitectureAdapter, mock_model_adapter: nn.Module
):
    """Test get_remote_component with the mock adapter."""
    # Test direct mapping
    ln_final = mock_adapter.get_component(mock_model_adapter, "ln_final")
    assert isinstance(ln_final, nn.LayerNorm)

    # Test block mapping
    block = mock_adapter.get_component(mock_model_adapter, "blocks.0")
    assert isinstance(block, nn.Module)

    # Test block subcomponent mapping
    ln1 = mock_adapter.get_component(mock_model_adapter, "blocks.0.ln1")
    assert isinstance(ln1, nn.LayerNorm)

    attn = mock_adapter.get_component(mock_model_adapter, "blocks.0.attn")
    assert isinstance(attn, nn.Module)

    mlp = mock_adapter.get_component(mock_model_adapter, "blocks.0.mlp")
    assert isinstance(mlp, nn.Module)


@pytest.fixture
def cfg():
    return TransformerBridgeConfig(
        d_model=128,
        d_head=16,  # 128 / 8 heads
        n_layers=2,
        n_ctx=1024,
        n_heads=8,
        d_vocab=1000,
        d_mlp=512,
        n_key_value_heads=8,
        default_prepend_bos=True,
        architecture="Gemma3ForCausalLM",  # Test architecture
    )


@pytest.fixture
def adapter(cfg) -> Gemma3ArchitectureAdapter:
    """Create a Gemma3 adapter."""
    return Gemma3ArchitectureAdapter(cfg)


@pytest.fixture
def model() -> MockGemma3Model:
    """Create a mock Gemma 3 model."""
    return MockGemma3Model()


def test_translate_transformer_lens_path(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test path translation from TransformerLens to Remote paths."""
    # Test direct mapping
    assert adapter.translate_transformer_lens_path("embed") == "model.embed_tokens"
    assert adapter.translate_transformer_lens_path("ln_final") == "model.norm"
    assert adapter.translate_transformer_lens_path("unembed") == "lm_head"

    # Test block mapping
    assert adapter.translate_transformer_lens_path("blocks") == "model.layers"
    assert adapter.translate_transformer_lens_path("blocks.0") == "model.layers.0"
    assert adapter.translate_transformer_lens_path("blocks.1") == "model.layers.1"

    # Test block subcomponent mapping
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1") == "model.layers.0.input_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1_post")
        == "model.layers.0.post_attention_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2")
        == "model.layers.0.pre_feedforward_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2_post")
        == "model.layers.0.post_feedforward_layernorm"
    )
    assert adapter.translate_transformer_lens_path("blocks.0.attn") == "model.layers.0.self_attn"
    assert adapter.translate_transformer_lens_path("blocks.0.mlp") == "model.layers.0.mlp"

    # Test deeper subcomponent paths
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.q")
        == "model.layers.0.self_attn.q_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.k")
        == "model.layers.0.self_attn.k_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.v")
        == "model.layers.0.self_attn.v_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.o")
        == "model.layers.0.self_attn.o_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.gate")
        == "model.layers.0.mlp.gate_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.in") == "model.layers.0.mlp.up_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.out")
        == "model.layers.0.mlp.down_proj"
    )


def test_translate_transformer_lens_path_last_component(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test path translation with last_component_only=True."""
    # Test direct mapping
    assert (
        adapter.translate_transformer_lens_path("embed", last_component_only=True) == "embed_tokens"
    )
    assert adapter.translate_transformer_lens_path("ln_final", last_component_only=True) == "norm"
    assert adapter.translate_transformer_lens_path("unembed", last_component_only=True) == "lm_head"

    # Test block mapping
    assert adapter.translate_transformer_lens_path("blocks", last_component_only=True) == "layers"
    assert adapter.translate_transformer_lens_path("blocks.0", last_component_only=True) == "0"
    assert adapter.translate_transformer_lens_path("blocks.1", last_component_only=True) == "1"

    # Test block subcomponent mapping
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1", last_component_only=True)
        == "input_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln1_post", last_component_only=True)
        == "post_attention_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2", last_component_only=True)
        == "pre_feedforward_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.ln2_post", last_component_only=True)
        == "post_feedforward_layernorm"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn", last_component_only=True)
        == "self_attn"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp", last_component_only=True) == "mlp"
    )

    # Test deeper subcomponent paths with last_component_only
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.q", last_component_only=True)
        == "q_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.k", last_component_only=True)
        == "k_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.v", last_component_only=True)
        == "v_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.attn.o", last_component_only=True)
        == "o_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.gate", last_component_only=True)
        == "gate_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.in", last_component_only=True)
        == "up_proj"
    )
    assert (
        adapter.translate_transformer_lens_path("blocks.0.mlp.out", last_component_only=True)
        == "down_proj"
    )


def test_component_mapping_structure(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test that the component mapping has the expected structure."""
    mapping = adapter.get_component_mapping()

    # Test that we have the expected top-level components
    assert "embed" in mapping
    assert "blocks" in mapping
    assert "ln_final" in mapping
    assert "unembed" in mapping

    # Test that components are bridge instances
    from transformer_lens.model_bridge.generalized_components import (
        AttentionBridge,
        BlockBridge,
        EmbeddingBridge,
        LinearBridge,
        MLPBridge,
        NormalizationBridge,
        UnembeddingBridge,
    )

    assert isinstance(mapping["embed"], EmbeddingBridge)
    assert isinstance(mapping["blocks"], BlockBridge)
    assert isinstance(mapping["ln_final"], NormalizationBridge)
    assert isinstance(mapping["unembed"], UnembeddingBridge)

    # Test that blocks has submodules
    blocks_bridge = mapping["blocks"]
    assert hasattr(blocks_bridge, "submodules")
    assert "ln1" in blocks_bridge.submodules
    assert "ln2" in blocks_bridge.submodules
    assert "attn" in blocks_bridge.submodules
    assert "mlp" in blocks_bridge.submodules

    # Test that the submodules are the expected types
    assert isinstance(blocks_bridge.submodules["ln1"], NormalizationBridge)
    assert isinstance(blocks_bridge.submodules["ln2"], NormalizationBridge)
    assert isinstance(blocks_bridge.submodules["attn"], AttentionBridge)
    assert isinstance(blocks_bridge.submodules["mlp"], MLPBridge)

    # Test that attention has submodules
    attn_bridge = blocks_bridge.submodules["attn"]
    assert hasattr(attn_bridge, "submodules")
    assert "q" in attn_bridge.submodules
    assert "k" in attn_bridge.submodules
    assert "v" in attn_bridge.submodules
    assert "o" in attn_bridge.submodules
    assert isinstance(attn_bridge.submodules["q"], LinearBridge)
    assert isinstance(attn_bridge.submodules["k"], LinearBridge)
    assert isinstance(attn_bridge.submodules["v"], LinearBridge)
    assert isinstance(attn_bridge.submodules["o"], LinearBridge)

    # Test that MLP has submodules
    mlp_bridge = blocks_bridge.submodules["mlp"]
    assert hasattr(mlp_bridge, "submodules")
    assert "gate" in mlp_bridge.submodules
    assert "in" in mlp_bridge.submodules
    assert "out" in mlp_bridge.submodules
    assert isinstance(mlp_bridge.submodules["gate"], LinearBridge)
    assert isinstance(mlp_bridge.submodules["in"], LinearBridge)
    assert isinstance(mlp_bridge.submodules["out"], LinearBridge)


def test_get_component(adapter: Gemma3ArchitectureAdapter, model: MockGemma3Model) -> None:
    """Test getting components from the model."""
    # Test direct mapping
    assert isinstance(adapter.get_component(model, "embed"), nn.Embedding)

    # Test block mapping
    block = adapter.get_component(model, "blocks.0")
    assert isinstance(block, nn.Module)

    # Test block subcomponent mapping
    ln1 = adapter.get_component(model, "blocks.0.ln1")
    assert isinstance(ln1, nn.LayerNorm)

    attn = adapter.get_component(model, "blocks.0.attn")
    assert isinstance(attn, nn.Module)

    mlp = adapter.get_component(model, "blocks.0.mlp")
    assert isinstance(mlp, nn.Module)


def test_invalid_paths(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test handling of invalid paths."""
    with pytest.raises(ValueError, match="Component not_found not found in component mapping"):
        adapter.translate_transformer_lens_path("not_found")

    with pytest.raises(ValueError, match="Expected item index, got invalid"):
        adapter.translate_transformer_lens_path("blocks.invalid")

    with pytest.raises(ValueError, match="Component not_found not found in blocks components"):
        adapter.translate_transformer_lens_path("blocks.0.not_found")


def test_get_component_invalid_paths(
    adapter: Gemma3ArchitectureAdapter, model: MockGemma3Model
) -> None:
    """Test handling of invalid paths in get_component."""
    with pytest.raises(ValueError, match="Component not_found not found in component mapping"):
        adapter.get_component(model, "not_found")

    with pytest.raises(ValueError, match="Expected item index, got invalid"):
        adapter.get_component(model, "blocks.invalid")


def test_translate_weight_processing_paths(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test translation of paths used in weight processing functions.

    This ensures that all paths used in ProcessWeights methods can be properly
    translated by the adapter.
    """
    # Test embedding paths (Gemma3 doesn't have positional embeddings)
    assert adapter.translate_transformer_lens_path("embed.W_E") == "model.embed_tokens.weight"

    # Test unembedding paths
    assert adapter.translate_transformer_lens_path("unembed.W_U") == "lm_head.weight"
    assert adapter.translate_transformer_lens_path("unembed.b_U") == "lm_head.bias"

    # Test layer norm paths
    assert adapter.translate_transformer_lens_path("ln_final.w") == "model.norm.weight"
    assert adapter.translate_transformer_lens_path("ln_final.b") == "model.norm.bias"

    # Test attention weight and bias paths for multiple layers
    for layer in [0, 1]:
        # Attention weights
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_Q")
            == f"model.layers.{layer}.self_attn.q_proj.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_K")
            == f"model.layers.{layer}.self_attn.k_proj.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_V")
            == f"model.layers.{layer}.self_attn.v_proj.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_O")
            == f"model.layers.{layer}.self_attn.o_proj.weight"
        )

        # Attention biases
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_Q")
            == f"model.layers.{layer}.self_attn.q_proj.bias"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_K")
            == f"model.layers.{layer}.self_attn.k_proj.bias"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_V")
            == f"model.layers.{layer}.self_attn.v_proj.bias"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_O")
            == f"model.layers.{layer}.self_attn.o_proj.bias"
        )

        # MLP weights
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.mlp.W_in")
            == f"model.layers.{layer}.mlp.up_proj.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.mlp.W_out")
            == f"model.layers.{layer}.mlp.down_proj.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.mlp.W_gate")
            == f"model.layers.{layer}.mlp.gate_proj.weight"
        )

        # MLP biases
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.mlp.b_in")
            == f"model.layers.{layer}.mlp.up_proj.bias"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.mlp.b_out")
            == f"model.layers.{layer}.mlp.down_proj.bias"
        )

        # Layer norm paths within blocks
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.ln1.w")
            == f"model.layers.{layer}.input_layernorm.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.ln1.b")
            == f"model.layers.{layer}.input_layernorm.bias"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.ln2.w")
            == f"model.layers.{layer}.pre_feedforward_layernorm.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.ln2.b")
            == f"model.layers.{layer}.pre_feedforward_layernorm.bias"
        )


def test_translate_weight_processing_paths_gqa(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test translation of GQA (Grouped Query Attention) specific paths used in weight processing."""
    # Test GQA-specific paths (with underscore prefix for grouped keys/values)
    for layer in [0, 1]:
        # GQA paths use underscore prefix for grouped K/V
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn._b_K")
            == f"model.layers.{layer}.self_attn.k_proj.bias"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn._W_K")
            == f"model.layers.{layer}.self_attn.k_proj.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn._b_V")
            == f"model.layers.{layer}.self_attn.v_proj.bias"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.attn._W_V")
            == f"model.layers.{layer}.self_attn.v_proj.weight"
        )


def test_translate_weight_processing_solu_paths(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test translation of SoLU-specific paths used in weight processing."""
    # Test SoLU MLP layer norm paths (used in some older models)
    for layer in [0, 1]:
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.mlp.ln.w")
            == f"model.layers.{layer}.mlp.ln.weight"
        )
        assert (
            adapter.translate_transformer_lens_path(f"blocks.{layer}.mlp.ln.b")
            == f"model.layers.{layer}.mlp.ln.bias"
        )


def test_get_generalized_component_top_level(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test getting top-level components."""
    # Test top-level components
    embed_component = adapter.get_generalized_component("embed")
    assert isinstance(embed_component, EmbeddingBridge)
    assert embed_component.name == "model.embed_tokens"

    blocks_component = adapter.get_generalized_component("blocks")
    assert isinstance(blocks_component, BlockBridge)
    assert blocks_component.name == "model.layers"

    ln_final_component = adapter.get_generalized_component("ln_final")
    assert isinstance(ln_final_component, NormalizationBridge)
    assert ln_final_component.name == "model.norm"

    unembed_component = adapter.get_generalized_component("unembed")
    assert isinstance(unembed_component, UnembeddingBridge)
    assert unembed_component.name == "lm_head"


def test_get_generalized_component_with_parameters(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test getting components with parameter suffixes."""
    # Test that parameter suffixes are stripped correctly
    embed_component = adapter.get_generalized_component("embed.W_E")
    assert isinstance(embed_component, EmbeddingBridge)
    assert embed_component.name == "model.embed_tokens"

    embed_component = adapter.get_generalized_component("embed.b_E")
    assert isinstance(embed_component, EmbeddingBridge)
    assert embed_component.name == "model.embed_tokens"

    ln_final_component = adapter.get_generalized_component("ln_final.w")
    assert isinstance(ln_final_component, NormalizationBridge)
    assert ln_final_component.name == "model.norm"

    ln_final_component = adapter.get_generalized_component("ln_final.b")
    assert isinstance(ln_final_component, NormalizationBridge)
    assert ln_final_component.name == "model.norm"


def test_get_generalized_component_nested_with_indexing(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test getting nested components with layer indexing."""
    # Test nested components with layer indexing
    # Note: For list items, we return the bridge component itself
    # since indexing is handled at the model level
    blocks_component = adapter.get_generalized_component("blocks.0")
    assert isinstance(blocks_component, BlockBridge)
    assert blocks_component.name == "model.layers"

    # Test deeper nesting
    ln1_component = adapter.get_generalized_component("blocks.0.ln1")
    assert isinstance(ln1_component, NormalizationBridge)
    assert ln1_component.name == "input_layernorm"

    ln2_component = adapter.get_generalized_component("blocks.0.ln2")
    assert isinstance(ln2_component, NormalizationBridge)
    assert ln2_component.name == "pre_feedforward_layernorm"

    attn_component = adapter.get_generalized_component("blocks.0.attn")
    assert isinstance(attn_component, AttentionBridge)
    assert attn_component.name == "self_attn"

    mlp_component = adapter.get_generalized_component("blocks.0.mlp")
    assert isinstance(mlp_component, MLPBridge)
    assert mlp_component.name == "mlp"


def test_get_generalized_component_nested_with_parameters(
    adapter: Gemma3ArchitectureAdapter,
) -> None:
    """Test getting nested components with parameter suffixes."""
    # Test nested components with parameter suffixes
    ln1_component = adapter.get_generalized_component("blocks.0.ln1.w")
    assert isinstance(ln1_component, NormalizationBridge)
    assert ln1_component.name == "input_layernorm"

    ln1_component = adapter.get_generalized_component("blocks.0.ln1.b")
    assert isinstance(ln1_component, NormalizationBridge)
    assert ln1_component.name == "input_layernorm"

    # For attention parameters, the method should return the specific subcomponent
    # since Gemma3 has separate Q, K, V components
    q_component = adapter.get_generalized_component("blocks.0.attn.W_Q")
    assert isinstance(q_component, LinearBridge)
    assert q_component.name == "q_proj"

    q_component = adapter.get_generalized_component("blocks.0.attn.b_Q")
    assert isinstance(q_component, LinearBridge)
    assert q_component.name == "q_proj"

    # For MLP parameters, the method should return the specific subcomponent
    # since Gemma3 has separate in, out, gate components
    in_component = adapter.get_generalized_component("blocks.0.mlp.W_in")
    assert isinstance(in_component, LinearBridge)
    assert in_component.name == "up_proj"

    in_component = adapter.get_generalized_component("blocks.0.mlp.b_in")
    assert isinstance(in_component, LinearBridge)
    assert in_component.name == "up_proj"


def test_get_generalized_component_deeply_nested(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test getting deeply nested components."""
    # Test deeply nested components
    q_component = adapter.get_generalized_component("blocks.0.attn.q")
    assert isinstance(q_component, LinearBridge)
    assert q_component.name == "q_proj"

    k_component = adapter.get_generalized_component("blocks.0.attn.k")
    assert isinstance(k_component, LinearBridge)
    assert k_component.name == "k_proj"

    v_component = adapter.get_generalized_component("blocks.0.attn.v")
    assert isinstance(v_component, LinearBridge)
    assert v_component.name == "v_proj"

    o_component = adapter.get_generalized_component("blocks.0.attn.o")
    assert isinstance(o_component, LinearBridge)
    assert o_component.name == "o_proj"

    gate_component = adapter.get_generalized_component("blocks.0.mlp.gate")
    assert isinstance(gate_component, LinearBridge)
    assert gate_component.name == "gate_proj"

    in_component = adapter.get_generalized_component("blocks.0.mlp.in")
    assert isinstance(in_component, LinearBridge)
    assert in_component.name == "up_proj"

    out_component = adapter.get_generalized_component("blocks.0.mlp.out")
    assert isinstance(out_component, LinearBridge)
    assert out_component.name == "down_proj"


def test_get_generalized_component_deeply_nested_with_parameters(
    adapter: Gemma3ArchitectureAdapter,
) -> None:
    """Test getting deeply nested components with parameter suffixes."""
    # Test deeply nested components with parameter suffixes
    # Note: The parameter suffixes (.weight, .bias) are stripped by _preprocess_parameter_path
    # so these should return the same components as without the suffixes
    # However, the _preprocess_parameter_path method only handles TransformerLens-specific parameter names
    # like W_Q, b_Q, etc., not generic PyTorch parameter names like .weight, .bias

    # Test with TransformerLens parameter names (these should work)
    q_component = adapter.get_generalized_component("blocks.0.attn.W_Q")
    assert isinstance(q_component, LinearBridge)
    assert q_component.name == "q_proj"

    q_component = adapter.get_generalized_component("blocks.0.attn.b_Q")
    assert isinstance(q_component, LinearBridge)
    assert q_component.name == "q_proj"

    gate_component = adapter.get_generalized_component("blocks.0.mlp.W_gate")
    assert isinstance(gate_component, LinearBridge)
    assert gate_component.name == "gate_proj"

    in_component = adapter.get_generalized_component("blocks.0.mlp.W_in")
    assert isinstance(in_component, LinearBridge)
    assert in_component.name == "up_proj"


def test_get_generalized_component_error_cases(adapter: Gemma3ArchitectureAdapter) -> None:
    """Test error cases for get_generalized_component."""
    # Test empty path - this will be processed by _preprocess_parameter_path and result in empty component mapping lookup
    with pytest.raises(ValueError, match="Component  not found in component mapping"):
        adapter.get_generalized_component("")

    # Test non-existent top-level component
    with pytest.raises(ValueError, match="Component nonexistent not found in component mapping"):
        adapter.get_generalized_component("nonexistent")

    # Test non-existent nested component
    with pytest.raises(ValueError, match="Component nonexistent not found in blocks.0 components"):
        adapter.get_generalized_component("blocks.0.nonexistent")

    # Test non-existent deeply nested component
    with pytest.raises(
        ValueError, match="Component nonexistent not found in blocks.0.attn components"
    ):
        adapter.get_generalized_component("blocks.0.attn.nonexistent")


def test_get_generalized_component_no_component_mapping() -> None:
    """Test error when component_mapping is None."""
    cfg = TransformerBridgeConfig(
        d_model=128,
        d_head=16,
        n_layers=2,
        n_ctx=1024,
        n_heads=8,
        d_vocab=1000,
        d_mlp=512,
        n_key_value_heads=8,
        default_prepend_bos=True,
        architecture="Gemma3ForCausalLM",
    )
    adapter = Gemma3ArchitectureAdapter(cfg)
    adapter.component_mapping = None

    with pytest.raises(
        ValueError, match="component_mapping must be set before calling get_generalized_component"
    ):
        adapter.get_generalized_component("embed")


def test_get_generalized_component_various_layer_indices(
    adapter: Gemma3ArchitectureAdapter,
) -> None:
    """Test getting components for various layer indices."""
    # Test different layer indices
    for layer_idx in [0, 1]:
        ln1_component = adapter.get_generalized_component(f"blocks.{layer_idx}.ln1")
        assert isinstance(ln1_component, NormalizationBridge)
        assert ln1_component.name == "input_layernorm"

        attn_component = adapter.get_generalized_component(f"blocks.{layer_idx}.attn")
        assert isinstance(attn_component, AttentionBridge)
        assert attn_component.name == "self_attn"

        mlp_component = adapter.get_generalized_component(f"blocks.{layer_idx}.mlp")
        assert isinstance(mlp_component, MLPBridge)
        assert mlp_component.name == "mlp"


def test_hf_key_filled_into_original_components():
    """Test that HuggingFace keys are properly converted and filled into _original_component structure."""
    from transformer_lens.model_bridge import TransformerBridge

    # Load a small model for testing
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

    # Get the original state dict to see the _original_component structure
    original_state_dict = bridge.original_model.state_dict()

    # Test cases: HF key -> expected _original_component key
    test_cases = [
        (
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0._original_component.attn._original_component.c_attn._original_component.weight",
        ),
        (
            "transformer.h.0.attn.c_attn.bias",
            "transformer.h.0._original_component.attn._original_component.c_attn._original_component.bias",
        ),
        (
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.h.0._original_component.mlp._original_component.c_fc._original_component.weight",
        ),
        (
            "transformer.h.0.mlp.c_fc.bias",
            "transformer.h.0._original_component.mlp._original_component.c_fc._original_component.bias",
        ),
        ("transformer.wte.weight", "transformer.wte._original_component.weight"),
        ("transformer.wpe.weight", "transformer.wpe._original_component.weight"),
        ("lm_head.weight", "lm_head._original_component.weight"),
    ]

    for hf_key, expected_bridge_key in test_cases:
        # Convert the HF key to bridge key using the adapter
        converted_key = bridge.adapter.convert_hf_key_to_bridge_key(hf_key)

        # Assert the conversion is correct
        assert (
            converted_key == expected_bridge_key
        ), f"Key conversion failed for {hf_key}: got {converted_key}, expected {expected_bridge_key}"

        # Assert the converted key exists in the original model's state dict
        assert (
            converted_key in original_state_dict
        ), f"Converted key {converted_key} not found in original model state dict"

        # Test that we can actually load a value using this key
        # Create a test state dict with the HF key
        test_value = torch.randn_like(original_state_dict[converted_key])
        test_state_dict = {hf_key: test_value}

        # Load it using the bridge's load_state_dict (which should convert the key)
        result = bridge.load_state_dict(test_state_dict, strict=False, assign=False)

        # Assert no unexpected keys (meaning the conversion worked)
        assert (
            len(result.unexpected_keys) == 0
        ), f"Unexpected keys found when loading {hf_key}: {result.unexpected_keys}"

        # Verify the value was actually loaded by checking if it changed
        # (We'll compare with a small tolerance due to floating point precision)
        loaded_value = bridge.original_model.state_dict()[converted_key]
        assert torch.allclose(
            loaded_value, test_value, atol=1e-6
        ), f"Value not properly loaded for key {hf_key}"

    print(
        "✅ All HuggingFace keys properly converted and filled into _original_component structure!"
    )


def test_correct_key_mapping():
    """Test correct key mapping for processed weights."""
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.weight_processing import ProcessWeights

    model_name = "gpt2"
    device = "cpu"

    # Load TransformerBridge
    bridge = TransformerBridge.boot_transformers(model_name, device=device)

    # Get original weights
    original_state_dict = bridge._extract_hf_weights()

    # Process weights
    processed_state_dict = ProcessWeights.process_weights(
        original_state_dict,
        bridge.cfg,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
        adapter=bridge.adapter,
    )

    # Get bridge keys
    bridge_keys = list(bridge.original_model.state_dict().keys())

    # Test specific key mapping
    layer = 0
    hf_key = f"transformer.h.{layer}.attn.c_attn.weight"
    expected_bridge_key = f"transformer.h.{layer}._original_component.attn._original_component.c_attn._original_component.weight"

    assert (
        hf_key in processed_state_dict
    ), f"HuggingFace key {hf_key} should exist in processed state dict"
    assert (
        expected_bridge_key in bridge_keys
    ), f"Bridge key {expected_bridge_key} should exist in bridge state dict"

    # Test key conversion using adapter
    converted_key = bridge.adapter.convert_hf_key_to_bridge_key(hf_key)
    assert (
        converted_key == expected_bridge_key
    ), f"Key conversion failed: got {converted_key}, expected {expected_bridge_key}"

    # Test that we can load the processed weight
    processed_weight = processed_state_dict[hf_key]
    bridge_weight = bridge.original_model.state_dict()[expected_bridge_key]

    assert (
        processed_weight.shape == bridge_weight.shape
    ), f"Shape mismatch: {processed_weight.shape} vs {bridge_weight.shape}"

    # Test loading the weight
    mapped_state_dict = {expected_bridge_key: processed_weight}
    result = bridge.load_state_dict(mapped_state_dict, strict=False, assign=False)

    assert len(result.unexpected_keys) == 0, f"Unexpected keys: {result.unexpected_keys}"

    # Verify the weight was loaded
    loaded_weight = bridge.original_model.state_dict()[expected_bridge_key]
    assert torch.allclose(
        loaded_weight, processed_weight, atol=1e-6
    ), "Weight was not loaded correctly"


def test_adapter_key_mapping_comprehensive():
    """Test comprehensive key mapping using architecture adapter."""
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.weight_processing import ProcessWeights

    model_name = "gpt2"
    device = "cpu"

    # Load TransformerBridge
    bridge = TransformerBridge.boot_transformers(model_name, device=device)

    # Get original weights
    original_state_dict = bridge._extract_hf_weights()

    # Process weights
    processed_state_dict = ProcessWeights.process_weights(
        original_state_dict,
        bridge.cfg,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
        adapter=bridge.adapter,
    )

    # Get bridge keys
    bridge_keys = list(bridge.original_model.state_dict().keys())

    # Test key conversion for all processed keys
    successful_conversions = 0
    total_conversions = 0

    for processed_key, value in processed_state_dict.items():
        total_conversions += 1

        # Convert HuggingFace key to bridge key using adapter
        bridge_key = bridge.adapter.convert_hf_key_to_bridge_key(processed_key)

        if bridge_key in bridge_keys:
            successful_conversions += 1

            # Test that we can load this weight
            mapped_state_dict = {bridge_key: value}
            result = bridge.load_state_dict(mapped_state_dict, strict=False, assign=False)

            # Should have no unexpected keys
            assert (
                len(result.unexpected_keys) == 0
            ), f"Unexpected keys for {processed_key}: {result.unexpected_keys}"

    conversion_rate = successful_conversions / total_conversions
    assert (
        conversion_rate >= 0.8
    ), f"Key conversion rate too low: {conversion_rate*100:.1f}% ({successful_conversions}/{total_conversions})"

    print(
        f"✅ Key conversion successful: {conversion_rate*100:.1f}% ({successful_conversions}/{total_conversions})"
    )
