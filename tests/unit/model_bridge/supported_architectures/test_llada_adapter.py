"""Offline unit tests for the dense LLaDA architecture adapter."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.sources.transformers import (
    determine_architecture_from_hf_config,
)
from transformer_lens.model_bridge.supported_architectures.llada import (
    LLaDAArchitectureAdapter,
)


def _hf_config(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "activation_type": "silu",
        "alibi": False,
        "architectures": ["LLaDAModelLM"],
        "attention_dropout": 0.0,
        "attention_layer_norm": False,
        "block_group_size": 1,
        "block_type": "llama",
        "d_model": 32,
        "embedding_dropout": 0.0,
        "embedding_size": 64,
        "eos_token_id": 1,
        "include_bias": False,
        "include_qkv_bias": False,
        "input_emb_norm": False,
        "layer_norm_type": "rms",
        "mask_token_id": 63,
        "max_sequence_length": 16,
        "mlp_hidden_size": 48,
        "model_type": "llada",
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_layers": 2,
        "pad_token_id": 0,
        "residual_dropout": 0.0,
        "rms_norm_eps": 1e-5,
        "rope": True,
        "rope_full_precision": True,
        "rope_theta": 500_000.0,
        "scale_logits": False,
        "use_cache": False,
        "vocab_size": 64,
        "weight_tying": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bridge_config(**overrides: Any) -> TransformerBridgeConfig:
    config = build_bridge_config_from_hf(
        _hf_config(),
        architecture="LLaDAModelLM",
        model_name="tiny-llada",
        dtype=torch.float32,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


@pytest.fixture(scope="module")
def adapter() -> LLaDAArchitectureAdapter:
    return LLaDAArchitectureAdapter(_bridge_config())


def _rearrange(adapter: LLaDAArchitectureAdapter, key: str) -> RearrangeTensorConversion:
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    conversion = conversions[key]
    assert isinstance(conversion, ParamProcessingConversion)
    assert isinstance(conversion.tensor_conversion, RearrangeTensorConversion)
    return conversion.tensor_conversion


def test_config_conversion_preserves_llada_fields() -> None:
    config = _bridge_config()

    expected = {
        "d_model": 32,
        "d_head": 8,
        "n_layers": 2,
        "n_ctx": 16,
        "n_heads": 4,
        "n_key_value_heads": 2,
        "d_mlp": 48,
        "d_vocab": 64,
        "act_fn": "silu",
        "eps": 1e-5,
        "rotary_base": 500_000,
        "tie_word_embeddings": False,
        "mask_token_id": 63,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "block_type": "llama",
        "block_group_size": 1,
        "rope": True,
        "rope_full_precision": True,
        "attention_layer_norm": False,
        "include_bias": False,
        "include_qkv_bias": False,
        "scale_logits": False,
        "input_emb_norm": False,
        "layer_norm_type": "rms",
    }
    for name, value in expected.items():
        assert getattr(config, name) == value


def test_factory_and_model_type_dispatch_to_llada() -> None:
    selected = ArchitectureAdapterFactory.select_architecture_adapter(_bridge_config())
    assert isinstance(selected, LLaDAArchitectureAdapter)

    config = SimpleNamespace(architectures=[], model_type="llada")
    assert determine_architecture_from_hf_config(config) == "LLaDAModelLM"


def test_adapter_sets_bidirectional_forward_only_semantics(
    adapter: LLaDAArchitectureAdapter,
) -> None:
    assert adapter.cfg.attention_dir == "bidirectional"
    assert adapter.cfg.normalization_type == "RMS"
    assert adapter.cfg.uses_rms_norm is True
    assert adapter.cfg.positional_embedding_type == "rotary"
    assert adapter.cfg.rotary_adjacent_pairs is False
    assert adapter.cfg.gated_mlp is True
    assert adapter.cfg.final_rms is True
    assert adapter.cfg.d_vocab_out == adapter.cfg.d_vocab == 64
    assert adapter.cfg.default_prepend_bos is False
    assert adapter.cfg.default_padding_side == "right"
    assert adapter.supports_generation is False
    assert adapter.supports_hf_output_attentions is False
    assert adapter.supports_causal_loss is False

    attention = adapter.component_mapping["blocks"].submodules["attn"]
    assert isinstance(attention, AttentionBridge)
    assert attention.is_causal is False


def test_component_mapping_matches_block_local_module_tree(
    adapter: LLaDAArchitectureAdapter,
) -> None:
    mapping = adapter.component_mapping
    assert isinstance(mapping["embed"], EmbeddingBridge)
    assert isinstance(mapping["blocks"], BlockBridge)
    assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
    assert isinstance(mapping["unembed"], UnembeddingBridge)
    assert mapping["embed"].name == "model.transformer.wte"
    assert mapping["blocks"].name == "model.transformer.blocks"
    assert mapping["ln_final"].name == "model.transformer.ln_f"
    assert mapping["unembed"].name == "model.transformer.ff_out"

    block = mapping["blocks"]
    assert block.submodules["ln1"].name == "attn_norm"
    assert block.submodules["ln2"].name == "ff_norm"
    assert isinstance(block.submodules["ln1"], RMSNormalizationBridge)
    assert isinstance(block.submodules["ln2"], RMSNormalizationBridge)

    attention = block.submodules["attn"]
    assert isinstance(attention, AttentionBridge)
    assert {name: component.name for name, component in attention.submodules.items()} == {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "o": "attn_out",
    }
    assert all(isinstance(component, LinearBridge) for component in attention.submodules.values())

    mlp = block.submodules["mlp"]
    assert isinstance(mlp, GatedMLPBridge)
    assert {name: component.name for name, component in mlp.submodules.items()} == {
        "gate": "ff_proj",
        "in": "up_proj",
        "act": "act",
        "out": "ff_out",
    }


@pytest.mark.parametrize(
    "tl_path,hf_path",
    [
        ("blocks.0.attn.W_Q", "model.transformer.blocks.0.q_proj.weight"),
        ("blocks.0.attn.W_K", "model.transformer.blocks.0.k_proj.weight"),
        ("blocks.0.attn.W_V", "model.transformer.blocks.0.v_proj.weight"),
        ("blocks.0.attn.W_O", "model.transformer.blocks.0.attn_out.weight"),
        ("blocks.0.mlp.W_gate", "model.transformer.blocks.0.ff_proj.weight"),
        ("blocks.0.mlp.W_in", "model.transformer.blocks.0.up_proj.weight"),
        ("blocks.0.mlp.W_out", "model.transformer.blocks.0.ff_out.weight"),
    ],
)
def test_containerless_paths_translate_to_direct_block_fields(
    adapter: LLaDAArchitectureAdapter, tl_path: str, hf_path: str
) -> None:
    assert adapter.translate_transformer_lens_path(tl_path) == hf_path


@pytest.mark.parametrize(
    "hf_key,tl_key",
    [
        ("model.transformer.blocks.0.q_proj.weight", "blocks.0.attn.q.weight"),
        ("model.transformer.blocks.0.k_proj.weight", "blocks.0.attn.k.weight"),
        ("model.transformer.blocks.0.v_proj.weight", "blocks.0.attn.v.weight"),
        ("model.transformer.blocks.0.attn_out.weight", "blocks.0.attn.o.weight"),
        ("model.transformer.blocks.0.ff_proj.weight", "blocks.0.mlp.gate.weight"),
        ("model.transformer.blocks.0.up_proj.weight", "blocks.0.mlp.in.weight"),
        ("model.transformer.blocks.0.ff_out.weight", "blocks.0.mlp.out.weight"),
    ],
)
def test_hf_keys_translate_back_to_containerless_components(
    adapter: LLaDAArchitectureAdapter, hf_key: str, tl_key: str
) -> None:
    assert adapter.convert_hf_key_to_tl_key(hf_key) == tl_key


def test_qkvo_weight_conversions_preserve_gqa_head_counts(
    adapter: LLaDAArchitectureAdapter,
) -> None:
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    assert set(conversions) == {
        "blocks.{i}.attn.q.weight",
        "blocks.{i}.attn.k.weight",
        "blocks.{i}.attn.v.weight",
        "blocks.{i}.attn.o.weight",
    }
    assert _rearrange(adapter, "blocks.{i}.attn.q.weight").axes_lengths["n"] == 4
    assert _rearrange(adapter, "blocks.{i}.attn.k.weight").axes_lengths["n"] == 2
    assert _rearrange(adapter, "blocks.{i}.attn.v.weight").axes_lengths["n"] == 2
    assert _rearrange(adapter, "blocks.{i}.attn.o.weight").axes_lengths["n"] == 4


def test_equal_kv_and_query_head_counts_use_mha_default() -> None:
    config = build_bridge_config_from_hf(
        _hf_config(n_kv_heads=4),
        architecture="LLaDAModelLM",
        model_name="tiny-llada",
        dtype=torch.float32,
    )

    assert config.n_key_value_heads is None


@pytest.mark.parametrize(
    "field,value,error_fragment",
    [
        ("block_type", "sequential", "block_type"),
        ("block_group_size", 2, "block_group_size"),
        ("rope", False, "rope"),
        ("rope_full_precision", False, "rope_full_precision"),
        ("alibi", True, "alibi"),
        ("attention_layer_norm", True, "attention_layer_norm"),
        ("include_bias", True, "include_bias"),
        ("include_qkv_bias", True, "include_qkv_bias"),
        ("scale_logits", True, "scale_logits"),
        ("input_emb_norm", True, "input_emb_norm"),
        ("layer_norm_type", "default", "layer_norm_type"),
        ("act_fn", "relu", "activation_type"),
        ("tie_word_embeddings", True, "tied embeddings"),
        ("embedding_size", 128, "embedding_size"),
    ],
)
def test_unsupported_dense_variants_fail_clearly(
    field: str, value: Any, error_fragment: str
) -> None:
    with pytest.raises(ValueError, match=error_fragment):
        LLaDAArchitectureAdapter(_bridge_config(**{field: value}))


def test_existing_causal_attention_default_remains_triangular() -> None:
    config = make_bridge_cfg(
        "TransformerLensNative",
        d_model=16,
        d_head=4,
        n_heads=4,
        n_layers=1,
        n_ctx=5,
        d_vocab=32,
        d_mlp=32,
        act_fn="silu",
        normalization_type="RMS",
        final_rms=True,
        gated_mlp=True,
        attention_dir="causal",
        default_prepend_bos=False,
        seed=0,
    )
    bridge = TransformerBridge.boot_native(config, device="cpu")
    tokens = torch.tensor([[1, 2, 3, 4, 5]])

    _, cache = bridge.run_with_cache(
        tokens,
        names_filter="blocks.0.attn.hook_pattern",
    )

    pattern = cache["blocks.0.attn.hook_pattern"]
    assert pattern.shape == (1, 4, 5, 5)
    assert torch.count_nonzero(torch.triu(pattern, diagonal=1)) == 0
