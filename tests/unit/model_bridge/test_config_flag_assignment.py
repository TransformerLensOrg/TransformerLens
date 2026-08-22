"""Tests for direct assignment of Bridge-managed hook flags (#1689)."""

from __future__ import annotations

import copy
import gc

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.model_bridge.sources.native import NativeModel


def _cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )


def _tiny_gpt2_bridge() -> TransformerBridge:
    hf_config = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=32,
        n_positions=8,
        n_ctx=8,
        vocab_size=16,
    )
    hf_model = GPT2LMHeadModel(hf_config).eval()
    return build_bridge_from_module(
        hf_model,
        "GPT2LMHeadModel",
        hf_config=hf_config,
        tokenizer=None,
        device="cpu",
    )


def _tiny_llama_bridge() -> TransformerBridge:
    hf_config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=16,
        max_position_embeddings=8,
    )
    hf_model = LlamaForCausalLM(hf_config).eval()
    return build_bridge_from_module(
        hf_model,
        "LlamaForCausalLM",
        hf_config=hf_config,
        tokenizer=None,
        device="cpu",
    )


@pytest.fixture(params=["gpt2", "llama"], ids=["shared-config", "cloned-config"])
def bridge_with_config_mode(request: pytest.FixtureRequest) -> TransformerBridge:
    bridge = _tiny_gpt2_bridge() if request.param == "gpt2" else _tiny_llama_bridge()
    attn_config_is_shared = bridge.blocks[0].attn.config is bridge.cfg
    assert attn_config_is_shared is (request.param == "gpt2")
    return bridge


@pytest.mark.parametrize(
    ("flag_name", "hook_name"),
    [
        ("use_attn_result", "blocks.0.attn.hook_result"),
        ("use_attn_in", "blocks.0.attn.hook_attn_in"),
        ("use_hook_mlp_in", "blocks.0.hook_mlp_in"),
        ("use_split_qkv_input", "blocks.0.attn.hook_q_input"),
    ],
)
def test_direct_assignment_matches_setter_hook_behavior(
    bridge_with_config_mode: TransformerBridge, flag_name: str, hook_name: str
) -> None:
    bridge = bridge_with_config_mode
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 8))

    setattr(bridge.cfg, flag_name, True)
    _, direct_cache = bridge.run_with_cache(tokens, names_filter=[hook_name])

    setattr(bridge.cfg, flag_name, False)
    getattr(bridge, f"set_{flag_name}")(True)
    _, setter_cache = bridge.run_with_cache(tokens, names_filter=[hook_name])

    assert list(direct_cache) == [hook_name]
    assert list(setter_cache) == [hook_name]
    assert direct_cache[hook_name].shape == setter_cache[hook_name].shape


def test_direct_assignment_preserves_mutual_exclusivity() -> None:
    bridge = _tiny_gpt2_bridge()

    bridge.cfg.use_split_qkv_input = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        bridge.cfg.use_attn_in = True
    assert bridge.cfg.use_attn_in is False

    bridge.cfg.use_split_qkv_input = False
    bridge.cfg.use_attn_in = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        bridge.cfg.use_split_qkv_input = True
    assert bridge.cfg.use_split_qkv_input is False


@pytest.mark.parametrize("flag_name", ["use_attn_result", "use_attn_in", "use_split_qkv_input"])
def test_direct_assignment_preserves_unsupported_architecture_errors(
    monkeypatch: pytest.MonkeyPatch, flag_name: str
) -> None:
    bridge = TransformerBridge.boot_native(_cfg())

    class _FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Identity()

    monkeypatch.setattr(bridge, "blocks", nn.ModuleList([_FakeBlock()]), raising=True)

    with pytest.raises(NotImplementedError, match=flag_name):
        setattr(bridge.cfg, flag_name, True)
    assert getattr(bridge.cfg, flag_name) is False


def test_deepcopied_live_config_is_not_bound_to_original_bridge() -> None:
    bridge = TransformerBridge.boot_native(_cfg())
    copied_cfg = copy.deepcopy(bridge.cfg)

    copied_cfg.use_hook_mlp_in = True

    assert copied_cfg.use_hook_mlp_in is True
    assert bridge.cfg.use_hook_mlp_in is False


def test_deepcopied_bridge_rebinds_its_config() -> None:
    bridge = TransformerBridge.boot_native(_cfg())
    copied_bridge = copy.deepcopy(bridge)

    copied_bridge.cfg.use_hook_mlp_in = True

    assert copied_bridge.cfg.use_hook_mlp_in is True
    assert copied_bridge.blocks[0].config.use_hook_mlp_in is True
    assert bridge.cfg.use_hook_mlp_in is False


def test_shallow_copied_bridge_does_not_replace_live_config_binding() -> None:
    bridge = TransformerBridge.boot_native(_cfg())
    with pytest.warns(UserWarning, match="already bound to another live"):
        copied_bridge = copy.copy(bridge)

    assert copied_bridge.cfg is bridge.cfg
    assert bridge.cfg._bridge_ref() is bridge

    del copied_bridge
    gc.collect()
    bridge.cfg.use_hook_mlp_in = True

    assert bridge.blocks[0].config.use_hook_mlp_in is True


def test_constructor_warns_when_live_bridge_already_owns_config() -> None:
    cfg = _cfg()
    cfg.architecture = "TransformerLensNative"
    first_model = NativeModel(cfg)
    second_model = NativeModel(cfg)
    first_adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    second_adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    first_adapter.prepare_model(first_model)
    second_adapter.prepare_model(second_model)
    first_bridge = TransformerBridge(first_model, first_adapter, tokenizer=None)

    with pytest.warns(UserWarning, match="already bound to another live"):
        second_bridge = TransformerBridge(second_model, second_adapter, tokenizer=None)

    assert second_bridge.cfg is first_bridge.cfg
    assert cfg._bridge_ref() is first_bridge


def test_attention_flag_propagation_does_not_dispatch_bound_cloned_config() -> None:
    bridge = _tiny_llama_bridge()
    cloned_cfg = bridge.blocks[0].attn.config
    other_bridge = TransformerBridge.boot_native(_cfg())
    assert cloned_cfg is not bridge.cfg
    cloned_cfg._bind_bridge(other_bridge)

    bridge.set_use_attn_in(True)

    assert cloned_cfg.use_attn_in is True
    assert other_bridge.cfg.use_attn_in is False


def test_mlp_flag_propagation_does_not_dispatch_bound_cloned_config() -> None:
    bridge = _tiny_gpt2_bridge()
    cloned_cfg = bridge.blocks[0].config
    other_bridge = TransformerBridge.boot_native(_cfg())
    assert cloned_cfg is not bridge.cfg
    cloned_cfg._bind_bridge(other_bridge)

    bridge.set_use_hook_mlp_in(True)

    assert cloned_cfg.use_hook_mlp_in is True
    assert other_bridge.cfg.use_hook_mlp_in is False
