"""Hand-reimplemented SSM hooked forwards vs raw HF, same weights, same input.

`GatedDeltaNetBridge._hooked_forward` (qwen3_next linear attention) and
`SSM2MixerBridge` (nemotron_h, zamba2) re-derive HF's recurrences to place
hooks — the run_with_cache path for those layers. A drift here is silent wrong
numbers for every cached activation, so parity is pinned bit-exact on the
torch fallback path. (Kernel paths — fla/causal-conv1d — are absent in CI.)
"""

from __future__ import annotations

import copy
import sys

import pytest
import torch

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.component_setup import setup_submodules


def _wire(arch: str, key: str, hf_module):
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(
        make_bridge_cfg(arch, d_model=32, n_heads=4, d_head=8)
    )
    bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules[key])
    bridge.set_original_component(hf_module)
    setup_submodules(bridge, adapter, hf_module)
    return bridge


def _first(output):
    return output[0] if isinstance(output, tuple) else output


def test_gated_delta_net_hooked_forward_is_bit_exact() -> None:
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet,
    )

    torch.manual_seed(0)
    cfg = Qwen3NextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        vocab_size=64,
        intermediate_size=64,
    )
    hf_module = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).eval()
    reference = copy.deepcopy(hf_module)
    bridge = _wire("Qwen3NextForCausalLM", "linear_attn", hf_module)

    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        torch.testing.assert_close(_first(bridge(x)), _first(reference(x)), rtol=0.0, atol=0.0)


def test_gated_delta_net_hooked_forward_without_instance_kernel_attrs() -> None:
    """transformers >= 5.15 no longer sets causal_conv1d_fn / chunk_gated_delta_rule
    on the GatedDeltaNet instance; the hooked forward must not depend on them."""
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet,
    )

    torch.manual_seed(0)
    cfg = Qwen3NextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        vocab_size=64,
        intermediate_size=64,
    )
    hf_module = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).eval()
    reference = copy.deepcopy(hf_module)
    for attr in ("causal_conv1d_fn", "chunk_gated_delta_rule"):
        hf_module.__dict__.pop(attr, None)
    bridge = _wire("Qwen3NextForCausalLM", "linear_attn", hf_module)

    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        torch.testing.assert_close(_first(bridge(x)), _first(reference(x)), rtol=0.0, atol=0.0)


def _qwen3_next_gdn_without_instance_kernel_attrs():
    """A Qwen3Next GatedDeltaNet shaped like transformers >= 5.15: no per-instance
    causal_conv1d_fn / chunk_gated_delta_rule, the forward uses module-level functions."""
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet,
    )

    torch.manual_seed(0)
    cfg = Qwen3NextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        vocab_size=64,
        intermediate_size=64,
    )
    hf_module = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).eval()
    for attr in ("causal_conv1d_fn", "chunk_gated_delta_rule"):
        hf_module.__dict__.pop(attr, None)
    return hf_module


class _ConvCalled(Exception):
    pass


def test_gated_delta_net_hooked_forward_calls_module_level_causal_conv1d_fn(
    monkeypatch,
) -> None:
    """On transformers >= 5.15 HF calls the module-level causal_conv1d_fn (the
    causal-conv1d kernel when installed); the bridge must call the same function."""
    hf_module = _qwen3_next_gdn_without_instance_kernel_attrs()
    bridge = _wire("Qwen3NextForCausalLM", "linear_attn", hf_module)

    def sentinel(*args, **kwargs):
        raise _ConvCalled

    monkeypatch.setattr(sys.modules[type(hf_module).__module__], "causal_conv1d_fn", sentinel)
    with torch.no_grad(), pytest.raises(_ConvCalled):
        bridge(torch.randn(2, 7, 32))


def test_gated_delta_net_hooked_forward_names_transformers_range_when_fn_missing(
    monkeypatch,
) -> None:
    hf_module = _qwen3_next_gdn_without_instance_kernel_attrs()
    bridge = _wire("Qwen3NextForCausalLM", "linear_attn", hf_module)

    monkeypatch.delattr(sys.modules[type(hf_module).__module__], "causal_conv1d_fn")
    with torch.no_grad(), pytest.raises(AttributeError, match="transformers >= 5.9"):
        bridge(torch.randn(2, 7, 32))


def test_ssm2_mixer_hooked_forward_is_bit_exact() -> None:
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

    torch.manual_seed(0)
    cfg = NemotronHConfig(
        hidden_size=32,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=8,
        conv_kernel=4,
        n_groups=1,
        num_hidden_layers=2,
        vocab_size=64,
        intermediate_size=64,
        expand=2,
        hybrid_override_pattern="MM",
    )
    hf_module = NemotronHMamba2Mixer(cfg, layer_idx=0).eval()
    reference = copy.deepcopy(hf_module)
    bridge = _wire("NemotronHForCausalLM", "mixer", hf_module)

    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        torch.testing.assert_close(_first(bridge(x)), _first(reference(x)), rtol=0.0, atol=0.0)
