"""Unit tests for VaultGemmaArchitectureAdapter.

VaultGemma is Gemma 2 without the two post-norms, so these tests check the
parts the subclass changes: the block norm layout, the hook aliases that used
to point at the post-norms, and compat mode being gated off.
"""

import pytest
from transformers import VaultGemmaConfig
from transformers.models.vaultgemma.modeling_vaultgemma import VaultGemmaDecoderLayer

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.gemma2 import (
    Gemma2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.vaultgemma import (
    VaultGemmaArchitectureAdapter,
)


@pytest.fixture(scope="module")
def adapter() -> VaultGemmaArchitectureAdapter:
    return VaultGemmaArchitectureAdapter(make_bridge_cfg("VaultGemmaForCausalLM", d_mlp=128))


def test_factory_resolves_vaultgemma_arch_string() -> None:
    assert SUPPORTED_ARCHITECTURES["VaultGemmaForCausalLM"] is VaultGemmaArchitectureAdapter


def test_block_has_only_pre_norms(adapter: VaultGemmaArchitectureAdapter) -> None:
    submodules = adapter.component_mapping["blocks"].submodules
    assert set(submodules) == {"ln1", "ln2", "attn", "mlp"}
    assert submodules["ln1"].name == "input_layernorm"
    assert submodules["ln2"].name == "pre_feedforward_layernorm"


def test_hook_aliases_do_not_point_at_removed_post_norms(
    adapter: VaultGemmaArchitectureAdapter,
) -> None:
    aliases = adapter.component_mapping["blocks"].hook_aliases
    assert aliases["hook_attn_out"] == "attn.hook_out"
    assert aliases["hook_mlp_out"] == "mlp.hook_out"
    assert all("_post" not in str(target) for target in aliases.values())


def test_block_mapping_names_exist_on_hf_decoder_layer(
    adapter: VaultGemmaArchitectureAdapter,
) -> None:
    # Catches the mapping drifting from what transformers actually builds.
    hf_cfg = VaultGemmaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    hf_layer = VaultGemmaDecoderLayer(hf_cfg, layer_idx=0)
    hf_names = {name for name, _ in hf_layer.named_modules()}

    for key, bridge in adapter.component_mapping["blocks"].submodules.items():
        assert bridge.name in hf_names, key
        for sub_key, sub_bridge in bridge.submodules.items():
            assert f"{bridge.name}.{sub_bridge.name}" in hf_names, f"{key}.{sub_key}"


def test_compat_mode_is_gated_off(adapter: VaultGemmaArchitectureAdapter) -> None:
    assert adapter.supports_compatibility_mode is False
    assert 3 not in adapter.applicable_phases
    # Gemma 2 itself still supports it, so this is a deliberate override.
    assert Gemma2ArchitectureAdapter.supports_compatibility_mode is True
