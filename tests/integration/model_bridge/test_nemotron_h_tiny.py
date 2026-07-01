"""Tiny from_config NemotronH integration tests — CI coverage without the 8B checkpoint.

The full nvidia/Nemotron-H-8B-Base parity suite (test_nemotron_h_adapter.py) is
``@pytest.mark.slow`` and needs ~18 GB, so it does not run in normal CI. This file
builds a tiny synthetic NemotronH (real random CPU weights, no Hub download) that
exercises the same surfaces — the single passthrough ``.mixer`` slot on every
layer, forward parity, hook coverage, and the family-agnostic SSM dispatch.
"""
import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.nemotron_h import NemotronHConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    SSM2MixerBridge,
    SSMBlockBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.nemotron_h import (
    NemotronHArchitectureAdapter,
)

LAYERS = ["mamba", "attention", "mamba", "mlp"]
MAMBA_LAYERS = [0, 2]
ATTN_LAYER = 1


class _Tok:
    pass


@pytest.fixture(scope="module")
def bridge():
    torch.manual_seed(0)
    cfg = NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        layers_block_type=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        ssm_state_size=16,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=2,
        conv_kernel=4,
        expand=2,
        intermediate_size=128,
        chunk_size=8,
    )
    cfg.architectures = ["NemotronHForCausalLM"]
    hf = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    bridge_cfg = build_bridge_config_from_hf(
        hf.config, "NemotronHForCausalLM", "nh-tiny", torch.float32
    )
    return TransformerBridge(hf, NemotronHArchitectureAdapter(bridge_cfg), tokenizer=_Tok())


@pytest.fixture(scope="module")
def tokens():
    return torch.tensor([[1, 2, 3, 4, 5]])


@pytest.fixture(scope="module")
def cache(bridge, tokens):
    with torch.no_grad():
        _, c = bridge.run_with_cache(tokens)
    return c


class TestNemotronHTinyStructure:
    def test_block_count(self, bridge):
        assert len(bridge.blocks) == len(LAYERS)

    def test_blocks_are_ssm_block_bridge(self, bridge):
        assert isinstance(bridge.blocks[0], SSMBlockBridge)

    def test_every_block_wires_ssm2_mixer_passthrough(self, bridge):
        # NemotronH wires one SSM2MixerBridge .mixer per layer, for all types.
        for i in range(len(LAYERS)):
            assert isinstance(bridge.blocks[i].mixer, SSM2MixerBridge)

    def test_layers_block_type_populated(self, bridge):
        assert getattr(bridge.cfg, "layers_block_type", None) == LAYERS


class TestNemotronHTinyForwardPass:
    def test_forward_matches_hf_exactly(self, bridge, tokens):
        with torch.no_grad():
            bridge_out = bridge(tokens)
            hf_out = bridge.original_model(tokens).logits
        assert (bridge_out.float() - hf_out.float()).abs().max().item() == 0.0

    def test_no_nan_longer_sequence(self, bridge):
        with torch.no_grad():
            out = bridge(torch.arange(1, 17).unsqueeze(0))
        assert not torch.isnan(out).any()


class TestNemotronHTinyHookCoverage:
    def test_block_hooks_fire(self, cache):
        for i in range(len(LAYERS)):
            assert f"blocks.{i}.hook_in" in cache
            assert f"blocks.{i}.hook_out" in cache

    def test_mamba_submodule_hooks_fire(self, cache):
        for i in MAMBA_LAYERS:
            for submod in ("in_proj", "conv1d", "out_proj"):
                assert f"blocks.{i}.mixer.{submod}.hook_out" in cache

    def test_no_transformer_specific_hooks(self, cache):
        forbidden = ("hook_resid_mid", "hook_attn_out", "hook_mlp_out")
        assert [k for k in cache if any(f in k for f in forbidden)] == []


class TestNemotronHTinyDispatch:
    def test_ssm_layers_excludes_passthrough(self, cache):
        # Structural passthrough exclusion: only the real Mamba layers, no attn/mlp.
        assert cache.ssm_layers() == MAMBA_LAYERS

    def test_effective_attention_per_ssm_layer_dict(self, cache):
        M = cache.compute_ssm_effective_attention()
        assert isinstance(M, dict)
        assert sorted(M.keys()) == MAMBA_LAYERS

    def test_ssm_state_per_ssm_layer_dict(self, cache):
        S = cache.compute_ssm_state()
        assert isinstance(S, dict)
        assert sorted(S.keys()) == MAMBA_LAYERS

    def test_attention_layer_raises_typeerror(self, cache):
        with pytest.raises(TypeError):
            cache.compute_ssm_effective_attention(layer=ATTN_LAYER)
