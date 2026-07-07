"""Integration tests for the Bamba hybrid architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "hmellor/tiny-random-BambaForCausalLM"


@pytest.fixture(scope="module")
def bamba_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(bamba_bridge):
    torch.manual_seed(0)
    return torch.randint(0, bamba_bridge.cfg.d_vocab - 10, (1, 8))


class TestBambaBridgeCreation:
    def test_adapter_and_layer_types(self, bamba_bridge):
        from transformer_lens.model_bridge.supported_architectures.bamba import (
            BambaArchitectureAdapter,
        )

        assert isinstance(bamba_bridge.adapter, BambaArchitectureAdapter)
        # The tiny checkpoint covers both mixer types.
        assert set(bamba_bridge.cfg.layers_block_type) == {"mamba", "attention"}

    def test_layer_type_module_presence(self, bamba_bridge):
        types = bamba_bridge.cfg.layers_block_type
        for i, layer_type in enumerate(types):
            block = bamba_bridge.blocks[i]
            if layer_type == "mamba":
                assert "mixer" in block._modules
                assert "attn" not in block._modules
            else:
                assert "attn" in block._modules
                assert "mixer" not in block._modules


class TestBambaForwardEquivalence:
    def test_forward_matches_hf(self, bamba_bridge, sample_tokens):
        hf_model = bamba_bridge.original_model
        with torch.no_grad():
            bridge_out = bamba_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestBambaHooks:
    def test_hooks_fire_on_both_layer_types(self, bamba_bridge, sample_tokens):
        types = bamba_bridge.cfg.layers_block_type
        mamba_i = types.index("mamba")
        attn_i = types.index("attention")
        d_model = bamba_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            f"blocks.{mamba_i}.mixer.hook_out",
            f"blocks.{mamba_i}.mlp.hook_out",
            f"blocks.{attn_i}.attn.hook_out",
            f"blocks.{attn_i}.mlp.hook_out",
        ]
        with torch.no_grad():
            bamba_bridge.run_with_hooks(
                sample_tokens, use_cache=False, fwd_hooks=[(name, grab) for name in hooks]
            )
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"


class TestBambaStatefulGeneration:
    def test_generation_runs_with_past_key_values_cache(self, bamba_bridge):
        """Bamba's top-level forward takes past_key_values, not cache_params;
        the stateful loop must pick the right kwarg or the layer receives a
        duplicate cache_params and crashes."""
        out = bamba_bridge.generate("Hello world", max_new_tokens=4, do_sample=False, verbose=False)
        assert isinstance(out, str)
        assert out.startswith("Hello world")
        assert len(out) > len("Hello world")


class TestBambaHFDelegation:
    def test_components_are_shared_wrappers(self, bamba_bridge):
        types = bamba_bridge.cfg.layers_block_type
        mamba_i = types.index("mamba")
        attn_i = types.index("attention")
        hf_model = bamba_bridge.original_model
        assert bamba_bridge.blocks[mamba_i].mixer is hf_model.model.layers[mamba_i].mamba
        assert bamba_bridge.blocks[attn_i].attn.q is hf_model.model.layers[attn_i].self_attn.q_proj
        assert (
            bamba_bridge.blocks[mamba_i].mlp.submodules["gate"]
            is hf_model.model.layers[mamba_i].feed_forward.gate_proj
        )
