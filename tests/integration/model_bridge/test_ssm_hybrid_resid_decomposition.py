"""Residual-decomposition APIs on SSM / hybrid bridges.

Blocks without attention (gated-delta-net, Mamba) expose their sequence-mixer
output instead of ``hook_attn_out``; blocks without an MLP (Mamba, NemotronH)
write only through their mixer. Every decomposition must still sum to the
residual stream it decomposes.
"""
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    _QWEN3_5_AVAILABLE = True
except ImportError:
    _QWEN3_5_AVAILABLE = False

ATOL = 1e-5
TOKENS = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])


def _boot(hf_model, arch):
    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    cfg = build_bridge_config_from_hf(hf_model.config, arch, "tiny", torch.float32)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=MagicMock())


def _qwen35():
    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )
    from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
        Qwen3_5ArchitectureAdapter,
    )

    torch.manual_seed(0)
    c = Qwen3_5TextConfig(
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=512,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    m = Qwen3_5ForCausalLM(c).eval()
    bcfg = TransformerBridgeConfig(
        d_model=128,
        d_head=32,
        n_heads=4,
        n_layers=8,
        n_ctx=2048,
        d_vocab=512,
        d_mlp=256,
        n_key_value_heads=2,
        architecture="Qwen3_5ForCausalLM",
    )
    return TransformerBridge(m, Qwen3_5ArchitectureAdapter(bcfg), tokenizer=MagicMock())


def _mamba2(use_bias: bool = False):
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba2 import Mamba2Config

    torch.manual_seed(0)
    c = Mamba2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        state_size=16,
        expand=2,
        head_dim=16,
        num_heads=4,
        n_groups=2,
        conv_kernel=4,
        chunk_size=8,
        use_bias=use_bias,
    )
    c.architectures = ["Mamba2ForCausalLM"]
    model = AutoModelForCausalLM.from_config(c).eval()
    if use_bias:
        # Non-zero output biases, so a double-counted bias row would show.
        with torch.no_grad():
            for layer in model.backbone.layers:
                layer.mixer.out_proj.bias.normal_()
    return _boot(model, "Mamba2ForCausalLM")


def _mamba2_bias():
    return _mamba2(use_bias=True)


def _nemotronh():
    from transformers import AutoModelForCausalLM
    from transformers.models.nemotron_h import NemotronHConfig

    torch.manual_seed(0)
    c = NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        layers_block_type=["mamba", "attention", "mamba", "mlp"],
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
    c.architectures = ["NemotronHForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "NemotronHForCausalLM")


_FACTORIES = {"mamba2": _mamba2, "mamba2_bias": _mamba2_bias, "nemotron_h": _nemotronh}
if _QWEN3_5_AVAILABLE:
    _FACTORIES["qwen3_5"] = _qwen35


@pytest.fixture(scope="module", params=sorted(_FACTORIES))
def bridge_and_cache(request):
    bridge = _FACTORIES[request.param]()
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
    return request.param, bridge, cache


def _final_resid(bridge, cache):
    return cache[f"blocks.{bridge.cfg.n_layers - 1}.hook_resid_post"]


def _attn_layers(bridge):
    return [i for i, _ in bridge.blocks_with("attn")]


class TestDecomposeResid:
    def test_sums_to_final_residual(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        comps = cache.decompose_resid()
        torch.testing.assert_close(comps.sum(0), _final_resid(bridge, cache), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("layer", [1, 2, 3])
    def test_sums_to_resid_pre(self, bridge_and_cache, layer):
        _, bridge, cache = bridge_and_cache
        if layer >= bridge.cfg.n_layers:
            pytest.skip("model has fewer layers")
        comps = cache.decompose_resid(layer=layer)
        torch.testing.assert_close(
            comps.sum(0), cache[f"blocks.{layer}.hook_resid_pre"], atol=ATOL, rtol=0
        )

    def test_attn_and_mlp_modes_partition_the_stream(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        attn = cache.decompose_resid(mode="attn", incl_embeds=False)
        mlp = cache.decompose_resid(mode="mlp", incl_embeds=False)
        embed = cache["hook_embed"]
        torch.testing.assert_close(
            attn.sum(0) + mlp.sum(0) + embed, _final_resid(bridge, cache), atol=ATOL, rtol=0
        )

    def test_labels_name_the_mixer(self, bridge_and_cache):
        name, bridge, cache = bridge_and_cache
        _, labels = cache.decompose_resid(return_labels=True)
        expected = {
            "qwen3_5": ["0_linear_attn_out", "0_mlp_out", "3_attn_out", "3_mlp_out"],
            "mamba2": ["0_mixer_out", "1_mixer_out"],
            "mamba2_bias": ["0_mixer_out", "1_mixer_out"],
            "nemotron_h": ["0_mixer_out", "1_mixer_out", "2_mixer_out", "3_mixer_out"],
        }[name]
        for label in expected:
            assert label in labels
        if name != "qwen3_5":
            assert not any(label.endswith("_mlp_out") for label in labels)

    def test_mlp_input_sums_to_resid_mid(self, bridge_and_cache):
        name, bridge, cache = bridge_and_cache
        if name != "qwen3_5":
            pytest.skip("only Qwen3.5 blocks have a separate MLP input")
        for layer in (0, 3):
            comps = cache.decompose_resid(layer=layer, mlp_input=True)
            torch.testing.assert_close(
                comps.sum(0), cache[f"blocks.{layer}.hook_resid_mid"], atol=ATOL, rtol=0
            )

    def test_logit_attrs_runs(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        attrs = cache.logit_attrs(cache.decompose_resid(), tokens=torch.tensor([3, 3]))
        assert attrs.shape[0] == cache.decompose_resid().shape[0]


class TestHeadResults:
    def test_heads_only_on_attention_layers(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        _, labels = cache.stack_head_results(incl_remainder=False, return_labels=True)
        expected = [f"L{l}H{h}" for l in _attn_layers(bridge) for h in range(bridge.cfg.n_heads)]
        assert labels == expected

    def test_heads_plus_remainder_sum_to_final_residual(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        stack = cache.stack_head_results(incl_remainder=True)
        torch.testing.assert_close(stack.sum(0), _final_resid(bridge, cache), atol=ATOL, rtol=0)

    def test_head_results_sum_to_attn_out(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        cache.compute_head_results()
        for layer in _attn_layers(bridge):
            heads = cache[f"blocks.{layer}.attn.hook_result"].sum(-2)
            b_O = bridge.blocks[layer].attn.b_O
            if b_O is not None:
                heads = heads + b_O
            torch.testing.assert_close(
                heads, cache[f"blocks.{layer}.hook_attn_out"], atol=ATOL, rtol=0
            )


class TestFullResidDecomposition:
    @pytest.mark.parametrize("expand_neurons", [False, True])
    def test_sums_to_final_residual(self, bridge_and_cache, expand_neurons):
        _, bridge, cache = bridge_and_cache
        stack, labels = cache.get_full_resid_decomposition(
            expand_neurons=expand_neurons, return_labels=True
        )
        assert stack.shape[0] == len(labels)
        torch.testing.assert_close(stack.sum(0), _final_resid(bridge, cache), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("expand_neurons", [False, True])
    def test_ln_and_projection_sum_to_projected_scaled_residual(
        self, bridge_and_cache, expand_neurons
    ):
        _, bridge, cache = bridge_and_cache
        torch.manual_seed(1)
        directions = torch.randn(bridge.cfg.d_model, 3)
        stack = cache.get_full_resid_decomposition(
            expand_neurons=expand_neurons, apply_ln=True, project_output_onto=directions
        )
        final = _final_resid(bridge, cache)
        expected = cache.apply_ln_to_stack(final[None], bridge.cfg.n_layers)[0] @ directions
        torch.testing.assert_close(stack.sum(0), expected, atol=1e-4, rtol=0)
