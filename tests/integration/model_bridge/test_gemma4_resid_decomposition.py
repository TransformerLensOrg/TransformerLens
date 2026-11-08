"""Residual-decomposition APIs on Gemma 4.

Each Gemma 4 decoder layer normalizes its attention and MLP outputs (sandwich norms),
E2B/E4B add a per-layer input branch after the MLP, and every layer ends with
``hidden_states *= layer_scalar``, a learned per-layer scale on the whole residual stream.
The decompositions must still sum to the residual stream they decompose.
"""
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.gemma4 import Gemma4TextConfig
except ImportError:  # pragma: no cover - older transformers
    pytest.skip("transformers has no Gemma 4", allow_module_level=True)

ATOL = 1e-5
TOKENS = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
_BASE = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    global_head_dim=16,
    sliding_window=8,
    max_position_embeddings=64,
    layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
)
_VARIANTS = {
    # Per-layer inputs and KV sharing, as on E2B / E4B.
    "e2b_like": dict(
        hidden_size_per_layer_input=16, vocab_size_per_layer_input=128, num_kv_shared_layers=2
    ),
    # K == V global attention, as on 31B.
    "31b_like": dict(
        hidden_size_per_layer_input=0, attention_k_eq_v=True, num_global_key_value_heads=1
    ),
    # Dense MLP plus experts, as on 26B-A4B.
    "moe_like": dict(
        hidden_size_per_layer_input=0,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
    ),
}
# Distinct per-layer scales, so a contribution carried through the wrong layers is caught.
_LAYER_SCALARS = [0.3, 0.7, 1.1, 1.5]


def _gemma4(variant: str) -> TransformerBridge:
    from transformers import AutoModelForCausalLM

    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    torch.manual_seed(0)
    c = Gemma4TextConfig(**_BASE, **_VARIANTS[variant])
    c.architectures = ["Gemma4ForCausalLM"]
    hf = AutoModelForCausalLM.from_config(c, attn_implementation="eager").eval()
    with torch.no_grad():
        for layer, scalar in zip(hf.model.layers, _LAYER_SCALARS):
            layer.layer_scalar.fill_(scalar)
            for name, param in layer.named_parameters():
                if "norm" in name and param.ndim == 1:
                    param.normal_(1.0, 0.3)
    cfg = build_bridge_config_from_hf(hf.config, "Gemma4ForCausalLM", "tiny", torch.float32)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    return TransformerBridge(model=hf, adapter=adapter, tokenizer=MagicMock())


@pytest.fixture(scope="module", params=sorted(_VARIANTS))
def bridge_and_cache(request):
    bridge = _gemma4(request.param)
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
    return request.param, bridge, cache


def _final(bridge, cache):
    return cache[f"blocks.{bridge.cfg.n_layers - 1}.hook_resid_post"]


def _target(bridge, cache, layer, mlp_input=False):
    if mlp_input:
        return cache[f"blocks.{layer}.hook_resid_mid"]
    if layer == bridge.cfg.n_layers:
        return _final(bridge, cache)
    return cache[f"blocks.{layer}.hook_resid_pre"]


def test_layer_scalar_rescales_the_residual_stream(bridge_and_cache):
    """The setup this file relies on: each block's output is scaled after its residual adds."""
    _, bridge, cache = bridge_and_cache
    write = cache["blocks.0.hook_resid_mid"] - cache["blocks.0.hook_resid_pre"]
    torch.testing.assert_close(write, cache["blocks.0.hook_attn_out"], atol=ATOL, rtol=0)
    unscaled = cache["blocks.0.hook_resid_post"] / _LAYER_SCALARS[0]
    assert not torch.allclose(unscaled, cache["blocks.0.hook_resid_post"])


class TestDecomposeResid:
    @pytest.mark.parametrize("layer", [0, 1, 2, 3, 4])
    def test_sums_to_resid_pre(self, bridge_and_cache, layer):
        _, bridge, cache = bridge_and_cache
        comps = cache.decompose_resid(layer=layer)
        torch.testing.assert_close(comps.sum(0), _target(bridge, cache, layer), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("layer", [0, 1, 2, 3])
    def test_mlp_input_sums_to_resid_mid(self, bridge_and_cache, layer):
        _, bridge, cache = bridge_and_cache
        comps = cache.decompose_resid(layer=layer, mlp_input=True)
        torch.testing.assert_close(
            comps.sum(0), _target(bridge, cache, layer, mlp_input=True), atol=ATOL, rtol=0
        )

    def test_labels(self, bridge_and_cache):
        name, bridge, cache = bridge_and_cache
        _, labels = cache.decompose_resid(return_labels=True)
        per_layer = ["attn_out", "mlp_out"]
        if name == "e2b_like":
            per_layer.append("per_layer_input_out")
        assert labels == ["embed"] + [
            f"{l}_{kind}" for l in range(bridge.cfg.n_layers) for kind in per_layer
        ]

    def test_attn_and_mlp_modes_partition_the_stream(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        attn = cache.decompose_resid(mode="attn", incl_embeds=False).sum(0)
        mlp = cache.decompose_resid(mode="mlp", incl_embeds=False).sum(0)
        embed = cache.decompose_resid(mode="attn", return_labels=False)[0]
        torch.testing.assert_close(attn + mlp + embed, _final(bridge, cache), atol=ATOL, rtol=0)


class TestHeadsAndNeurons:
    @pytest.mark.parametrize("layer", [1, 2, 3, 4])
    def test_heads_plus_remainder(self, bridge_and_cache, layer):
        _, bridge, cache = bridge_and_cache
        stack = cache.stack_head_results(layer=layer, incl_remainder=True)
        torch.testing.assert_close(
            stack.sum(0), cache[f"blocks.{layer - 1}.hook_resid_post"], atol=ATOL, rtol=0
        )

    def test_heads_sum_to_scaled_attn_out(self, bridge_and_cache):
        """Through ln1_post and the layer's own layer_scalar."""
        _, bridge, cache = bridge_and_cache
        n_heads = bridge.cfg.n_heads
        for layer in range(bridge.cfg.n_layers):
            heads = cache.stack_head_results(layer=layer + 1)[layer * n_heads :]
            torch.testing.assert_close(
                heads.sum(0),
                cache[f"blocks.{layer}.hook_attn_out"] * _LAYER_SCALARS[layer],
                atol=ATOL,
                rtol=0,
            )

    def test_neurons_plus_remainder(self, bridge_and_cache):
        name, bridge, cache = bridge_and_cache
        if name == "moe_like":
            with pytest.raises(NotImplementedError, match="post_feedforward_layernorm_1"):
                cache.stack_neuron_results(layer=bridge.cfg.n_layers, incl_remainder=True)
            return
        stack = cache.stack_neuron_results(layer=bridge.cfg.n_layers, incl_remainder=True)
        torch.testing.assert_close(stack.sum(0), _final(bridge, cache), atol=ATOL, rtol=0)


class TestFullResidDecomposition:
    @pytest.mark.parametrize("expand_neurons", [False, True])
    def test_sums_to_final_residual(self, bridge_and_cache, expand_neurons):
        name, bridge, cache = bridge_and_cache
        if name == "moe_like" and expand_neurons:
            with pytest.raises(NotImplementedError, match="experts"):
                cache.get_full_resid_decomposition(expand_neurons=True)
            return
        stack, labels = cache.get_full_resid_decomposition(
            expand_neurons=expand_neurons, return_labels=True
        )
        assert stack.shape[0] == len(labels)
        if name == "e2b_like":
            assert "0_per_layer_input_out" in labels
        torch.testing.assert_close(stack.sum(0), _final(bridge, cache), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("expand_neurons", [False, True])
    @pytest.mark.parametrize("layer", [1, 3])
    def test_mlp_input_sums_to_resid_mid(self, bridge_and_cache, expand_neurons, layer):
        name, bridge, cache = bridge_and_cache
        if name == "moe_like" and expand_neurons:
            pytest.skip("neuron expansion is refused on MoE layers")
        stack = cache.get_full_resid_decomposition(
            layer=layer, mlp_input=True, expand_neurons=expand_neurons
        )
        torch.testing.assert_close(
            stack.sum(0), _target(bridge, cache, layer, mlp_input=True), atol=ATOL, rtol=0
        )

    @pytest.mark.parametrize("expand_neurons", [False, True])
    def test_ln_and_projection(self, bridge_and_cache, expand_neurons):
        name, bridge, cache = bridge_and_cache
        if name == "moe_like" and expand_neurons:
            pytest.skip("neuron expansion is refused on MoE layers")
        torch.manual_seed(1)
        directions = torch.randn(bridge.cfg.d_model, 3)
        stack = cache.get_full_resid_decomposition(
            expand_neurons=expand_neurons, apply_ln=True, project_output_onto=directions
        )
        n = bridge.cfg.n_layers
        expected = cache.apply_ln_to_stack(_final(bridge, cache)[None], n)[0] @ directions
        torch.testing.assert_close(stack.sum(0), expected, atol=1e-4, rtol=0)

    def test_logit_attrs_runs(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        resid = cache.decompose_resid()
        attrs = cache.logit_attrs(resid, tokens=torch.tensor([3, 3]))
        assert attrs.shape[0] == resid.shape[0]
