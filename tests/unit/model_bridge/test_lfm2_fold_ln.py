"""fold_ln reaches every LFM2 norm, not only the ones an attention layer feeds.

LFM2 interleaves attention layers with short-conv layers, and LFM2-MoE adds
dense-FFN and sparse-MoE layers to the same stack. Folding used to run only where a
q_proj and an unbatched gated MLP existed, so a folded model kept its learned gains on
the conv layers' operator_norm and (LFM2-MoE) on every ffn_norm. Logits stayed right,
so the mixed basis surfaced only in DLA / logit-lens / factored-matrix reads, where
some layers were scaled and others were not.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.lfm2 import Lfm2Config
from transformers.models.lfm2_moe import Lfm2MoeConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.lfm2 import (
    Lfm2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.lfm2_moe import (
    Lfm2MoeArchitectureAdapter,
)

LAYER_TYPES = ["conv", "full_attention", "conv", "full_attention"]
CONV_LAYERS = [0, 2]
DENSE_FFN_LAYERS = [0]
SPARSE_FFN_LAYERS = [1, 2, 3]
TOKENS = torch.tensor([[1, 5, 9, 13, 17, 21]])


class _Tok:
    pass


def _unwrap(module):
    """Descend through bridge wrappers to the HuggingFace module holding the weights."""
    while hasattr(module, "_original_component"):
        module = module._original_component
    return module


def _randomize_weights(hf_model):
    """Non-unit gains and O(1) logits.

    from_config leaves every norm at 1.0, which makes folding a vacuous multiply, and
    leaves the projections at std=0.02, which puts the logits so close to zero that
    even a badly misplaced fold moves them less than the fp tolerance.
    """
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                param.copy_(torch.rand_like(param) + 0.5)
            else:
                param.normal_(0.0, 0.2)


def _build(hf_config, architecture, adapter_class):
    hf_config.architectures = [architecture]
    hf_model = AutoModelForCausalLM.from_config(hf_config).to(torch.float32).eval()
    _randomize_weights(hf_model)
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, architecture, "lfm2-tiny", torch.float32
    )
    return TransformerBridge(hf_model, adapter_class(bridge_config), tokenizer=_Tok())


@pytest.fixture
def lfm2_bridge():
    torch.manual_seed(0)
    return _build(
        Lfm2Config(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=len(LAYER_TYPES),
            layer_types=LAYER_TYPES,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            conv_L_cache=3,
        ),
        "Lfm2ForCausalLM",
        Lfm2ArchitectureAdapter,
    )


@pytest.fixture
def lfm2_moe_bridge():
    torch.manual_seed(0)
    return _build(
        Lfm2MoeConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=len(LAYER_TYPES),
            layer_types=LAYER_TYPES,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            moe_intermediate_size=48,
            num_experts=4,
            num_experts_per_tok=2,
            num_dense_layers=len(DENSE_FFN_LAYERS),
            conv_L_cache=3,
            norm_topk_prob=True,
        ),
        "Lfm2MoeForCausalLM",
        Lfm2MoeArchitectureAdapter,
    )


def _fold(bridge):
    """Fold layer norms only, so any logit movement is the fold's own doing."""
    bridge.process_weights(
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )


def _norm_weights(bridge):
    return {
        f"blocks.{i}.{name}": _unwrap(getattr(block, name)).weight.detach().clone()
        for i, block in enumerate(bridge.blocks)
        for name in ("ln1", "ln2")
    }


def _assert_all_identity(norms):
    unfolded = {
        key: (float(w.min()), float(w.max()))
        for key, w in norms.items()
        if not torch.allclose(w, torch.ones_like(w), atol=1e-6)
    }
    assert not unfolded, f"norms left unfolded: {unfolded}"


class TestLfm2FoldLn:
    def test_logits_invariant(self, lfm2_bridge):
        with torch.no_grad():
            before = lfm2_bridge(TOKENS).clone()
        _fold(lfm2_bridge)
        with torch.no_grad():
            after = lfm2_bridge(TOKENS)
        torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)

    def test_every_norm_reaches_identity(self, lfm2_bridge):
        _fold(lfm2_bridge)
        _assert_all_identity(_norm_weights(lfm2_bridge))

    def test_conv_in_projection_absorbs_the_gain(self, lfm2_bridge):
        gains = {i: _norm_weights(lfm2_bridge)[f"blocks.{i}.ln1"] for i in CONV_LAYERS}
        before = {
            i: _unwrap(_unwrap(lfm2_bridge.blocks[i].conv).in_proj).weight.detach().clone()
            for i in CONV_LAYERS
        }
        _fold(lfm2_bridge)
        for i in CONV_LAYERS:
            folded = _unwrap(_unwrap(lfm2_bridge.blocks[i].conv).in_proj).weight
            torch.testing.assert_close(folded, before[i] * gains[i][None, :], atol=1e-5, rtol=1e-5)


class TestLfm2MoeFoldLn:
    def test_logits_invariant(self, lfm2_moe_bridge):
        with torch.no_grad():
            before = lfm2_moe_bridge(TOKENS).clone()
        _fold(lfm2_moe_bridge)
        with torch.no_grad():
            after = lfm2_moe_bridge(TOKENS)
        torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)

    def test_every_norm_reaches_identity(self, lfm2_moe_bridge):
        _fold(lfm2_moe_bridge)
        _assert_all_identity(_norm_weights(lfm2_moe_bridge))

    def test_dense_ffn_layer_absorbs_the_gain(self, lfm2_moe_bridge):
        layer = DENSE_FFN_LAYERS[0]
        gain = _norm_weights(lfm2_moe_bridge)[f"blocks.{layer}.ln2"]
        feed_forward = _unwrap(lfm2_moe_bridge.blocks[layer].mlp)
        before = {
            name: _unwrap(getattr(feed_forward, name)).weight.detach().clone()
            for name in ("w1", "w3")
        }
        _fold(lfm2_moe_bridge)
        for name, original in before.items():
            folded = _unwrap(getattr(feed_forward, name)).weight
            torch.testing.assert_close(folded, original * gain[None, :], atol=1e-5, rtol=1e-5)

    def test_sparse_router_and_experts_absorb_the_gain(self, lfm2_moe_bridge):
        gains = {i: _norm_weights(lfm2_moe_bridge)[f"blocks.{i}.ln2"] for i in SPARSE_FFN_LAYERS}
        feed_forwards = {i: _unwrap(lfm2_moe_bridge.blocks[i].mlp) for i in SPARSE_FFN_LAYERS}
        before = {
            i: (
                ff.experts.gate_up_proj.detach().clone(),
                _unwrap(ff.gate).weight.detach().clone(),
                ff.experts.down_proj.detach().clone(),
            )
            for i, ff in feed_forwards.items()
        }
        _fold(lfm2_moe_bridge)
        for i, ff in feed_forwards.items():
            gate_up, router, down = before[i]
            torch.testing.assert_close(
                ff.experts.gate_up_proj, gate_up * gains[i], atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                _unwrap(ff.gate).weight, router * gains[i][None, :], atol=1e-5, rtol=1e-5
            )
            # down_proj reads the expert intermediate, not the norm — folding into it
            # would double-count the gain.
            torch.testing.assert_close(ff.experts.down_proj, down, atol=0.0, rtol=0.0)

    def test_fold_ln_false_leaves_sparse_norms_alone(self, lfm2_moe_bridge):
        before = _norm_weights(lfm2_moe_bridge)
        lfm2_moe_bridge.process_weights(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )
        after = _norm_weights(lfm2_moe_bridge)
        for key, weight in before.items():
            torch.testing.assert_close(after[key], weight, atol=0.0, rtol=0.0)

    def test_refolding_is_a_no_op(self, lfm2_moe_bridge):
        _fold(lfm2_moe_bridge)
        with torch.no_grad():
            once = lfm2_moe_bridge(TOKENS).clone()
        _fold(lfm2_moe_bridge)
        with torch.no_grad():
            twice = lfm2_moe_bridge(TOKENS)
        torch.testing.assert_close(twice, once, atol=1e-4, rtol=1e-4)
        _assert_all_identity(_norm_weights(lfm2_moe_bridge))
