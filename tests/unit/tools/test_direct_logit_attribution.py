"""Unit tests for Direct Logit Attribution guards and argument validation.

These exercise the fast-failing checks (argument validation, the Bridge
compatibility-mode requirement, and the hybrid-architecture refusal) without
loading a real model — using a ``spec``-ed mock TransformerBridge so the checks
fire before any forward pass.
"""

from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    LinearBridge,
)
from transformer_lens.tools.analysis import direct_logit_attribution

TOKENS = torch.tensor([[1, 2, 3, 4]])


def _mock_bridge(compatibility_mode=True, layer_types=("attn+mlp",)):
    """A mock TransformerBridge that satisfies isinstance checks."""
    bridge = MagicMock(spec=TransformerBridge)
    bridge.compatibility_mode = compatibility_mode
    bridge.layer_types.return_value = list(layer_types)
    return bridge


def test_invalid_unit_raises():
    bridge = _mock_bridge()
    with pytest.raises(ValueError, match="unit must be one of"):
        direct_logit_attribution(bridge, "hi", answer_tokens=" world", unit="neuron")


def test_missing_answer_tokens_raises():
    bridge = _mock_bridge()
    with pytest.raises(ValueError, match="answer_tokens is required"):
        direct_logit_attribution(bridge, "hi")


def test_requires_compatibility_mode():
    bridge = _mock_bridge(compatibility_mode=False)
    with pytest.raises(ValueError, match="compatibility mode"):
        direct_logit_attribution(bridge, "hi", answer_tokens=" world")


def test_rejects_hybrid_architecture():
    bridge = _mock_bridge(layer_types=("attn+mlp", "mamba+mlp"))
    with pytest.raises(NotImplementedError, match="hybrid"):
        direct_logit_attribution(bridge, "hi", answer_tokens=" world")


def test_direct_logit_attribution_is_exported_from_analysis_package():
    from transformer_lens.tools import analysis

    assert analysis.direct_logit_attribution is direct_logit_attribution


def _native_bridge(compat=True):
    cfg = TransformerBridgeConfig(
        n_layers=2,
        d_model=32,
        d_head=8,
        n_heads=4,
        d_mlp=64,
        d_vocab=16,
        n_ctx=8,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )
    model = TransformerBridge.boot_native(cfg).eval()
    if compat:
        model.enable_compatibility_mode()
    return model


def _forbid_forward(model, monkeypatch, reason="the model was run before the readout-path check"):
    def forward_ran(*args, **kwargs):
        raise AssertionError(reason)

    monkeypatch.setattr(model, "run_with_cache", forward_ran)


class _StackedStreamBlock(BlockBridge):
    hook_out_is_single_residual_stream = False


@pytest.mark.parametrize("compat", [True, False])
def test_rejects_head_transform_component_before_any_forward(compat, monkeypatch):
    model = _native_bridge(compat)
    model.add_module("mlm_head", LinearBridge(name="cls.predictions.transform.dense").eval())
    _forbid_forward(model, monkeypatch)
    with pytest.raises(ValueError, match=r"DLA .* mounts \['mlm_head'\]"):
        direct_logit_attribution(model, TOKENS, answer_tokens=5)


def test_rejects_multi_stream_block_before_any_forward(monkeypatch):
    model = _native_bridge()
    model.blocks[0].__class__ = _StackedStreamBlock
    _forbid_forward(model, monkeypatch)
    with pytest.raises(ValueError, match="DLA requires single-stream"):
        direct_logit_attribution(model, TOKENS, answer_tokens=5)


def test_prompt_path_rejects_a_readout_transform_it_cannot_see():
    model = _native_bridge()
    torch.manual_seed(1)
    dense = torch.nn.Linear(32, 32)
    model.unembed.register_forward_pre_hook(lambda module, args: (torch.tanh(dense(args[0])),))
    with pytest.raises(ValueError, match="differs from the model's own log-probabilities"):
        direct_logit_attribution(model, TOKENS, answer_tokens=5, incorrect_tokens=6)


def _training_flags(model):
    modules = [*model.named_modules(), *model.original_model.named_modules()]
    return [(name, module.training) for name, module in modules]


@pytest.mark.parametrize(
    ("enter_training", "root_training"),
    [
        (lambda model: model.train(), True),
        (lambda model: model.blocks[1].mlp.train(), False),
    ],
    ids=["whole-model", "one-mlp"],
)
def test_prompt_path_refuses_training_mode_before_any_forward(
    monkeypatch, enter_training, root_training
):
    model = _native_bridge()
    enter_training(model)
    flags = _training_flags(model)
    _forbid_forward(model, monkeypatch, "the model was run before the eval-mode check")
    with pytest.raises(ValueError, match=r"direct_logit_attribution\(\) .*model\.eval\(\)"):
        direct_logit_attribution(model, TOKENS, answer_tokens=5)
    assert model.training is root_training
    assert _training_flags(model) == flags


def test_supplied_cache_is_accepted_after_the_model_enters_training_mode(monkeypatch):
    model = _native_bridge()
    _, cache = model.run_with_cache(TOKENS)
    expected = direct_logit_attribution(model, answer_tokens=5, incorrect_tokens=6, cache=cache)
    model.train()
    _forbid_forward(model, monkeypatch, "a supplied cache must not rerun the model")
    result = direct_logit_attribution(model, answer_tokens=5, incorrect_tokens=6, cache=cache)
    assert torch.equal(result.attribution, expected.attribution)
    assert result.labels == expected.labels
