"""DLA and logit-lens contracts on SSM / hybrid bridges.

Component and head attributions must sum to the logit difference on every hybrid,
since neither reads ``resid_mid``. Only the ``resid_mid`` readers (DLA
``unit="layer"``, logit-lens ``incl_mid=True``) refuse, and only on blocks without one.
"""

from typing import Any, Callable, Dict

import pytest
import torch

from tests.integration.model_bridge.test_ssm_hybrid_resid_decomposition import (
    _QWEN3_5_AVAILABLE,
    TOKENS,
    _boot,
    _granite_dense,
    _granite_moe,
    _mamba2,
    _nemotronh,
    _qwen35,
)
from transformer_lens.tools.analysis import direct_logit_attribution, logit_lens
from transformer_lens.tools.analysis.logit_lens import logit_readout

ATOL = 1e-5
ANSWER = torch.tensor([3, 3])
WRONG = torch.tensor([5, 5])

needs_qwen35 = pytest.mark.skipif(not _QWEN3_5_AVAILABLE, reason="needs Qwen3.5 in transformers")


def _falcon_h1():
    """Falcon-H1 runs Mamba beside attention in every block (MHA, so compat mode folds)."""
    from transformers import AutoModelForCausalLM
    from transformers.models.falcon_h1 import FalconH1Config

    torch.manual_seed(0)
    c = FalconH1Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        mamba_d_ssm=64,
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=8,
    )
    c.architectures = ["FalconH1ForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "FalconH1ForCausalLM")


_FACTORIES: Dict[str, Callable[[], Any]] = {
    "qwen3_5": _qwen35,
    "mamba2": _mamba2,
    "nemotron_h": _nemotronh,
    "granite_dense": _granite_dense,
    "granite_moe": _granite_moe,
    "falcon_h1": _falcon_h1,
}


def _compat(name):
    bridge = _FACTORIES[name]()
    bridge.enable_compatibility_mode(disable_warnings=True)
    return bridge.eval()


def _logit_diff(bridge):
    """Residual-stream share of the answer-minus-wrong logit at the last position."""
    with torch.no_grad():
        logits = bridge(TOKENS)
    diff = logits[:, -1, 3] - logits[:, -1, 5]
    b_U = getattr(bridge, "b_U", None)
    return diff if b_U is None else diff - (b_U[3] - b_U[5])


def _dla(bridge, unit):
    return direct_logit_attribution(
        bridge, TOKENS, answer_tokens=ANSWER, incorrect_tokens=WRONG, unit=unit
    )


@pytest.mark.parametrize(
    "name", [pytest.param("qwen3_5", marks=needs_qwen35), "mamba2", "nemotron_h"]
)
def test_component_attribution_sums_to_logit_diff(name):
    bridge = _compat(name)
    result = _dla(bridge, "component")
    torch.testing.assert_close(result.attribution.sum(0), _logit_diff(bridge), atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    "name", [pytest.param("qwen3_5", marks=needs_qwen35), "granite_dense", "granite_moe"]
)
def test_head_attribution_sums_to_logit_diff(name):
    bridge = _compat(name)
    result = _dla(bridge, "head")
    assert result.labels[-1] == "remainder"
    torch.testing.assert_close(result.attribution.sum(0), _logit_diff(bridge), atol=ATOL, rtol=0)


def test_component_attribution_covers_blocks_without_hook_mlp_out():
    """GraniteMoeHybrid with experts (the shipped Granite-4.0-H layout) sums its MoE and shared
    MLP inline, so no hook_mlp_out exists; the MLP-side row must still be attributed."""
    bridge = _compat("granite_moe")
    assert not any(name.endswith("hook_mlp_out") for name in bridge.hook_dict)
    result = _dla(bridge, "component")
    assert [f"{l}_mlp_out" for l in range(bridge.cfg.n_layers)] == [
        label for label in result.labels if label.endswith("_mlp_out")
    ]
    torch.testing.assert_close(result.attribution.sum(0), _logit_diff(bridge), atol=ATOL, rtol=0)


_WITH_RESID_MID = [pytest.param("qwen3_5", marks=needs_qwen35), "granite_dense", "granite_moe"]


@pytest.mark.parametrize("name", _WITH_RESID_MID)
def test_layer_attribution_mid_rows_add_the_attention_slot(name):
    """``{l}_mid`` must read the residual right after block l's attention-slot write."""
    bridge = _compat(name)
    by_layer = _dla(bridge, "layer")
    by_component = _dla(bridge, "component")
    for layer in range(bridge.cfg.n_layers):
        step = (
            by_layer.attribution[by_layer.labels.index(f"{layer}_mid")]
            - by_layer.attribution[by_layer.labels.index(f"{layer}_pre")]
        )
        slot = [
            row
            for row, label in zip(by_component.attribution, by_component.labels)
            if label.startswith(f"{layer}_") and not label.endswith("_mlp_out")
        ]
        assert slot, by_component.labels
        torch.testing.assert_close(step, torch.stack(slot).sum(0), atol=ATOL, rtol=0)


def test_layer_attribution_refuses_mixer_only_blocks_before_any_forward(monkeypatch):
    bridge = _compat("mamba2")

    def no_forward(*args, **kwargs):
        raise AssertionError("the forward pass ran before the resid_mid check")

    monkeypatch.setattr(bridge, "run_with_cache", no_forward)
    with pytest.raises(ValueError, match=r"blocks \[0, 1\]"):
        _dla(bridge, "layer")


def test_falcon_h1_component_attribution_is_complete_or_refused():
    """A parallel Mamba branch must be attributed, or DLA must refuse; never a short sum."""
    bridge = _compat("falcon_h1")
    try:
        result = _dla(bridge, "component")
    except ValueError as err:
        assert "mamba output is not decomposed yet" in str(err)
        return
    torch.testing.assert_close(result.attribution.sum(0), _logit_diff(bridge), atol=ATOL, rtol=0)


@pytest.mark.parametrize("name", _WITH_RESID_MID)
def test_logit_lens_incl_mid_on_hybrid_with_resid_mid(name):
    bridge = _FACTORIES[name]().eval()
    n_layers = bridge.cfg.n_layers
    with torch.no_grad():
        ref = bridge(TOKENS)
    _, cache = bridge.run_with_cache(TOKENS)
    for result in (
        logit_lens(bridge, TOKENS, incl_mid=True, positions=None),
        logit_lens(bridge, cache=cache, incl_mid=True, positions=None),
    ):
        assert len(result.labels) == 2 * n_layers + 1
        torch.testing.assert_close(result.values[-1], ref, atol=1e-4, rtol=1e-4)
        for layer in range(n_layers):
            # The mid row must be the readout of resid_mid itself, not of a neighbouring hook.
            mid = logit_readout(bridge, cache[f"blocks.{layer}.hook_resid_mid"]).values
            torch.testing.assert_close(
                result.values[result.labels.index(f"{layer}_mid")], mid, atol=1e-4, rtol=1e-4
            )
