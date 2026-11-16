"""boot()'s device for a pre-loaded ``hf_model``: applied when asked for, reported as it is."""

import logging

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge

TINY = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64)


def _tiny_neox():
    return AutoModelForCausalLM.from_config(AutoConfig.for_model("gpt_neox", **TINY))


def _param_devices(model):
    return {p.device.type for p in model.parameters()}


def test_device_defaults_to_hf_model_device(monkeypatch):
    # Without device= boot never moves hf_model, so cfg.device must be the model's device,
    # not get_device()'s.
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.transformers.source.get_device", lambda: "cuda"
    )
    with torch.device("meta"):
        hf_model = _tiny_neox()

    bridge = TransformerBridge.boot_transformers("EleutherAI/pythia-70m", hf_model=hf_model)

    assert bridge.cfg.device == "meta"


def test_explicit_device_is_applied_to_hf_model():
    # meta is the one target besides cpu that every machine can move to, so the move shows.
    hf_model = _tiny_neox()

    bridge = TransformerBridge.boot_transformers(
        "EleutherAI/pythia-70m", hf_model=hf_model, device="meta"
    )

    assert _param_devices(hf_model) == {"meta"}
    assert bridge.cfg.device == "meta"


def test_cfg_device_reports_the_parameters_when_a_request_is_not_applied(caplog):
    # Meta parameters have nothing to move, so the request cannot land.
    with torch.device("meta"):
        hf_model = _tiny_neox()

    with caplog.at_level(logging.WARNING):
        bridge = TransformerBridge.boot_transformers(
            "EleutherAI/pythia-70m", hf_model=hf_model, device="cpu"
        )

    assert bridge.cfg.device == "meta"
    assert "device=cpu was requested" in caplog.text


@pytest.mark.parametrize("owner", ["device_map", "quantizer"])
def test_owned_placement_rejects_a_conflicting_device_and_accepts_its_own(owner):
    hf_model = _tiny_neox()
    if owner == "device_map":
        hf_model.hf_device_map = {"": "cpu"}
    else:
        hf_model.config.quantization_config = {"quant_method": "mxfp4"}

    with pytest.raises(ValueError, match="does not move dispatched or quantized"):
        TransformerBridge.boot_transformers(
            "EleutherAI/pythia-70m", hf_model=hf_model, device="cuda"
        )

    bridge = TransformerBridge.boot_transformers(
        "EleutherAI/pythia-70m", hf_model=hf_model, device="cpu"
    )
    assert _param_devices(hf_model) == {"cpu"}
    assert bridge.cfg.device == "cpu"


def test_no_weights_build_reports_the_cpu_it_was_built_on(monkeypatch, caplog):
    # from_config builds on the CPU and nothing applies device there, so cfg must not echo it.
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.transformers.source.get_device", lambda: "cuda"
    )
    with caplog.at_level(logging.WARNING):
        by_default = TransformerBridge.boot_transformers(
            "EleutherAI/pythia-70m", load_weights=False, hf_config_overrides=TINY
        )
        assert by_default.cfg.device == "cpu"
        assert "was requested" not in caplog.text
        explicit = TransformerBridge.boot_transformers(
            "EleutherAI/pythia-70m", load_weights=False, hf_config_overrides=TINY, device="meta"
        )

    assert explicit.cfg.device == "cpu"
    assert "device=meta was requested" in caplog.text
