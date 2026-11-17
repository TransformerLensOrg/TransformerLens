"""``place_on_device`` moves what the bridge may and reports what it may not."""

import logging
from types import SimpleNamespace

import pytest
from torch import nn

from transformer_lens.utilities.multi_gpu import (
    first_parameter_device,
    place_on_device,
    report_device,
)


def _devices(model):
    return {p.device.type for p in model.parameters()}


def test_moves_a_model_nobody_else_owns():
    model = nn.Linear(2, 2)

    assert place_on_device(model, "meta") == "meta"
    assert _devices(model) == {"meta"}


def test_dispatched_model_keeps_its_map_and_an_index_less_request_fits_any_index():
    model = nn.Linear(2, 2)
    model.hf_device_map = {"": 1}

    assert place_on_device(model, "cuda") == "cuda:1"
    with pytest.raises(ValueError, match="device_map"):
        place_on_device(model, "cuda:0")
    with pytest.raises(ValueError, match="device_map"):
        place_on_device(model, "cpu")
    assert _devices(model) == {"cpu"}


def test_quantized_model_keeps_its_placement():
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4"))

    assert place_on_device(model, "cpu") == "cpu"
    with pytest.raises(ValueError, match="quantizer"):
        place_on_device(model, "meta")
    assert _devices(model) == {"cpu"}


def test_owned_model_without_parameters_cannot_contradict_a_request():
    model = nn.Module()
    model.config = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4"))

    assert place_on_device(model, "cuda") == "cuda"


def test_meta_parameters_are_reported_not_moved(caplog):
    model = nn.Linear(2, 2).to("meta")

    with caplog.at_level(logging.WARNING):
        assert place_on_device(model, "cpu") == "meta"

    assert "device=cpu was requested" in caplog.text


def test_report_device_without_parameters_cannot_contradict_a_request():
    assert first_parameter_device(object()) is None
    assert report_device(nn.Module(), None) == "cpu"
    assert report_device(nn.Module(), "cuda") == "cuda"
