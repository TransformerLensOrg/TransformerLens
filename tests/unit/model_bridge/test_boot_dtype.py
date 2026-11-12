"""boot()'s dtype for a pre-loaded ``hf_model``: applied when asked for, reported as it is."""

import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge


def _tiny_neox(**kwargs):
    hf_config = AutoConfig.for_model(
        "gpt_neox", hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64
    )
    return AutoModelForCausalLM.from_config(hf_config, **kwargs)


def _param_dtypes(model):
    return {p.dtype for p in model.parameters() if p.is_floating_point()}


def test_dtype_defaults_to_hf_model_dtype():
    hf_model = _tiny_neox(dtype=torch.bfloat16)

    bridge = TransformerBridge.boot_transformers("EleutherAI/pythia-70m", hf_model=hf_model)

    assert bridge.cfg.dtype is torch.bfloat16
    assert _param_dtypes(bridge.original_model) == {torch.bfloat16}


def test_explicit_dtype_is_applied_to_hf_model():
    hf_model = _tiny_neox()

    bridge = TransformerBridge.boot_transformers(
        "EleutherAI/pythia-70m", hf_model=hf_model, dtype=torch.float16
    )

    assert _param_dtypes(bridge.original_model) == {torch.float16}
    assert bridge.cfg.dtype is torch.float16


def test_cfg_dtype_reports_the_parameters_when_a_request_is_not_applied(caplog):
    # Meta parameters are never cast (accelerate's offload placeholders live there too),
    # so the request cannot take effect; cfg must say what the model is, and say why.
    with torch.device("meta"):
        hf_model = _tiny_neox()

    with caplog.at_level(logging.WARNING):
        bridge = TransformerBridge.boot_transformers(
            "EleutherAI/pythia-70m", hf_model=hf_model, dtype=torch.float16
        )

    assert bridge.cfg.dtype is torch.float32
    assert "dtype=torch.float16 was requested" in caplog.text
