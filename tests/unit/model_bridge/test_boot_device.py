"""boot()'s device for a pre-loaded ``hf_model``."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge


def test_device_defaults_to_hf_model_device(monkeypatch):
    # boot never moves hf_model, so cfg.device must be the model's device, not get_device()'s.
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.transformers.source.get_device", lambda: "cuda"
    )
    hf_config = AutoConfig.for_model(
        "gpt_neox", hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64
    )
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(hf_config)

    bridge = TransformerBridge.boot_transformers("EleutherAI/pythia-70m", hf_model=hf_model)

    assert bridge.cfg.device == "meta"
