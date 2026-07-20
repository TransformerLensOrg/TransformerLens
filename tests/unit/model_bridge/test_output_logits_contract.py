"""Output-logit transform contracts shared by Bridge analysis tools."""

from types import SimpleNamespace

import pytest
import torch

from transformer_lens.model_bridge.sources import build_bridge_config_from_hf
from transformer_lens.model_bridge.supported_architectures.falcon_h1 import (
    FalconH1ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.granite import (
    GraniteArchitectureAdapter,
)


def _text_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "intermediate_size": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("architecture", "field", "value"),
    [
        ("GraniteForCausalLM", "logits_scaling", 4.0),
        ("FalconH1ForCausalLM", "lm_head_multiplier", 0.25),
    ],
)
def test_output_scalars_survive_bridge_config_translation(
    architecture: str, field: str, value: float
) -> None:
    cfg = build_bridge_config_from_hf(
        _text_config(**{field: value}),
        architecture,
        "tiny-output-contract",
        torch.float32,
    )

    assert getattr(cfg, field) == value


def test_recurrent_gemma_softcap_survives_bridge_config_translation() -> None:
    cfg = build_bridge_config_from_hf(
        _text_config(logits_soft_cap=30.0),
        "RecurrentGemmaForCausalLM",
        "tiny-recurrent-gemma-output-contract",
        torch.float32,
    )

    assert cfg.output_logits_soft_cap == 30.0


def test_nested_text_config_softcap_is_used() -> None:
    wrapper = SimpleNamespace(text_config=_text_config(final_logit_softcapping=17.0))
    cfg = build_bridge_config_from_hf(
        wrapper,
        "Gemma4ForConditionalGeneration",
        "tiny-gemma4-output-contract",
        torch.float32,
    )

    assert cfg.output_logits_soft_cap == 17.0


def test_granite_and_falcon_apply_declared_output_scalars() -> None:
    granite_cfg = build_bridge_config_from_hf(
        _text_config(logits_scaling=4.0),
        "GraniteForCausalLM",
        "tiny-granite-output-contract",
        torch.float32,
    )
    falcon_cfg = build_bridge_config_from_hf(
        _text_config(lm_head_multiplier=0.25),
        "FalconH1ForCausalLM",
        "tiny-falcon-h1-output-contract",
        torch.float32,
    )
    logits = torch.tensor([[-8.0, 4.0]])

    torch.testing.assert_close(
        GraniteArchitectureAdapter(granite_cfg).apply_output_logits_transform(logits),
        torch.tensor([[-2.0, 1.0]]),
    )
    torch.testing.assert_close(
        FalconH1ArchitectureAdapter(falcon_cfg).apply_output_logits_transform(logits),
        torch.tensor([[-2.0, 1.0]]),
    )
