"""Minimal one-block TransformerBridge over a stub torch model + logits-returning driver."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from tests.mocks.architecture_adapter import MockArchitectureAdapter
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.driver_protocol import ForwardResult
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    NormalizationBridge,
)
from transformer_lens.model_bridge.sources._driver_base import DriverBase


def mock_bridge_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=4,
        d_head=2,
        n_layers=1,
        n_ctx=8,
        n_heads=2,
        d_vocab=16,
        d_mlp=8,
        architecture="Mock",
    )


class LogitsDriver(DriverBase):
    """Returns real logits so post-hook forward paths complete."""

    def __init__(self, model, cfg, tokenizer=None):
        super().__init__(cfg, tokenizer)
        # forward()'s encoder-decoder probe reads bridge.original_model.
        self.underlying_model = model

    def forward(
        self,
        input_ids=None,
        *,
        capture=(),
        intervene=None,
        max_new_tokens=1,
        return_logits=True,
        **kw,
    ):
        return ForwardResult(logits=torch.randn(1, 3, 16))


def build_mock_logits_bridge(driver_cls=LogitsDriver) -> TransformerBridge:
    """Encoder-shaped stub bridge (ln_final + one block with ln1/attn)."""
    model = nn.Module()
    model.final_norm = nn.LayerNorm(10)
    model.encoder = nn.Module()
    model.encoder.layers = nn.ModuleList([nn.Module() for _ in range(1)])
    model.encoder.layers[0].norm1 = nn.LayerNorm(10)
    model.encoder.layers[0].self_attn = nn.Module()

    adapter = MockArchitectureAdapter()
    adapter.component_mapping = {
        "ln_final": NormalizationBridge(name="final_norm", config={}),
        "blocks": BlockBridge(
            name="encoder.layers",
            submodules={
                "ln1": NormalizationBridge(name="norm1", config={}),
                "attn": AttentionBridge(name="self_attn", config=SimpleNamespace(n_heads=1)),
            },
        ),
    }
    driver = driver_cls(model, mock_bridge_cfg(), tokenizer=None)
    return TransformerBridge(model, adapter, tokenizer=MagicMock(), driver=driver)
