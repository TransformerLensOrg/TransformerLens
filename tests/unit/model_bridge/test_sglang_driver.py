"""Unit tests for ``SGLangDriver``: input normalization, intervention validation,
and RPC dispatch shape. The engine is mocked so no SGLang install is needed; live
end-to-end verification (real SGLang, real GPU) is in the Colab walkthrough.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.sources.sglang.driver import SGLangDriver
from transformer_lens.model_bridge.sources.sglang.overlays import DecoderOnlyOverlay


def _hf_config(n_layers: int = 2, vocab: int = 16) -> SimpleNamespace:
    return SimpleNamespace(hidden_size=8, num_hidden_layers=n_layers, vocab_size=vocab)


def _adapter() -> SimpleNamespace:
    cfg = TransformerBridgeConfig(
        d_model=8,
        d_head=4,
        n_layers=2,
        n_heads=2,
        n_ctx=64,
        d_vocab=16,
        act_fn="gelu",
        normalization_type="LN",
    )
    return SimpleNamespace(cfg=cfg)


def _driver(engine: Any | None = None) -> SGLangDriver:
    return SGLangDriver(
        engine=engine if engine is not None else MagicMock(),
        adapter=_adapter(),
        tokenizer=None,
        overlay=DecoderOnlyOverlay(),
        hf_config=_hf_config(),
        max_num_batched_tokens=128,
    )


class TestConstruction:
    def test_supported_hook_points_match_overlay(self):
        d = _driver()
        # 2 module-level + 3 per-layer * 2 layers = 8 fireable hooks.
        assert len(d.supported_hook_points) == 8
        assert "blocks.0.hook_out" in d.supported_hook_points
        assert "embed.hook_out" in d.supported_hook_points

    def test_non_fireable_expands_layer_templates(self):
        d = _driver()
        # 4 per-layer templates expanded * 2 layers + 1 module-level (unembed).
        assert len(d.non_fireable_hook_points) == 4 * 2 + 1
        assert "blocks.0.attn.hook_pattern" in d.non_fireable_hook_points
        assert "blocks.1.attn.hook_attn_scores" in d.non_fireable_hook_points
        assert "unembed.hook_out" in d.non_fireable_hook_points


class TestNormalizeInputIds:
    def test_tensor_to_list(self):
        d = _driver()
        ids = d._normalize_input_ids(torch.tensor([1, 2, 3]))
        assert ids == [1, 2, 3]

    def test_nested_singleton_unwrapped(self):
        d = _driver()
        ids = d._normalize_input_ids([[1, 2, 3]])
        assert ids == [1, 2, 3]

    def test_batch_size_above_one_raises(self):
        d = _driver()
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            d._normalize_input_ids([[1, 2], [3, 4]])


class TestValidateInterventions:
    def test_callable_rejected(self):
        d = _driver()
        with pytest.raises(NotImplementedError, match="dict"):
            d._validate_interventions({"blocks.0.hook_out": lambda x: x})

    def test_unknown_hook_rejected(self):
        d = _driver()
        with pytest.raises(ValueError, match="not in supported_hook_points"):
            d._validate_interventions({"blocks.99.hook_out": {"op": "suppress"}})

    def test_unsupported_op_rejected(self):
        d = _driver()
        with pytest.raises(ValueError, match="Unsupported"):
            d._validate_interventions({"blocks.0.hook_out": {"op": "rotate"}})

    def test_scale_requires_factor(self):
        d = _driver()
        with pytest.raises(ValueError, match="'factor'"):
            d._validate_interventions({"blocks.0.hook_out": {"op": "scale"}})

    def test_set_requires_value(self):
        d = _driver()
        with pytest.raises(ValueError, match="requires 'value'"):
            d._validate_interventions({"blocks.0.hook_out": {"op": "set"}})

    def test_valid_specs_pass_through(self):
        d = _driver()
        out = d._validate_interventions({"blocks.0.hook_out": {"op": "suppress"}})
        assert out == {"blocks.0.hook_out": {"op": "suppress"}}


class TestForwardDispatch:
    """The forward path: RPC sequence + sampling-params shape."""

    def _engine_with_capture(self, captures: dict | None = None):
        """Mock engine whose ``generate`` returns a stub output. RPC dispatch is
        intercepted by patching the ``rpc`` module in the driver."""
        engine = MagicMock()
        engine.generate.return_value = [
            {"token_ids": [7], "meta_info": {"output_top_logprobs": [[(-0.1, 7, "A")]]}}
        ]
        return engine

    def test_forward_pushes_interventions_then_captures(self, monkeypatch):
        from transformer_lens.model_bridge.sources.sglang import rpc as rpc_module

        # Track every call to the rpc module's methods.
        captures_returned = {"blocks.0.hook_out": torch.full((3, 8), 0.5)}
        calls: list[tuple[str, dict]] = []

        def _set_interventions(engine, specs):
            calls.append(("set_interventions", {"specs": specs}))

        def _reset_flags(engine):
            calls.append(("reset_capture_flags", {}))

        def _read(engine, method, prompt_lens, names):
            calls.append(("read_captures", {"prompt_lens": prompt_lens, "names": names}))
            return captures_returned

        monkeypatch.setattr(rpc_module, "set_interventions", _set_interventions)
        monkeypatch.setattr(rpc_module, "reset_capture_flags", _reset_flags)
        monkeypatch.setattr(rpc_module, "call_with_prompt_lens", _read)

        engine = self._engine_with_capture()
        d = _driver(engine=engine)

        result = d.forward(
            input_ids=torch.tensor([1, 2, 3]),
            capture=("blocks.0.hook_out",),
        )

        # RPC sequence: set_interventions → reset_capture_flags → generate → read.
        rpc_only = [name for name, _ in calls]
        assert rpc_only == ["set_interventions", "reset_capture_flags", "read_captures"]
        # read_captures got the right prompt_lens / names.
        assert calls[2][1] == {"prompt_lens": [3], "names": ["blocks.0.hook_out"]}
        # Captured tensors carry the batch dim added by the driver.
        assert result.captured["blocks.0.hook_out"].shape == (1, 3, 8)

    def test_multi_token_generate_rejected(self):
        d = _driver()
        with pytest.raises(NotImplementedError, match="max_new_tokens=1"):
            d.forward(input_ids=torch.tensor([1, 2]), max_new_tokens=2)

    def test_prompt_over_capacity_rejected(self):
        d = _driver()
        d._max_num_batched_tokens = 4
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            d.forward(input_ids=torch.tensor(list(range(10))))

    def test_input_ids_required(self):
        d = _driver()
        with pytest.raises(ValueError, match="requires input_ids"):
            d.forward()


class TestSynthesizeLogits:
    def test_top_logprobs_fill_gen_position(self):
        outputs = [
            {
                "token_ids": [3],
                "meta_info": {
                    "output_top_logprobs": [[(-0.5, 3, "C"), (-1.0, 5, "E"), (-2.0, 7, "G")]]
                },
            }
        ]
        logits = SGLangDriver._synthesize_logits(outputs, n_tokens=4, d_vocab=16)
        assert logits.shape == (1, 4, 16)
        # Earlier positions are -inf.
        assert torch.isinf(logits[0, 0]).all() and (logits[0, 0] < 0).all()
        # Last position carries the three top logprobs.
        assert logits[0, -1, 3].item() == pytest.approx(-0.5)
        assert logits[0, -1, 5].item() == pytest.approx(-1.0)
        assert logits[0, -1, 7].item() == pytest.approx(-2.0)

    def test_falls_back_to_one_hot_when_no_top_logprobs(self):
        outputs = [{"token_ids": [4], "meta_info": {}}]
        logits = SGLangDriver._synthesize_logits(outputs, n_tokens=3, d_vocab=16)
        # Only the generated token id is finite at the last position.
        assert logits[0, -1, 4].item() == 0.0
        # All others at the last position stay -inf.
        mask = torch.ones(16, dtype=torch.bool)
        mask[4] = False
        assert torch.isinf(logits[0, -1, mask]).all()
