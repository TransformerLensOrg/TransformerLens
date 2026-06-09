"""Unit tests for :class:`SGLangDriver`: orchestration of ``collective_rpc`` for
control, ``engine.generate`` for the forward, and :class:`CapturePuller` for
tensor collection. Engine + puller are mocked so no SGLang / GPU needed."""
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


def _stub_engine_and_puller(
    token_id: int = 7, captures: list[dict] | None = None
) -> tuple[MagicMock, MagicMock]:
    """Coupled engine+puller stubs: ``engine.generate`` enqueues ``captures`` so
    ``puller.drain`` returns them on the post-generate drain (not the pre-drain).
    Models the real worker behavior — hooks fire during generate."""
    queue: list[dict] = []
    engine = MagicMock()

    def _generate(*args, **kwargs):
        queue.extend(captures or [])
        return [
            {
                "output_ids": [token_id],
                "meta_info": {"output_top_logprobs": [[(-0.1, token_id, "A")]]},
            }
        ]

    engine.generate.side_effect = _generate

    puller = MagicMock()

    def _drain(timeout_ms: int = 0):
        out = list(queue)
        queue.clear()
        return out

    puller.drain.side_effect = _drain
    return engine, puller


def _stub_engine(token_id: int = 7) -> MagicMock:
    engine, _ = _stub_engine_and_puller(token_id=token_id)
    return engine


def _stub_puller(messages: list[dict] | None = None) -> MagicMock:
    """Standalone puller that always returns ``messages`` on the next drain."""
    puller = MagicMock()
    queue = list(messages or [])

    def _drain(timeout_ms: int = 0):
        out = list(queue)
        queue.clear()
        return out

    puller.drain.side_effect = _drain
    return puller


def _driver(engine: Any | None = None, puller: Any | None = None) -> SGLangDriver:
    return SGLangDriver(
        engine=engine or _stub_engine(),
        adapter=_adapter(),
        tokenizer=None,
        overlay=DecoderOnlyOverlay(),
        hf_config=_hf_config(),
        max_num_batched_tokens=128,
        puller=puller or _stub_puller(),
    )


class TestConstruction:
    def test_supported_hook_points_match_overlay(self):
        d = _driver()
        # 2 module-level + 3 per-layer * 2 layers = 8.
        assert len(d.supported_hook_points) == 8
        assert "blocks.0.hook_out" in d.supported_hook_points

    def test_non_fireable_expands_layer_templates(self):
        d = _driver()
        assert "blocks.0.attn.hook_pattern" in d.non_fireable_hook_points
        assert "blocks.1.attn.hook_attn_scores" in d.non_fireable_hook_points
        assert "unembed.hook_out" in d.non_fireable_hook_points


class TestNormalizeInputIds:
    def test_tensor_to_list(self):
        assert _driver()._normalize_input_ids(torch.tensor([1, 2, 3])) == [1, 2, 3]

    def test_nested_singleton_unwrapped(self):
        assert _driver()._normalize_input_ids([[1, 2, 3]]) == [1, 2, 3]

    def test_batch_size_above_one_raises(self):
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            _driver()._normalize_input_ids([[1, 2], [3, 4]])


class TestValidateInterventions:
    def test_callable_rejected(self):
        with pytest.raises(NotImplementedError, match="dict"):
            _driver()._validate_interventions({"blocks.0.hook_out": lambda x: x})

    def test_unknown_hook_rejected(self):
        with pytest.raises(ValueError, match="not in supported_hook_points"):
            _driver()._validate_interventions({"blocks.99.hook_out": {"op": "suppress"}})

    def test_unsupported_op_rejected(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _driver()._validate_interventions({"blocks.0.hook_out": {"op": "rotate"}})

    def test_scale_requires_factor(self):
        with pytest.raises(ValueError, match="'factor'"):
            _driver()._validate_interventions({"blocks.0.hook_out": {"op": "scale"}})

    def test_set_requires_value(self):
        with pytest.raises(ValueError, match="requires 'value'"):
            _driver()._validate_interventions({"blocks.0.hook_out": {"op": "set"}})


class TestForwardDispatch:
    """The orchestration: set_interventions → drain stale → set_capture_enabled →
    generate → set_capture_enabled(False) → collect."""

    def test_rpc_call_sequence(self):
        engine = _stub_engine()
        puller = _stub_puller([{"name": "blocks.0.hook_out", "tensor": torch.full((3, 8), 0.5)}])
        d = _driver(engine=engine, puller=puller)

        d.forward(input_ids=torch.tensor([1, 2, 3]), capture=("blocks.0.hook_out",))

        # Order: tl_set_interventions, tl_set_capture_enabled(True), generate, tl_set_capture_enabled(False).
        rpc_methods = [c.args[0] for c in engine.collective_rpc.call_args_list]
        assert rpc_methods == [
            "tl_set_interventions",
            "tl_set_capture_enabled",
            "tl_set_capture_enabled",
        ]
        # The two capture-enabled toggles: True then False.
        enabled_values = [c.kwargs.get("enabled") for c in engine.collective_rpc.call_args_list[1:]]
        assert enabled_values == [True, False]

    def test_captures_collected_from_puller(self):
        captures = [{"name": "blocks.0.hook_out", "tensor": torch.full((3, 8), 0.5)}]
        engine, puller = _stub_engine_and_puller(captures=captures)
        d = _driver(engine=engine, puller=puller)

        result = d.forward(input_ids=torch.tensor([1, 2, 3]), capture=("blocks.0.hook_out",))
        # Driver adds the batch dim.
        assert result.captured["blocks.0.hook_out"].shape == (1, 3, 8)

    def test_first_wins_on_repeated_name(self):
        """If two messages with the same name leak through, keep the first.
        Mirrors the inspect provider's behavior; protects against warmup straggler
        overwriting the real prefill capture."""
        prefill = {"name": "blocks.0.hook_out", "tensor": torch.full((3, 8), 1.0)}
        straggler = {"name": "blocks.0.hook_out", "tensor": torch.full((3, 8), 99.0)}
        engine, puller = _stub_engine_and_puller(captures=[prefill, straggler])
        d = _driver(engine=engine, puller=puller)
        result = d.forward(input_ids=torch.tensor([1, 2, 3]), capture=("blocks.0.hook_out",))
        # First-wins: prefill (1.0), not straggler (99.0).
        assert result.captured["blocks.0.hook_out"][0, 0, 0].item() == 1.0

    def test_capture_filters_to_wanted_names(self):
        captures = [
            {"name": "blocks.0.hook_out", "tensor": torch.zeros(3, 8)},
            {"name": "blocks.1.hook_out", "tensor": torch.ones(3, 8)},
        ]
        engine, puller = _stub_engine_and_puller(captures=captures)
        d = _driver(engine=engine, puller=puller)
        result = d.forward(
            input_ids=torch.tensor([1, 2, 3]),
            capture=("blocks.0.hook_out",),
        )
        # Only the wanted name shows up; the other is filtered.
        assert "blocks.0.hook_out" in result.captured
        assert "blocks.1.hook_out" not in result.captured

    def test_capture_disabled_even_on_generate_exception(self):
        engine = _stub_engine()
        engine.generate.side_effect = RuntimeError("boom")
        d = _driver(engine=engine, puller=_stub_puller())
        with pytest.raises(RuntimeError, match="boom"):
            d.forward(input_ids=torch.tensor([1, 2]))
        # The disable-capture RPC fires in the finally block.
        rpc_calls = engine.collective_rpc.call_args_list
        last = rpc_calls[-1]
        assert last.args[0] == "tl_set_capture_enabled"
        assert last.kwargs["enabled"] is False

    def test_multi_token_generate_rejected(self):
        with pytest.raises(NotImplementedError, match="max_new_tokens=1"):
            _driver().forward(input_ids=torch.tensor([1, 2]), max_new_tokens=2)

    def test_prompt_over_capacity_rejected(self):
        d = _driver()
        d._max_num_batched_tokens = 4
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            d.forward(input_ids=torch.tensor(list(range(10))))

    def test_input_ids_required(self):
        with pytest.raises(ValueError, match="requires input_ids"):
            _driver().forward()


class TestGetParam:
    """``get_param`` dispatches ``tl_get_param`` over collective_rpc and drains the
    PULL socket for the ``{"_param": dotted_name, "tensor": ...}`` response."""

    def test_returns_tensor_when_param_message_arrives(self):
        engine = _stub_engine()
        weight = torch.arange(8.0).reshape(2, 4)
        puller = _stub_puller([{"_param": "model.norm.weight", "tensor": weight}])
        d = _driver(engine=engine, puller=puller)
        out = d.get_param("model.norm.weight")
        assert out is not None and torch.equal(out, weight)
        # RPC dispatched with the channel + dotted_name.
        rpc_call = engine.collective_rpc.call_args
        assert rpc_call.args[0] == "tl_get_param"
        assert rpc_call.kwargs["dotted_name"] == "model.norm.weight"

    def test_returns_none_when_no_matching_message(self):
        d = _driver(puller=_stub_puller([]))
        assert d.get_param("model.norm.weight") is None

    def test_returns_none_after_close(self):
        d = _driver()
        d._engine = None
        assert d.get_param("anything") is None

    def test_ignores_non_param_messages(self):
        """Any leftover capture messages on the channel get drained and discarded."""
        msgs = [
            {"name": "blocks.0.hook_out", "tensor": torch.zeros(2, 4)},  # leftover capture
            {"_param": "model.norm.weight", "tensor": torch.ones(4)},
        ]
        d = _driver(puller=_stub_puller(msgs))
        out = d.get_param("model.norm.weight")
        assert out is not None and torch.equal(out, torch.ones(4))


class TestClose:
    def test_clear_state_rpc_then_shutdown(self):
        engine = _stub_engine()
        puller = _stub_puller()
        d = _driver(engine=engine, puller=puller)
        d.close()
        # close() runs tl_clear_state.
        methods = [c.args[0] for c in engine.collective_rpc.call_args_list]
        assert "tl_clear_state" in methods
        # And the engine's shutdown() was invoked.
        engine.shutdown.assert_called_once()
        # And the puller closed.
        puller.close.assert_called_once()


class TestSynthesizeLogits:
    def test_top_logprobs_fill_gen_position(self):
        outputs = [
            {
                "output_ids": [3],
                "meta_info": {
                    "output_top_logprobs": [[(-0.5, 3, "C"), (-1.0, 5, "E"), (-2.0, 7, "G")]]
                },
            }
        ]
        logits = SGLangDriver._synthesize_logits(outputs, n_tokens=4, d_vocab=16)
        assert logits.shape == (1, 4, 16)
        assert torch.isinf(logits[0, 0]).all()
        assert logits[0, -1, 3].item() == pytest.approx(-0.5)

    def test_falls_back_to_one_hot_when_no_top_logprobs(self):
        outputs = [{"output_ids": [4], "meta_info": {}}]
        logits = SGLangDriver._synthesize_logits(outputs, n_tokens=3, d_vocab=16)
        assert logits[0, -1, 4].item() == 0.0
