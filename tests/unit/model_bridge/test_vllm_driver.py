"""Unit tests for VLLMDriver — mocks vLLM's LLM so no vllm package install needed.

GPU integration tests (real LLM, real captures) live separately as a Colab
notebook; the compiled-graph path can't be reached from mocked unit tests.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.driver_protocol import (
    Driver,
    ForwardResult,
    validate_driver,
)
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources.vllm.driver import VLLMDriver


def _hf_config(num_hidden_layers: int = 2, hidden_size: int = 4, vocab_size: int = 16) -> Any:
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )


def _overlay(specs=None, nonfiring=None):
    """Minimal overlay stand-in — capture_specs + nonfiring_hooks."""
    return SimpleNamespace(
        capture_specs=lambda hf_config: specs
        or {
            "embed.hook_out": ("model.embed_tokens", hf_config.hidden_size),
            "blocks.0.hook_out": ("model.layers.0", hf_config.hidden_size),
            "blocks.1.hook_out": ("model.layers.1", hf_config.hidden_size),
            "unembed.hook_out": ("lm_head", hf_config.vocab_size),
        },
        nonfiring_hooks=lambda: nonfiring
        or [
            "blocks.{i}.attn.hook_pattern",
            "blocks.{i}.attn.hook_attn_scores",
        ],
    )


def _cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=4,
        d_head=2,
        n_layers=2,
        n_ctx=8,
        n_heads=2,
        d_vocab=16,
        d_mlp=8,
        architecture="LlamaForCausalLM",
    )


def _adapter() -> ArchitectureAdapter:
    adapter = ArchitectureAdapter(_cfg())
    adapter.component_mapping = {}
    return adapter


def _fake_request_output(generated_token=None, top_logprobs=None):
    """Build a vLLM RequestOutput-shaped mock for _synthesize_logits."""
    completion = MagicMock()
    completion.token_ids = [generated_token] if generated_token is not None else []
    completion.logprobs = (
        [{tid: MagicMock(logprob=lp) for tid, lp in top_logprobs.items()}]
        if top_logprobs is not None
        else []
    )
    ro = MagicMock()
    ro.outputs = [completion]
    return ro


def _driver(
    *,
    captures=None,
    hf_config=None,
    max_num_batched_tokens=2048,
    generated_token=None,
    top_logprobs=None,
    enable_position_interventions=False,
) -> VLLMDriver:
    """Build a VLLMDriver. ``captures`` populates llm.collective_rpc; ``generated_token``
    and ``top_logprobs`` populate llm.generate's RequestOutput so _synthesize_logits
    can be exercised on both code paths (logprobs preferred, token_id fallback)."""
    llm = MagicMock()
    if captures is not None:
        llm.collective_rpc = MagicMock(return_value=[captures])
    llm.generate = MagicMock(return_value=[_fake_request_output(generated_token, top_logprobs)])
    return VLLMDriver(
        llm=llm,
        adapter=_adapter(),
        tokenizer=None,
        overlay=_overlay(),
        hf_config=hf_config or _hf_config(),
        max_num_batched_tokens=max_num_batched_tokens,
        enable_position_interventions=enable_position_interventions,
    )


class TestVLLMDriverProtocolConformance:
    """VLLMDriver satisfies the Driver protocol and passes strict validation."""

    def test_passes_validate_driver(self):
        driver = _driver()
        assert isinstance(driver, Driver)
        validate_driver(driver)

    def test_no_torch_capability_flags(self):
        """vLLM owns the model in a worker — no torch-specific capability surface."""
        driver = _driver()
        for feature in ("parameters", "state_dict", "gradients", "weight_access"):
            assert driver.supports(feature) is False, f"vLLM shouldn't support {feature!r}"

    def test_non_fireable_expanded_per_layer(self):
        """``blocks.{i}.attn.hook_pattern`` template expands to one entry per layer."""
        assert _driver().non_fireable_hook_points == frozenset(
            {
                "blocks.0.attn.hook_pattern",
                "blocks.1.attn.hook_pattern",
                "blocks.0.attn.hook_attn_scores",
                "blocks.1.attn.hook_attn_scores",
            }
        )


class TestVLLMDriverConfig:
    """Boot-time config: logprobs request must match boot's max_logprobs."""

    def test_n_logprobs_uses_vocab_size(self):
        # hf_config.vocab_size (16) is the request count, matching boot's
        # max_logprobs — not bridge_config.d_vocab, which may be padded larger.
        assert _driver()._n_logprobs == 16


class TestVLLMDriverGetParam:
    """get_param fetches a named worker tensor via collective_rpc."""

    def test_returns_rpc_result(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[torch.ones(4)])
        out = driver.get_param("model.norm.weight")
        assert torch.equal(out, torch.ones(4))
        assert driver._llm.collective_rpc.call_args.args[0] == "tl_get_param"

    def test_returns_none_when_closed(self):
        driver = _driver(captures={})
        driver._llm = None
        assert driver.get_param("model.norm.weight") is None


class TestVLLMDriverForward:
    """forward dispatches via llm.generate and surfaces captures via ForwardResult."""

    def test_forward_logits_from_sampler_logprobs(self):
        """vLLM bypasses lm_head; driver synthesizes logits from sampler logprobs.
        Position -1 must carry the real next-token distribution."""
        pytest.importorskip("vllm")
        result = _driver(
            captures={"embed.hook_out": torch.randn(3, 4)},
            top_logprobs={7: 2.5, 3: 1.0, 11: -0.5},
        ).forward(torch.tensor([[1, 2, 3]]))
        assert isinstance(result, ForwardResult)
        assert tuple(result.captured["embed.hook_out"].shape) == (1, 3, 4)
        assert result.logits is not None and tuple(result.logits.shape) == (1, 3, 16)
        # Argmax = highest-logprob token. Values at non-listed positions are -inf.
        assert int(result.logits[0, -1].argmax().item()) == 7
        assert float(result.logits[0, -1, 7].item()) == 2.5

    def test_forward_logits_fallback_to_token_id(self):
        """If logprobs are absent (e.g. return_logits=False elsewhere upstream),
        _synthesize_logits falls back to the generated token id as a one-hot-ish."""
        pytest.importorskip("vllm")
        result = _driver(
            captures={"embed.hook_out": torch.randn(3, 4)},
            generated_token=9,
        ).forward(torch.tensor([[1, 2, 3]]))
        assert result.logits is not None
        assert int(result.logits[0, -1].argmax().item()) == 9

    def test_forward_rejects_batched_input(self):
        """batch_size=1 only. Raises in _normalize_input_ids before any vllm import."""
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            _driver(captures={}).forward(torch.tensor([[1, 2], [3, 4]]))

    def test_forward_rejects_callable_interventions(self):
        with pytest.raises(NotImplementedError, match="intervention specs"):
            _driver(captures={}).forward(
                torch.tensor([[1]]), intervene={"embed.hook_out": lambda a: a}
            )

    def test_forward_rejects_unsupported_intervention_op(self):
        with pytest.raises(ValueError, match="Unsupported intervention op"):
            _driver(captures={}).forward(
                torch.tensor([[1]]), intervene={"embed.hook_out": {"op": "clamp", "value": 1.0}}
            )

    def test_forward_rejects_unknown_intervention_hook(self):
        with pytest.raises(ValueError, match="not in supported_hook_points"):
            _driver(captures={}).forward(
                torch.tensor([[1]]), intervene={"blocks.99.hook_unknown": {"op": "suppress"}}
            )

    def test_forward_rejects_malformed_intervention_spec(self):
        with pytest.raises(ValueError, match="must be a dict with 'op' key"):
            _driver(captures={}).forward(
                torch.tensor([[1]]), intervene={"embed.hook_out": {"no_op_key": True}}
            )

    def test_forward_rejects_scale_missing_factor(self):
        with pytest.raises(ValueError, match="op='scale' requires 'factor'"):
            _driver(captures={}).forward(
                torch.tensor([[1]]), intervene={"embed.hook_out": {"op": "scale"}}
            )

    def test_forward_rejects_add_missing_value(self):
        with pytest.raises(ValueError, match="op='add' requires 'value'"):
            _driver(captures={}).forward(
                torch.tensor([[1]]), intervene={"embed.hook_out": {"op": "add"}}
            )

    def test_forward_rejects_set_missing_value(self):
        with pytest.raises(ValueError, match="op='set' requires 'value'"):
            _driver(captures={}).forward(
                torch.tensor([[1]]), intervene={"embed.hook_out": {"op": "set"}}
            )


class TestVLLMDriverPositionInterventions:
    """`pos`-scoped interventions require the opt-in (max_n, width) affine buffers."""

    def test_pos_rejected_without_flag(self):
        with pytest.raises(NotImplementedError, match="enable_position_interventions=True"):
            _driver()._validate_interventions({"embed.hook_out": {"op": "suppress", "pos": 0}})

    def test_pos_accepted_and_preserved_with_flag(self):
        driver = _driver(enable_position_interventions=True)
        out = driver._validate_interventions(
            {"embed.hook_out": {"op": "add", "value": 1.0, "pos": [0, 2]}}
        )
        assert out["embed.hook_out"]["pos"] == [0, 2]

    def test_pos_int_accepted_with_flag(self):
        driver = _driver(enable_position_interventions=True)
        out = driver._validate_interventions({"embed.hook_out": {"op": "suppress", "pos": 3}})
        assert out["embed.hook_out"]["pos"] == 3

    def test_pos_wrong_type_raises(self):
        driver = _driver(enable_position_interventions=True)
        with pytest.raises(ValueError, match="must be an int or list of ints"):
            driver._validate_interventions({"embed.hook_out": {"op": "suppress", "pos": "last"}})

    def test_pos_negative_raises(self):
        driver = _driver(enable_position_interventions=True)
        with pytest.raises(ValueError, match="must be non-negative"):
            driver._validate_interventions({"embed.hook_out": {"op": "suppress", "pos": [-1]}})

    def test_pos_rejected_on_batched_path(self):
        """The batched/eager path has no affine buffers, so 'pos' is unsupported there."""
        driver = _driver()
        driver._enable_batching = True
        with pytest.raises(NotImplementedError, match="batched/eager path"):
            driver._validate_interventions({"embed.hook_out": {"op": "suppress", "pos": 0}})

    def test_no_pos_still_works_with_flag(self):
        """A whole-sequence spec (no pos) is unaffected by the flag."""
        driver = _driver(enable_position_interventions=True)
        out = driver._validate_interventions({"embed.hook_out": {"op": "suppress"}})
        assert "pos" not in out["embed.hook_out"]

    def test_pos_beyond_prompt_length_fails_loud(self):
        """pos within buffer capacity but past the prompt length must raise, not silently no-op.

        The rejection fires before driver.forward reaches the vllm import, so no install needed.
        """
        driver = _driver(captures={}, enable_position_interventions=True)
        with pytest.raises(ValueError, match="beyond the prompt length"):
            driver.forward(  # 3-token prompt, pos=50 is unreadable by the hook
                torch.tensor([[1, 2, 3]]),
                intervene={"embed.hook_out": {"op": "suppress", "pos": 50}},
            )

    def test_reject_pos_beyond_seq_bounds_against_actual_length(self):
        driver = _driver(enable_position_interventions=True)
        driver._reject_pos_beyond_seq({"embed.hook_out": {"op": "suppress", "pos": [0, 2]}}, 3)
        with pytest.raises(ValueError, match="beyond the prompt length"):
            driver._reject_pos_beyond_seq({"embed.hook_out": {"op": "suppress", "pos": 3}}, 3)

    def test_forward_pushes_interventions_before_generate(self):
        """The driver pushes spec dicts via collective_rpc('tl_set_interventions', ...)."""
        pytest.importorskip("vllm")
        driver = _driver(captures={"embed.hook_out": torch.zeros(3, 4)})
        driver.forward(
            torch.tensor([[1, 2, 3]]),
            intervene={"embed.hook_out": {"op": "suppress"}},
        )
        rpc_calls = [c.args for c in driver._llm.collective_rpc.call_args_list]
        assert any(
            args[0] == "tl_set_interventions"
            and args[1] == ({"embed.hook_out": {"op": "suppress"}},)
            for args in rpc_calls
        )

    def test_forward_always_pushes_interventions_for_reset(self):
        """Even with intervene=None, push {} so stale state from prior forwards clears."""
        pytest.importorskip("vllm")
        driver = _driver(captures={"embed.hook_out": torch.zeros(3, 4)})
        driver.forward(torch.tensor([[1, 2, 3]]))
        rpc_calls = [c.args for c in driver._llm.collective_rpc.call_args_list]
        assert any(args[0] == "tl_set_interventions" and args[1] == ({},) for args in rpc_calls)

    def test_forward_rejects_max_new_tokens_gt_one(self):
        """Decode-step writes overwrite the prefill buffer — silent capture corruption."""
        with pytest.raises(NotImplementedError, match="max_new_tokens=1 only"):
            _driver(captures={}).forward(torch.tensor([[1, 2]]), max_new_tokens=2)

    def test_forward_rejects_prompt_exceeding_buffer(self):
        """Worker buffers silently clamp on overflow — driver must fail loud."""
        with pytest.raises(ValueError, match="exceeds max_num_batched_tokens"):
            _driver(captures={}, max_num_batched_tokens=4).forward(torch.tensor([[1, 2, 3, 4, 5]]))


def _batched_request_output(request_id, generated_token=None, top_logprobs=None):
    """RequestOutput-shaped mock carrying a request_id for the accumulator join."""
    completion = MagicMock()
    completion.token_ids = [generated_token] if generated_token is not None else []
    completion.logprobs = (
        [{tid: MagicMock(logprob=lp) for tid, lp in top_logprobs.items()}]
        if top_logprobs is not None
        else []
    )
    ro = MagicMock()
    ro.request_id = request_id
    ro.outputs = [completion]
    return ro


def _batched_driver(*, outputs, captures_by_req, hf_config=None) -> VLLMDriver:
    """Batched-mode VLLMDriver. ``outputs`` is the llm.generate return (in submission
    order, each carrying .request_id); ``captures_by_req`` is the
    tl_read_batched_captures payload keyed by req_id."""
    llm = MagicMock()
    llm.generate = MagicMock(return_value=outputs)
    # collective_rpc("tl_read_batched_captures")[0] is the only indexed call; the
    # reset/set calls ignore the return value.
    llm.collective_rpc = MagicMock(return_value=[captures_by_req])
    return VLLMDriver(
        llm=llm,
        adapter=_adapter(),
        tokenizer=None,
        overlay=_overlay(),
        hf_config=hf_config or _hf_config(),
        max_num_batched_tokens=2048,
        enable_batching=True,
    )


class TestNormalizeInputIdsBatched:
    """_normalize_input_ids_batched accepts tensors, flat lists, and ragged lists."""

    def test_1d_tensor_is_single_prompt(self):
        assert VLLMDriver._normalize_input_ids_batched(torch.tensor([1, 2, 3])) == [[1, 2, 3]]

    def test_2d_tensor_is_per_row(self):
        assert VLLMDriver._normalize_input_ids_batched(torch.tensor([[1, 2], [3, 4]])) == [
            [1, 2],
            [3, 4],
        ]

    def test_flat_list_is_single_prompt(self):
        assert VLLMDriver._normalize_input_ids_batched([1, 2, 3]) == [[1, 2, 3]]

    def test_ragged_list_of_lists_preserved(self):
        assert VLLMDriver._normalize_input_ids_batched([[1, 2, 3], [4, 5]]) == [[1, 2, 3], [4, 5]]


class TestBatchedForward:
    """Batched path: req_id join, right-padded assembly, per-row logit synthesis."""

    def test_req_id_join_not_positional(self):
        """Accumulator keys are out-of-order req_ids; join via outputs[k].request_id."""
        pytest.importorskip("vllm")
        outputs = [
            _batched_request_output("req-A", top_logprobs={7: 2.0}),
            _batched_request_output("req-B", top_logprobs={3: 2.0}),
        ]
        # Captures keyed by req_id, deliberately reverse insertion order.
        captures_by_req = {
            "req-B": {"embed.hook_out": torch.full((2, 4), 2.0)},
            "req-A": {"embed.hook_out": torch.full((3, 4), 1.0)},
        }
        result = _batched_driver(outputs=outputs, captures_by_req=captures_by_req).forward(
            [[1, 2, 3], [4, 5]]
        )
        emb = result.captured["embed.hook_out"]
        assert tuple(emb.shape) == (2, 3, 4)  # (batch, max_seq, width)
        # Row 0 = req-A (3 real tokens, value 1.0); row 1 = req-B (2 real, value 2.0).
        assert torch.equal(emb[0, :3], torch.full((3, 4), 1.0))
        assert torch.equal(emb[1, :2], torch.full((2, 4), 2.0))

    def test_join_strips_internal_req_id_suffix(self):
        """vLLM keys the worker accumulator by f'{public}-{hash}'; RequestOutput
        carries only the public id. Join must match exact-or-prefix."""
        pytest.importorskip("vllm")
        outputs = [
            _batched_request_output("10", top_logprobs={7: 2.0}),
            _batched_request_output("11", top_logprobs={3: 2.0}),
        ]
        # Worker keys carry the engine-internal "-<hash>" suffix.
        captures_by_req = {
            "10-83c3532c": {"embed.hook_out": torch.full((3, 4), 1.0)},
            "11-a1b2c3d4": {"embed.hook_out": torch.full((2, 4), 2.0)},
        }
        result = _batched_driver(outputs=outputs, captures_by_req=captures_by_req).forward(
            [[1, 2, 3], [4, 5]]
        )
        emb = result.captured["embed.hook_out"]
        assert tuple(emb.shape) == (2, 3, 4)
        assert torch.equal(emb[0, :3], torch.full((3, 4), 1.0))
        assert torch.equal(emb[1, :2], torch.full((2, 4), 2.0))

    def test_join_prefix_does_not_collide_on_numeric_ids(self):
        """Public '1' must not match worker key '10-...'; the '-' delimiter guards it."""
        pytest.importorskip("vllm")
        outputs = [
            _batched_request_output("1", top_logprobs={7: 2.0}),
            _batched_request_output("10", top_logprobs={3: 2.0}),
        ]
        captures_by_req = {
            "1-aaaaaaaa": {"embed.hook_out": torch.full((2, 4), 1.0)},
            "10-bbbbbbbb": {"embed.hook_out": torch.full((3, 4), 9.0)},
        }
        result = _batched_driver(outputs=outputs, captures_by_req=captures_by_req).forward(
            [[1, 2], [1, 2, 3]]
        )
        emb = result.captured["embed.hook_out"]
        assert torch.equal(emb[0, :2], torch.full((2, 4), 1.0))  # req "1"
        assert torch.equal(emb[1, :3], torch.full((3, 4), 9.0))  # req "10"

    def test_shorter_rows_right_padded_with_zeros(self):
        pytest.importorskip("vllm")
        outputs = [
            _batched_request_output("r0", top_logprobs={1: 1.0}),
            _batched_request_output("r1", top_logprobs={1: 1.0}),
        ]
        captures_by_req = {
            "r0": {"embed.hook_out": torch.ones(3, 4)},
            "r1": {"embed.hook_out": torch.ones(1, 4)},
        }
        result = _batched_driver(outputs=outputs, captures_by_req=captures_by_req).forward(
            [[1, 2, 3], [9]]
        )
        emb = result.captured["embed.hook_out"]
        # Row 1 has 1 real token; positions 1,2 are zero pad.
        assert torch.equal(emb[1, 1:], torch.zeros(2, 4))

    def test_logits_at_per_row_last_token(self):
        """Next-token logits land at prompt_len-1 per row, never -1 (a pad row)."""
        pytest.importorskip("vllm")
        outputs = [
            _batched_request_output("r0", top_logprobs={7: 5.0}),
            _batched_request_output("r1", top_logprobs={3: 5.0}),
        ]
        captures_by_req = {
            "r0": {"embed.hook_out": torch.ones(3, 4)},
            "r1": {"embed.hook_out": torch.ones(1, 4)},
        }
        result = _batched_driver(outputs=outputs, captures_by_req=captures_by_req).forward(
            [[1, 2, 3], [9]]
        )
        logits = result.logits
        assert logits is not None and tuple(logits.shape) == (2, 3, 16)
        # Row 0 last token at pos 2; row 1 last token at pos 0 (not 2, which is pad).
        assert int(logits[0, 2].argmax().item()) == 7
        assert int(logits[1, 0].argmax().item()) == 3
        assert torch.isinf(logits[1, 2]).all()  # pad position stays -inf

    def test_assemble_padded_raises_on_missing_join(self):
        """A request with no matching worker key must raise, not zero-fill silently."""
        outputs = [SimpleNamespace(request_id="99")]
        worker_captures = {"10-83c3532c": {"embed.hook_out": torch.ones(2, 4)}}
        with pytest.raises(RuntimeError, match="Cannot join request '99'"):
            VLLMDriver._assemble_padded(outputs, worker_captures, [2])

    def test_assemble_padded_raises_on_ambiguous_join(self):
        """Two worker keys sharing the public-id prefix is ambiguous — raise."""
        outputs = [SimpleNamespace(request_id="1")]
        worker_captures = {
            "1-aaaaaaaa": {"embed.hook_out": torch.ones(2, 4)},
            "1-bbbbbbbb": {"embed.hook_out": torch.ones(2, 4)},
        }
        with pytest.raises(RuntimeError, match="found 2 key"):
            VLLMDriver._assemble_padded(outputs, worker_captures, [2])

    def test_resets_accumulator_before_generate(self):
        """tl_reset_accumulators must fire so prior-forward chunks don't leak."""
        pytest.importorskip("vllm")
        driver = _batched_driver(
            outputs=[_batched_request_output("r0", top_logprobs={1: 1.0})],
            captures_by_req={"r0": {"embed.hook_out": torch.ones(2, 4)}},
        )
        driver.forward([[1, 2]])
        methods = [c.args[0] for c in driver._llm.collective_rpc.call_args_list]
        assert "tl_reset_accumulators" in methods
        assert "tl_set_batched_interventions" in methods

    def test_batched_intervention_spec_pushed(self):
        pytest.importorskip("vllm")
        driver = _batched_driver(
            outputs=[_batched_request_output("r0", top_logprobs={1: 1.0})],
            captures_by_req={"r0": {"embed.hook_out": torch.ones(2, 4)}},
        )
        driver.forward([[1, 2]], intervene={"embed.hook_out": {"op": "suppress"}})
        calls = [c.args for c in driver._llm.collective_rpc.call_args_list]
        assert any(
            a[0] == "tl_set_batched_interventions"
            and a[1] == ({"embed.hook_out": {"op": "suppress"}},)
            for a in calls
        )


class TestVLLMDriverThroughBridge:
    """End-to-end via RemoteBridge — drivers' captures flow into the HookPoint tree."""

    def test_bridge_replays_vllm_captures(self):
        pytest.importorskip("vllm")
        driver = _driver(captures={"embed.hook_out": torch.randn(3, 4)})
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)

        fired: list = []
        bridge.add_hook("embed.hook_out", lambda act, hook: fired.append(act))
        bridge.forward(torch.tensor([[1, 2, 3]]))
        assert len(fired) == 1
        assert tuple(fired[0].shape) == (1, 3, 4)

    def test_loss_return_type_gated_on_sequence_logits(self):
        """The bridge's loss/both guard fires only when the driver lacks full-sequence
        logits. The vLLM driver now reconstructs them (provides_sequence_logits=True),
        so loss/both are permitted; a driver without that capability is still refused.

        The rejection path fires before driver.forward, so it needs no vllm install.
        """
        driver = _driver(captures={})
        assert driver.provides_sequence_logits is True  # reconstruction path is live
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)
        # Drop the capability → the guard must refuse loss/both (nan over -inf positions).
        driver.provides_sequence_logits = False
        for rt in ("loss", "both"):
            with pytest.raises(NotImplementedError, match="return_type"):
                bridge.forward(torch.tensor([[1, 2, 3]]), return_type=rt)

    def test_run_with_hooks_rejects_bwd_hooks(self):
        """No backward pass on a remote driver — bwd_hooks raise before any forward."""
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=_driver(captures={}))
        with pytest.raises(NotImplementedError, match="backward"):
            bridge.run_with_hooks(
                torch.tensor([[1, 2, 3]]),
                bwd_hooks=[("embed.hook_out", lambda a, hook: a)],
            )

    def test_context_manager_closes_engine(self):
        """`with bridge:` releases the engine on exit (close() nulls the LLM)."""
        driver = _driver(captures={})
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)
        with bridge as entered:
            assert entered is bridge
        assert driver._llm is None  # close() ran

    def test_run_with_hooks_warns_fwd_hooks_are_read_only(self):
        """A mutating fwd_hook is a no-op on a remote driver — warn loudly."""
        pytest.importorskip("vllm")
        driver = _driver(captures={"embed.hook_out": torch.randn(3, 4)})
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)
        with pytest.warns(UserWarning, match="read-only"):
            bridge.run_with_hooks(
                torch.tensor([[1, 2, 3]]),
                fwd_hooks=[("embed.hook_out", lambda a, hook: a)],
            )
