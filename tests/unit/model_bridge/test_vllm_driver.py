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

    def test_capability_flags(self):
        """vLLM owns the model in a worker — no torch module surface (parameters/
        state_dict/grads), but named-weight reads ARE served via the get_param RPC."""
        driver = _driver()
        for feature in ("parameters", "state_dict", "gradients"):
            assert driver.supports(feature) is False, f"vLLM shouldn't support {feature!r}"
        assert driver.supports("weight_access") is True

    def test_zero_layers_config_fails_loud(self):
        """A missing/zero num_hidden_layers must raise at boot, not leave a raw '{i}'
        template in non_fireable_hook_points."""
        with pytest.raises(ValueError, match="num_hidden_layers"):
            _driver(hf_config=_hf_config(num_hidden_layers=0))

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
        ).forward(torch.tensor([[1, 2, 3]]), capture=("embed.hook_out",))
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
        assert any(
            c.args[0] == "tl_set_interventions"
            and c.kwargs.get("args") == ({"embed.hook_out": {"op": "suppress"}},)
            for c in driver._llm.collective_rpc.call_args_list
        )

    def test_forward_always_pushes_interventions_for_reset(self):
        """Even with intervene=None, push {} so stale state from prior forwards clears."""
        pytest.importorskip("vllm")
        driver = _driver(captures={"embed.hook_out": torch.zeros(3, 4)})
        driver.forward(torch.tensor([[1, 2, 3]]))
        assert any(
            c.args[0] == "tl_set_interventions" and c.kwargs.get("args") == ({},)
            for c in driver._llm.collective_rpc.call_args_list
        )

    def test_forward_rejects_max_new_tokens_gt_one(self):
        """Decode-step writes overwrite the prefill buffer — silent capture corruption."""
        with pytest.raises(NotImplementedError, match="max_new_tokens=1 only"):
            _driver(captures={}).forward(torch.tensor([[1, 2]]), max_new_tokens=2)

    def test_forward_rejects_prompt_exceeding_buffer(self):
        """Worker buffers silently clamp on overflow — driver must fail loud."""
        with pytest.raises(ValueError, match="exceeds max_num_batched_tokens"):
            _driver(captures={}, max_num_batched_tokens=4).forward(torch.tensor([[1, 2, 3, 4, 5]]))

    def test_zero_capture_skips_worker_read(self):
        """capture=() must not trigger a GPU→CPU copy: with logits off there is no
        tl_read_captures RPC at all (an empty tuple used to collapse to None = read
        every buffer); with logits on, only the forced ln_final is read."""
        pytest.importorskip("vllm")
        driver = _driver(captures={"embed.hook_out": torch.zeros(3, 4)})
        driver.forward(torch.tensor([[1, 2, 3]]), return_logits=False)
        reads = [
            c.args
            for c in driver._llm.collective_rpc.call_args_list
            if c.args[0] == "tl_read_captures"
        ]
        assert reads == [], f"hookless forward still read the buffers: {reads}"

        # Method-aware mock: a blanket return value would hand the reconstruction
        # probe a captures dict as a "weight".
        driver2 = _driver(captures={})
        lnf = {"ln_final.hook_normalized": torch.zeros(3, 4)}

        def rpc2(method, args=(), **kwargs):
            if method == "tl_read_captures":
                return [lnf]
            if method == "tl_get_param":
                return [None]  # no unembedding -> sampler-logprob fallback
            return [[]]

        driver2._llm.collective_rpc = MagicMock(side_effect=rpc2)
        driver2.forward(torch.tensor([[1, 2, 3]]))  # return_logits=True default
        reads2 = [
            c for c in driver2._llm.collective_rpc.call_args_list if c.args[0] == "tl_read_captures"
        ]
        assert len(reads2) == 1
        assert reads2[0].kwargs.get("args")[1] == ["ln_final.hook_normalized"]


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
            [[1, 2, 3], [4, 5]], capture=("embed.hook_out",)
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
            [[1, 2, 3], [4, 5]], capture=("embed.hook_out",)
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
            [[1, 2], [1, 2, 3]], capture=("embed.hook_out",)
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
            [[1, 2, 3], [9]], capture=("embed.hook_out",)
        )
        emb = result.captured["embed.hook_out"]
        # Row 1 has 1 real token; positions 1,2 are zero pad.
        assert torch.equal(emb[1, 1:], torch.zeros(2, 4))

    def test_zero_capture_skips_batched_join(self):
        """capture=() with logits off must return empty captures, not crash: _assemble_padded
        needs one worker key per request, so it can't run on the empty read — mirror the
        single path. (Regression: the batched join was called unconditionally.)"""
        pytest.importorskip("vllm")
        outputs = [
            _batched_request_output("r0", top_logprobs={1: 1.0}),
            _batched_request_output("r1", top_logprobs={1: 1.0}),
        ]
        driver = _batched_driver(outputs=outputs, captures_by_req={})
        result = driver.forward([[1, 2, 3], [9]], return_logits=False)  # capture=() default
        assert result.captured == {}
        reads = [
            c.args
            for c in driver._llm.collective_rpc.call_args_list
            if c.args[0] == "tl_read_batched_captures"
        ]
        assert reads == [], f"batched hookless forward still read captures: {reads}"

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
            [[1, 2, 3], [9]], capture=("embed.hook_out",)
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
        assert any(
            c.args[0] == "tl_set_batched_interventions"
            and c.kwargs.get("args") == ({"embed.hook_out": {"op": "suppress"}},)
            for c in driver._llm.collective_rpc.call_args_list
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


class TestVLLMDriverSpecKeyValidation:
    """Unknown spec keys and mis-shaped values must fail loud, not silently degrade."""

    def test_unknown_spec_key_rejected(self):
        """A typo'd 'position' would otherwise become a whole-sequence edit."""
        with pytest.raises(ValueError, match="unknown spec key"):
            _driver(enable_position_interventions=True)._validate_interventions(
                {"embed.hook_out": {"op": "add", "value": 1.0, "position": 3}}
            )

    def test_known_keys_accepted(self):
        out = _driver(enable_position_interventions=True)._validate_interventions(
            {"embed.hook_out": {"op": "add", "value": 1.0, "pos": 2}}
        )
        assert out["embed.hook_out"]["pos"] == 2

    def test_value_width_mismatch_rejected(self):
        """embed.hook_out width is 4 (overlay spec); a 2-element value would broadcast
        wrong or crash mid-forward."""
        with pytest.raises(ValueError, match="2 elements"):
            _driver()._validate_interventions(
                {"embed.hook_out": {"op": "add", "value": [1.0, 2.0]}}
            )

    def test_value_width_match_accepted(self):
        out = _driver()._validate_interventions(
            {"embed.hook_out": {"op": "set", "value": [1.0, 2.0, 3.0, 4.0]}}
        )
        assert out["embed.hook_out"]["value"] == [1.0, 2.0, 3.0, 4.0]

    def test_scalar_value_accepted(self):
        out = _driver()._validate_interventions({"embed.hook_out": {"op": "add", "value": 2.5}})
        assert out["embed.hook_out"]["value"] == 2.5

    def test_pos_bool_rejected(self):
        """bool is an int subclass — pos=True must not pass as position 1."""
        driver = _driver(enable_position_interventions=True)
        with pytest.raises(ValueError, match="int or list of ints"):
            driver._validate_interventions({"embed.hook_out": {"op": "suppress", "pos": True}})
        with pytest.raises(ValueError, match="int or list of ints"):
            driver._validate_interventions({"embed.hook_out": {"op": "suppress", "pos": [True]}})


class TestVLLMDriverLogitReconstruction:
    """Boot-time probe caches the unembedding; reconstruction slices padded vocab."""

    def test_probe_unavailable_downgrades_sequence_logits(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[None])
        assert driver.probe_logit_reconstruction() is False
        assert driver._unembed is None and driver._unembed_probed
        assert driver.provides_sequence_logits is False

    def test_probe_available_caches_weight(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[torch.ones(16, 4)])
        assert driver.probe_logit_reconstruction() is True
        assert driver.provides_sequence_logits is True
        weight32, _bias = driver._unembed
        assert weight32.dtype == torch.float32
        # Idempotent: a second probe answers from the cache, no extra RPC.
        calls_after_first = driver._llm.collective_rpc.call_count
        assert driver.probe_logit_reconstruction() is True
        assert driver._llm.collective_rpc.call_count == calls_after_first

    def test_reconstruct_uses_cached_weight_without_rpc(self):
        """After the probe, per-forward reconstruction must not re-fetch the weight."""
        driver = _driver(captures={})
        driver._unembed = (torch.eye(4, dtype=torch.float32).repeat(4, 1), None)
        driver._unembed_probed = True
        driver._llm.collective_rpc = MagicMock(
            side_effect=AssertionError("reconstruction must not RPC")
        )
        out = driver._reconstruct_logits(torch.ones(3, 4))
        assert out is not None and out.shape == (3, 16)

    def test_reconstruct_slices_padded_vocab(self):
        """vLLM pads vocab to a multiple of 64; d_vocab=16 here, weight padded to 24.
        Un-sliced, the zero-filled pad columns become phantom argmax candidates when
        every real logit is negative."""
        driver = _driver(captures={})
        weight = torch.zeros(24, 4, dtype=torch.float32)
        weight[:16] = -1.0  # real vocab rows: all-negative logits
        driver._unembed = (weight, None)
        driver._unembed_probed = True
        out = driver._reconstruct_logits(torch.ones(3, 4))
        assert out.shape == (3, 16)
        assert int(out[0].argmax()) < 16

    def test_reconstruct_returns_none_when_probed_unavailable(self):
        driver = _driver(captures={})
        driver._unembed_probed = True  # probed, nothing found
        assert driver._reconstruct_logits(torch.ones(3, 4)) is None


class TestVLLMDriverLnFinalUnfold:
    """The exposed ln_final capture honors the hook name's pre-weight convention."""

    def test_unfold_divides_by_weight(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[torch.full((4,), 2.0)])
        out = driver._unfold_ln_final(torch.ones(1, 3, 4))
        assert torch.allclose(out, torch.full((1, 3, 4), 0.5))

    def test_unfold_gemma_uses_one_plus_weight(self):
        driver = _driver(captures={})
        driver.architecture = "Gemma2ForCausalLM"
        driver._llm.collective_rpc = MagicMock(return_value=[torch.full((4,), 1.0)])
        out = driver._unfold_ln_final(torch.ones(1, 3, 4))
        assert torch.allclose(out, torch.full((1, 3, 4), 0.5))

    def test_unfold_near_zero_weight_guarded(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[torch.zeros(4)])
        out = driver._unfold_ln_final(torch.ones(1, 3, 4))
        assert torch.isfinite(out).all()

    def test_unfold_warns_and_returns_raw_when_weight_unreachable(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[None])
        t = torch.ones(1, 3, 4)
        with pytest.warns(UserWarning, match="POST-weight"):
            out = driver._unfold_ln_final(t)
        assert torch.equal(out, t)
        # Negative outcome cached: no per-forward RPC retry, no repeat warning.
        out2 = driver._unfold_ln_final(t)
        assert torch.equal(out2, t)
        assert driver._llm.collective_rpc.call_count == 1

    def test_unfold_caches_fetched_weight(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[torch.full((4,), 2.0)])
        driver._unfold_ln_final(torch.ones(1, 3, 4))
        assert driver._llm.collective_rpc.call_count == 1
        driver._unfold_ln_final(torch.ones(1, 3, 4))
        assert driver._llm.collective_rpc.call_count == 1  # cached, no second RPC


class TestVLLMDriverCloseRefcount:
    """close() only tears down vLLM's process-global distributed state when this is
    the last live driver — a re-bound notebook boot must not break the new engine."""

    def _patch_distributed(self, monkeypatch):
        import sys as _sys

        dist = MagicMock()
        monkeypatch.setitem(_sys.modules, "vllm", MagicMock())
        monkeypatch.setitem(_sys.modules, "vllm.distributed", MagicMock())
        monkeypatch.setitem(_sys.modules, "vllm.distributed.parallel_state", dist)
        return dist

    def test_teardown_deferred_until_last_driver(self, monkeypatch):
        from transformer_lens.model_bridge.sources.vllm import driver as driver_module

        dist = self._patch_distributed(monkeypatch)
        monkeypatch.setattr(driver_module, "_LIVE_DRIVERS", 0)
        a, b = _driver(captures={}), _driver(captures={})
        a.close()
        dist.destroy_model_parallel.assert_not_called()
        b.close()
        dist.destroy_model_parallel.assert_called_once()
        dist.destroy_distributed_environment.assert_called_once()

    def test_double_close_decrements_once(self, monkeypatch):
        from transformer_lens.model_bridge.sources.vllm import driver as driver_module

        dist = self._patch_distributed(monkeypatch)
        monkeypatch.setattr(driver_module, "_LIVE_DRIVERS", 0)
        a, b = _driver(captures={}), _driver(captures={})
        a.close()
        a.close()  # idempotent: must not reach zero while b is live
        dist.destroy_model_parallel.assert_not_called()
        b.close()
        dist.destroy_model_parallel.assert_called_once()

    def test_close_drops_weight_caches(self, monkeypatch):
        self._patch_distributed(monkeypatch)
        driver = _driver(captures={})
        driver._unembed = (torch.ones(16, 4), None)
        driver._lnf_inv_denom = torch.ones(4)
        driver.close()
        assert driver._unembed is None and driver._lnf_inv_denom is None


class TestBatchedForwardPaddingSemantics:
    """Reconstruction pads and attention masks on the batched path."""

    def test_reconstructed_pad_positions_are_neg_inf(self):
        """Zero-filled ln_final pad rows reconstruct into finite garbage (0 @ W);
        they must be masked to the -inf convention the fallback path uses."""
        pytest.importorskip("vllm")
        outputs = [
            _batched_request_output("req-A", top_logprobs={7: 2.0}),
            _batched_request_output("req-B", top_logprobs={3: 2.0}),
        ]
        captures_by_req = {
            "req-A": {"ln_final.hook_normalized": torch.ones(3, 4)},
            "req-B": {"ln_final.hook_normalized": torch.ones(2, 4)},
        }
        driver = _batched_driver(outputs=outputs, captures_by_req=captures_by_req)
        driver._unembed = (torch.ones(16, 4, dtype=torch.float32), None)
        driver._unembed_probed = True
        result = driver.forward([[1, 2, 3], [4, 5]])
        assert result.logits.shape == (2, 3, 16)
        assert torch.isfinite(result.logits[0]).all()  # 3 real rows
        assert torch.isfinite(result.logits[1, :2]).all()
        assert torch.isinf(result.logits[1, 2]).all()  # pad row for the 2-token prompt

    def test_attention_mask_trims_padded_rows(self):
        """A right-padded (B, S) batch + mask must send only real tokens to vLLM."""
        pytest.importorskip("vllm")
        outputs = [
            _batched_request_output("req-A", top_logprobs={7: 2.0}),
            _batched_request_output("req-B", top_logprobs={3: 2.0}),
        ]
        driver = _batched_driver(outputs=outputs, captures_by_req={})
        driver.forward(
            torch.tensor([[1, 2, 3], [4, 5, 0]]),
            attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
            return_logits=False,
        )
        prompts = driver._llm.generate.call_args.kwargs["prompts"]
        assert prompts[0]["prompt_token_ids"] == [1, 2, 3]
        assert prompts[1]["prompt_token_ids"] == [4, 5]  # pad token trimmed

    def test_attention_mask_batch_mismatch_raises(self):
        pytest.importorskip("vllm")
        driver = _batched_driver(outputs=[], captures_by_req={})
        with pytest.raises(ValueError, match="attention_mask batch dim"):
            driver.forward(
                torch.tensor([[1, 2], [3, 4]]),
                attention_mask=torch.tensor([[1, 1]]),
                return_logits=False,
            )


class TestGatherParam:
    """Cross-rank param reads: replicated → rank 0; vocab-sharded → rank-order concat."""

    def test_single_rank_returns_shard(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[torch.ones(16, 4)])
        assert torch.equal(driver._gather_param("lm_head.weight"), torch.ones(16, 4))

    def test_replicated_shards_return_rank_zero(self):
        """Norm weights are identical on every rank — must NOT concatenate."""
        driver = _driver(captures={})
        w = torch.full((4,), 2.0)
        driver._llm.collective_rpc = MagicMock(return_value=[w, w.clone()])
        assert driver._gather_param("model.norm.weight").shape == (4,)

    def test_sharded_shards_concat_in_rank_order(self):
        driver = _driver(captures={})
        shard0, shard1 = torch.zeros(8, 4), torch.ones(8, 4)
        driver._llm.collective_rpc = MagicMock(return_value=[shard0, shard1])
        full = driver._gather_param("lm_head.weight")
        assert full.shape == (16, 4)
        assert torch.equal(full[:8], shard0) and torch.equal(full[8:], shard1)

    def test_missing_on_all_ranks_returns_none(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[None, None])
        assert driver._gather_param("lm_head.bias") is None

    def test_probe_reconstructs_across_sharded_vocab(self):
        """End-to-end: a 2-rank sharded unembedding must reassemble so tokens in the
        UPPER half of the vocab can win argmax (concat-order regression)."""
        driver = _driver(captures={})
        # Rank 0 rows favor token 0; rank 1 rows favor the ln_final direction strongly.
        shard0 = torch.zeros(8, 4)
        shard1 = torch.zeros(8, 4)
        shard1[3] = 10.0  # global token id 8 + 3 = 11

        def rpc(method, args=(), **kwargs):
            if method == "tl_get_param" and args[0] == "lm_head.weight":
                return [shard0, shard1]
            if method == "tl_get_param":
                return [None, None]  # no bias; no tied-embedding fallback needed
            return [[]]

        driver._llm.collective_rpc = MagicMock(side_effect=rpc)
        assert driver.probe_logit_reconstruction() is True
        logits = driver._reconstruct_logits(torch.ones(3, 4))
        assert logits.shape == (3, 16)
        assert int(logits[0].argmax()) == 11


class TestTPReplicationTripwire:
    """First capture-bearing forward under TP cross-checks all ranks."""

    def _tp_driver(self):
        driver = _driver(captures={})
        driver._tp_size = 2
        driver._tp_verified = False
        return driver

    def test_replicated_captures_pass(self):
        driver = self._tp_driver()
        t = torch.randn(3, 4)
        driver._llm.collective_rpc = MagicMock(return_value=[torch.zeros(1), torch.zeros(1)])
        driver._verify_tp_replication([{"embed.hook_out": t}, {"embed.hook_out": t.clone()}])

    def test_divergent_captures_fail_loud(self):
        driver = self._tp_driver()
        with pytest.raises(RuntimeError, match="no longer replicated"):
            driver._verify_tp_replication(
                [
                    {"embed.hook_out": torch.zeros(3, 4)},
                    {"embed.hook_out": torch.ones(3, 4)},
                ]
            )

    def test_missing_hook_on_other_rank_fails_loud(self):
        driver = self._tp_driver()
        with pytest.raises(RuntimeError, match="no longer replicated"):
            driver._verify_tp_replication([{"embed.hook_out": torch.zeros(3, 4)}, {}])

    def test_fire_counter_mismatch_fails_loud(self):
        driver = self._tp_driver()
        t = torch.randn(3, 4)
        driver._llm.collective_rpc = MagicMock(return_value=[torch.tensor([4]), torch.tensor([2])])
        with pytest.raises(RuntimeError, match="fire-counter mismatch"):
            driver._verify_tp_replication([{"embed.hook_out": t}, {"embed.hook_out": t.clone()}])

    def test_single_rank_driver_skips_verification(self):
        driver = _driver(captures={})
        assert driver._tp_size == 1 and driver._tp_verified is True


class TestRpcTensorCoercion:
    """Multiproc collective_rpc (TP>1) serializes tensors to nested lists —
    every RPC read must coerce back. GPU-verified failure mode on vllm 0.20.2."""

    def test_get_param_coerces_list_payload(self):
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[[[1.0, 2.0], [3.0, 4.0]]])
        out = driver.get_param("model.norm.weight")
        assert isinstance(out, torch.Tensor) and out.shape == (2, 2)

    def test_gather_param_concats_list_shards(self):
        """The exact crash from the box: list shards have no .shape."""
        driver = _driver(captures={})
        shard0 = [[0.0] * 4] * 8
        shard1 = [[1.0] * 4] * 8
        driver._llm.collective_rpc = MagicMock(return_value=[shard0, shard1])
        full = driver._gather_param("lm_head.weight")
        assert isinstance(full, torch.Tensor) and full.shape == (16, 4)
        assert torch.equal(full[8:], torch.ones(8, 4))

    def test_gather_param_detects_replicated_list_shards(self):
        driver = _driver(captures={})
        weight = [2.0, 2.0, 2.0, 2.0]
        driver._llm.collective_rpc = MagicMock(return_value=[weight, list(weight)])
        out = driver._gather_param("model.norm.weight")
        assert out.shape == (4,)

    def test_rpc_captures_coerces_dict_values(self):
        coerced = VLLMDriver._rpc_captures({"embed.hook_out": [[1.0, 2.0]]})
        assert isinstance(coerced["embed.hook_out"], torch.Tensor)

    def test_tensor_payloads_pass_through_unchanged(self):
        t = torch.randn(3, 4)
        assert VLLMDriver._rpc_tensor(t) is t

    def test_get_param_none_payload_stays_none(self):
        """None (missing path) is filtered before coercion at the call sites."""
        driver = _driver(captures={})
        driver._llm.collective_rpc = MagicMock(return_value=[None])
        assert driver.get_param("no.such.param") is None
