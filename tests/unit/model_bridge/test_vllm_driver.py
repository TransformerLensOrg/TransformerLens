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
