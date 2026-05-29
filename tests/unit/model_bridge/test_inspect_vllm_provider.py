"""Unit tests for the vLLM-backed Inspect provider (``tl_bridge_vllm``) — mocks vLLM so
no vllm install is needed. Live end-to-end verification (real vLLM, real GPU) is in the
Colab walkthrough; the mocked tests pin the construction, generate→ModelOutput conversion,
and the validation surface (locked kwargs, capture path, model_args["capture"])."""
from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


def _install_vllm_mocks(monkeypatch, llm_class: Any) -> None:
    """Install fake ``vllm``/``vllm.inputs``/``vllm.distributed.parallel_state`` modules
    in ``sys.modules`` so the provider's lazy ``from vllm import …`` resolves to mocks."""
    vllm = types.ModuleType("vllm")
    vllm.LLM = llm_class
    vllm.SamplingParams = MagicMock(name="SamplingParams")
    vllm_inputs = types.ModuleType("vllm.inputs")
    vllm_inputs.TokensPrompt = MagicMock(name="TokensPrompt")
    vllm.inputs = vllm_inputs
    vllm_dist = types.ModuleType("vllm.distributed")
    vllm_ps = types.ModuleType("vllm.distributed.parallel_state")
    vllm_ps.destroy_model_parallel = MagicMock(name="destroy_model_parallel")
    vllm_ps.destroy_distributed_environment = MagicMock(name="destroy_distributed_environment")
    vllm_dist.parallel_state = vllm_ps
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", vllm_inputs)
    monkeypatch.setitem(sys.modules, "vllm.distributed", vllm_dist)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", vllm_ps)


def _fake_tokenizer(monkeypatch) -> Any:
    """A fake ``AutoTokenizer.from_pretrained`` returning a tokenizer with a trivial
    chat-template-free tokenize/decode (just maps ints↔single-char strings)."""
    tok = SimpleNamespace(chat_template=None, pad_token_id=0, eos_token_id=1)

    def encode(text: str) -> list[int]:
        return [ord(c) % 50 for c in str(text)[:32]]

    tok.__call__ = lambda text, **kw: SimpleNamespace(input_ids=encode(text))  # type: ignore[attr-defined]
    callable_tok = MagicMock(wraps=tok)
    callable_tok.side_effect = lambda text, **kw: SimpleNamespace(input_ids=encode(text))
    callable_tok.decode = lambda ids, **kw: "".join(chr(int(i)) for i in ids)
    callable_tok.chat_template = None
    callable_tok.pad_token_id = 0
    callable_tok.eos_token_id = 1

    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: callable_tok)
    return callable_tok


def _fake_hf_config(n_layers: int = 2, d_model: int = 4, vocab_size: int = 128) -> Any:
    """Minimal HF config the overlay + provider reach into (vocab_size,
    num_hidden_layers, hidden_size, architectures, torch_dtype)."""
    return SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        hidden_size=d_model,
        num_hidden_layers=n_layers,
        vocab_size=vocab_size,
        torch_dtype="float16",
    )


def _patch_construction(monkeypatch, hf_config: Any) -> None:
    """Patch AutoConfig + plugin.configure/register + extract_hf_config so __init__ runs
    without vllm / real worker processes."""
    from transformers import AutoConfig

    from transformer_lens.model_bridge.sources.vllm import internals, plugin

    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: hf_config)
    monkeypatch.setattr(plugin, "configure", lambda **kw: None)
    monkeypatch.setattr(plugin, "register", lambda: None)
    monkeypatch.setattr(internals, "extract_hf_config", lambda llm: hf_config)


def _make_request_output(
    new_token_ids: list[int], finish_reason: str = "length", logprobs: Any = None
) -> Any:
    """Fake vLLM ``RequestOutput`` exposing ``.outputs[0].{token_ids, finish_reason, logprobs}``."""
    inner = SimpleNamespace(
        token_ids=list(new_token_ids), finish_reason=finish_reason, logprobs=logprobs
    )
    return SimpleNamespace(outputs=[inner], prompt_token_ids=[])


def _make_provider(monkeypatch, llm_instance: Any, hf_config: Any | None = None):
    """Construct the provider with mocked vllm.LLM + fake tokenizer + patched AutoConfig /
    plugin / extract_hf_config; device='cpu' so intermediate tensors don't require CUDA."""
    _fake_tokenizer(monkeypatch)
    hf_config = hf_config or _fake_hf_config()
    _patch_construction(monkeypatch, hf_config)
    llm_class = MagicMock(return_value=llm_instance, name="LLM")
    _install_vllm_mocks(monkeypatch, llm_class)
    from transformer_lens.model_bridge.sources.inspect.vllm_provider import (
        TransformerLensVLLMModelAPI,
    )

    return TransformerLensVLLMModelAPI("any/model", device="cpu"), llm_class


class TestModuleSafety:
    def test_module_imports_without_vllm(self):
        # Confirm the lazy-vllm pattern: importing the module loads no vllm symbols.
        assert "vllm" not in sys.modules
        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401
            vllm_provider,
        )

        assert "vllm" not in sys.modules

    def test_modelapi_registered_as_tl_bridge_vllm(self):
        from transformer_lens.model_bridge.sources.inspect.vllm_provider import (
            PROVIDER_NAME,
        )

        assert PROVIDER_NAME == "tl_bridge_vllm"


class TestConstruction:
    def test_rejects_locked_kwarg_override(self, monkeypatch):
        _fake_tokenizer(monkeypatch)
        _patch_construction(monkeypatch, _fake_hf_config())
        _install_vllm_mocks(monkeypatch, llm_class=MagicMock(name="LLM"))
        from transformer_lens.model_bridge.sources.inspect.vllm_provider import (
            TransformerLensVLLMModelAPI,
        )

        with pytest.raises(ValueError, match="tensor_parallel_size"):
            TransformerLensVLLMModelAPI("any/model", vllm_kwargs={"tensor_parallel_size": 2})

    def test_gated_capture_kind_raises(self, monkeypatch):
        # vLLM's overlay serves resid_post/attn_out/mlp_out; resid_pre and resid_mid are gated.
        provider_factory_args: dict[str, Any] = {"capture": ["blocks.0.hook_in"]}  # resid_pre
        _fake_tokenizer(monkeypatch)
        _patch_construction(monkeypatch, _fake_hf_config())
        _install_vllm_mocks(monkeypatch, llm_class=MagicMock(name="LLM"))
        from transformer_lens.model_bridge.sources.inspect.vllm_provider import (
            TransformerLensVLLMModelAPI,
        )

        with pytest.raises(ValueError, match="resid_pre|gated"):
            TransformerLensVLLMModelAPI("any/model", device="cpu", **provider_factory_args)

    def test_served_capture_kind_accepted(self, monkeypatch):
        # resid_post (blocks.{i}.hook_out) IS served by the vLLM overlay — must not raise.
        _fake_tokenizer(monkeypatch)
        _patch_construction(monkeypatch, _fake_hf_config())
        _install_vllm_mocks(monkeypatch, llm_class=MagicMock(name="LLM"))
        from transformer_lens.model_bridge.sources.inspect.vllm_provider import (
            TransformerLensVLLMModelAPI,
        )

        api = TransformerLensVLLMModelAPI("any/model", device="cpu", capture=["blocks.0.hook_out"])
        assert "0:resid_post" in api._eval_capture

    def test_passes_locked_kwargs_to_llm(self, monkeypatch):
        provider, llm_class = _make_provider(monkeypatch, MagicMock())
        kwargs = llm_class.call_args.kwargs
        assert kwargs["tensor_parallel_size"] == 1
        assert kwargs["skip_tokenizer_init"] is True
        assert kwargs["disable_log_stats"] is True
        # Inc 2: capture wiring requires worker_extension_cls + full-vocab logprobs.
        assert kwargs["worker_extension_cls"].endswith("TLWorkerExtension")
        assert kwargs["max_logprobs"] == 128  # _fake_hf_config().vocab_size
        assert kwargs["max_num_batched_tokens"] == 2048  # default

    def test_supported_kinds_match_overlay(self, monkeypatch):
        provider, _ = _make_provider(monkeypatch, MagicMock())
        assert provider.supported_kinds() == frozenset({"resid_post", "attn_out", "mlp_out"})
        assert "resid_pre" in provider.capability_note()


class TestGenerateEval:
    def _config(self, **overrides):
        from inspect_ai.model import GenerateConfig

        return GenerateConfig(**overrides)

    def _user_msg(self, content="hi"):
        from inspect_ai.model import ChatMessageUser

        return [ChatMessageUser(content=content)]

    def test_generate_eval_basic_completion_and_usage(self, monkeypatch):
        llm = MagicMock()
        llm.generate.return_value = [_make_request_output([65, 66, 67], finish_reason="length")]
        provider, _ = _make_provider(monkeypatch, llm)
        out = asyncio.run(
            provider.generate(self._user_msg("xy"), None, None, self._config(max_tokens=5))
        )
        choice = out.choices[0]
        assert choice.message.text == "ABC"
        assert choice.stop_reason == "max_tokens"  # vLLM 'length' → inspect 'max_tokens'
        assert out.usage.output_tokens == 3
        assert out.usage.total_tokens == out.usage.input_tokens + 3

    def test_generate_eval_finish_stop_maps_to_stop(self, monkeypatch):
        llm = MagicMock()
        llm.generate.return_value = [_make_request_output([65, 1], finish_reason="stop")]
        provider, _ = _make_provider(monkeypatch, llm)
        out = asyncio.run(
            provider.generate(self._user_msg(), None, None, self._config(max_tokens=4))
        )
        assert out.choices[0].stop_reason == "stop"

    def test_generate_eval_with_logprobs(self, monkeypatch):
        vllm_lp = SimpleNamespace(logprob=-0.5, rank=1, decoded_token="A")
        vllm_alt = SimpleNamespace(logprob=-2.0, rank=2, decoded_token="B")
        step = {65: vllm_lp, 66: vllm_alt}
        llm = MagicMock()
        llm.generate.return_value = [
            _make_request_output([65], finish_reason="length", logprobs=[step])
        ]
        provider, _ = _make_provider(monkeypatch, llm)
        out = asyncio.run(
            provider.generate(
                self._user_msg(),
                None,
                None,
                self._config(max_tokens=1, logprobs=True, top_logprobs=2),
            )
        )
        lp = out.choices[0].logprobs
        assert lp is not None and len(lp.content) == 1
        assert lp.content[0].logprob == -0.5
        assert len(lp.content[0].top_logprobs) == 2

    def test_generate_eval_passes_temperature_top_p(self, monkeypatch):
        llm = MagicMock()
        llm.generate.return_value = [_make_request_output([65], finish_reason="length")]
        provider, _ = _make_provider(monkeypatch, llm)
        asyncio.run(
            provider.generate(
                self._user_msg(),
                None,
                None,
                self._config(max_tokens=1, temperature=0.7, top_p=0.9, seed=42),
            )
        )
        # SamplingParams was instantiated; confirm the sampling kwargs flowed through.
        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401
            vllm_provider as _vp,
        )

        sp_cls = sys.modules["vllm"].SamplingParams
        sp_kwargs = sp_cls.call_args.kwargs
        assert sp_kwargs["temperature"] == 0.7
        assert sp_kwargs["top_p"] == 0.9
        assert sp_kwargs["seed"] == 42


class TestCapturePath:
    """TL-driven single-forward capture: extra_args carries input_ids/capture/interventions;
    provider pushes specs → generates → reads captures via collective_rpc."""

    def _config_with(self, extra_args: dict[str, Any]):
        from inspect_ai.model import GenerateConfig

        return GenerateConfig(extra_body={"extra_args": extra_args})

    def _make_capture_provider(self, monkeypatch, *, captures=None, next_token: int = 65):
        import torch

        vllm_lp = SimpleNamespace(logprob=-0.1, rank=1, decoded_token="A")
        request_output = SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[next_token],
                    logprobs=[{next_token: vllm_lp}],
                    finish_reason="stop",
                )
            ],
            prompt_token_ids=[],
        )
        llm = MagicMock()
        llm.generate.return_value = [request_output]
        captures = (
            captures
            if captures is not None
            else {"blocks.0.hook_out": torch.zeros(4, 4, dtype=torch.float32)}
        )

        def rpc(method, args=()):
            return [captures] if method == "tl_read_captures" else [None]

        llm.collective_rpc.side_effect = rpc
        provider, llm_class = _make_provider(monkeypatch, llm)
        return provider, llm, llm_class

    def test_capture_pushes_interventions_and_reads(self, monkeypatch):
        provider, llm, _ = self._make_capture_provider(monkeypatch)
        asyncio.run(
            provider.generate(
                [],
                None,
                None,
                self._config_with(
                    {
                        "input_ids": [10, 20, 30, 40],
                        "capture": ["0:resid_post"],
                        "interventions": {"0:resid_post": {"op": "suppress"}},
                    }
                ),
            )
        )
        # tl_set_interventions runs first (resets stale state), then the read.
        methods = [c.args[0] for c in llm.collective_rpc.call_args_list]
        assert methods == ["tl_set_interventions", "tl_read_captures"]
        # Specs translated wire key → TL hook name (worker is keyed by hook name).
        set_call = llm.collective_rpc.call_args_list[0]
        assert set_call.kwargs["args"] == ({"blocks.0.hook_out": {"op": "suppress"}},)
        # Read takes [prompt_lens] + hook names so the worker slices the GPU buffer.
        read_call = llm.collective_rpc.call_args_list[1]
        assert read_call.kwargs["args"] == ([4], ["blocks.0.hook_out"])

    def test_capture_returns_wire_format_activations(self, monkeypatch):
        import torch

        captures = {"blocks.0.hook_out": torch.full((3, 4), 0.5, dtype=torch.float32)}
        provider, _, _ = self._make_capture_provider(monkeypatch, captures=captures)
        out = asyncio.run(
            provider.generate(
                [],
                None,
                None,
                self._config_with({"input_ids": [1, 2, 3], "capture": ["0:resid_post"]}),
            )
        )
        assert "0:resid_post" in out.metadata["activations"]
        # Wire envelope is {"data": b64, "dtype": str, "shape": list} — driver decodes verbatim.
        entry = out.metadata["activations"]["0:resid_post"]
        assert entry["shape"] == [3, 4] and entry["dtype"] == "float32"

    def test_capture_synthesizes_logits_when_requested(self, monkeypatch):
        provider, _, _ = self._make_capture_provider(monkeypatch)
        out = asyncio.run(
            provider.generate(
                [],
                None,
                None,
                self._config_with(
                    {
                        "input_ids": [10, 20, 30],
                        "capture": ["0:resid_post"],
                        "return_logits": True,
                    }
                ),
            )
        )
        # logits shape (n_tokens, d_vocab); last position holds vLLM's logprobs at the gen step.
        entry = out.metadata["tl_logits"]
        assert entry["shape"] == [3, 128]

    def test_capture_skips_logits_when_disabled(self, monkeypatch):
        provider, _, _ = self._make_capture_provider(monkeypatch)
        out = asyncio.run(
            provider.generate(
                [],
                None,
                None,
                self._config_with(
                    {
                        "input_ids": [10, 20, 30],
                        "capture": ["0:resid_post"],
                        "return_logits": False,
                    }
                ),
            )
        )
        assert "tl_logits" not in out.metadata

    def test_capture_gated_kind_raises(self, monkeypatch):
        # resid_pre (blocks.{i}.hook_in) is gated by vLLM's fused execution.
        provider, _, _ = self._make_capture_provider(monkeypatch)
        with pytest.raises(ValueError, match="resid_pre|gated"):
            asyncio.run(
                provider.generate(
                    [],
                    None,
                    None,
                    self._config_with({"input_ids": [1, 2], "capture": ["0:resid_pre"]}),
                )
            )

    def test_capture_unparseable_wire_key_raises(self, monkeypatch):
        # Kind is served, but the layer prefix is non-numeric — name_from_wire_key returns None.
        provider, _, _ = self._make_capture_provider(monkeypatch)
        with pytest.raises(ValueError, match="unrecognised wire keys"):
            asyncio.run(
                provider.generate(
                    [],
                    None,
                    None,
                    self._config_with({"input_ids": [1, 2], "capture": ["abc:resid_post"]}),
                )
            )

    def test_intervention_unknown_wire_key_raises(self, monkeypatch):
        provider, _, _ = self._make_capture_provider(monkeypatch)
        with pytest.raises(ValueError, match="not a fireable hook"):
            asyncio.run(
                provider.generate(
                    [],
                    None,
                    None,
                    self._config_with(
                        {
                            "input_ids": [1, 2],
                            "capture": ["0:resid_post"],
                            "interventions": {"abc:resid_post": {"op": "suppress"}},
                        }
                    ),
                )
            )

    def test_prompt_exceeds_max_batched_tokens_raises(self, monkeypatch):
        provider, _, _ = self._make_capture_provider(monkeypatch)
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            asyncio.run(
                provider.generate(
                    [],
                    None,
                    None,
                    self._config_with(
                        {"input_ids": list(range(3000)), "capture": ["0:resid_post"]}
                    ),
                )
            )

    def test_capture_completion_decodes_generated_token(self, monkeypatch):
        # The ModelOutput choice's text must be the decoded next token (parity with HF provider).
        provider, _, _ = self._make_capture_provider(monkeypatch, next_token=72)
        out = asyncio.run(
            provider.generate(
                [],
                None,
                None,
                self._config_with({"input_ids": [1, 2], "capture": ["0:resid_post"]}),
            )
        )
        # The fake tokenizer decodes by joining chr(id) — 72 → 'H'.
        assert out.choices[0].message.text == "H"


class TestPerTurnCapture:
    """``model_args['capture']`` triggers a snapshot-via-single-token-forward before every
    eval generate (vLLM's decode steps overwrite buffer row 0, so the HF first-write-wins
    trick can't apply); the snapshot lands in ``ModelOutput.metadata`` for agent rollouts."""

    def _user_msg(self, content: str = "hi"):
        from inspect_ai.model import ChatMessageUser

        return [ChatMessageUser(content=content)]

    def _make_provider_with_capture(self, monkeypatch, capture: list[str]):
        import torch

        # llm.generate runs twice per eval (snapshot then real generate); the same fake
        # output works for both. collective_rpc returns canned captures for tl_read_captures.
        llm = MagicMock()
        llm.generate.return_value = [_make_request_output([65, 66], finish_reason="length")]
        captures = {"blocks.0.hook_out": torch.zeros(4, 4, dtype=torch.float32)}

        def rpc(method, args=()):
            return [captures] if method == "tl_read_captures" else [None]

        llm.collective_rpc.side_effect = rpc

        _fake_tokenizer(monkeypatch)
        _patch_construction(monkeypatch, _fake_hf_config())
        llm_class = MagicMock(return_value=llm, name="LLM")
        _install_vllm_mocks(monkeypatch, llm_class)
        from transformer_lens.model_bridge.sources.inspect.vllm_provider import (
            TransformerLensVLLMModelAPI,
        )

        return TransformerLensVLLMModelAPI("any/model", device="cpu", capture=capture), llm

    def test_eval_capture_snapshots_before_generate(self, monkeypatch):
        from inspect_ai.model import GenerateConfig

        provider, llm = self._make_provider_with_capture(monkeypatch, ["blocks.0.hook_out"])
        out = asyncio.run(
            provider.generate(self._user_msg(), None, None, GenerateConfig(max_tokens=4))
        )
        # Snapshot fires: empty interventions (resets stale state) then a read.
        methods = [c.args[0] for c in llm.collective_rpc.call_args_list]
        assert methods == ["tl_set_interventions", "tl_read_captures"]
        set_call = llm.collective_rpc.call_args_list[0]
        assert set_call.kwargs["args"] == ({},)
        # llm.generate called twice: once for the snapshot, once for the real eval.
        assert llm.generate.call_count == 2
        # Snapshot lands in metadata as the same wire format the InspectDriver consumes.
        assert out.metadata is not None and "0:resid_post" in out.metadata["activations"]

    def test_no_capture_no_metadata_and_no_rpc(self, monkeypatch):
        from inspect_ai.model import GenerateConfig

        llm = MagicMock()
        llm.generate.return_value = [_make_request_output([65], finish_reason="stop")]
        provider, _ = _make_provider(monkeypatch, llm)
        out = asyncio.run(
            provider.generate(self._user_msg(), None, None, GenerateConfig(max_tokens=1))
        )
        # No model_args['capture'] → no snapshot, no metadata, no collective_rpc fires.
        assert out.metadata is None
        assert not llm.collective_rpc.called

    def test_eval_capture_emits_eval_completion_too(self, monkeypatch):
        # The eval still returns the real eval generate's completion + usage — snapshot is
        # an additive metadata side channel, not a substitute for the eval output.
        from inspect_ai.model import GenerateConfig

        provider, _ = self._make_provider_with_capture(monkeypatch, ["blocks.0.hook_out"])
        out = asyncio.run(
            provider.generate(self._user_msg(), None, None, GenerateConfig(max_tokens=4))
        )
        # _make_request_output([65, 66]) ⇒ "AB" via fake tokenizer.
        assert out.choices[0].message.text == "AB"
        assert out.usage.output_tokens == 2
