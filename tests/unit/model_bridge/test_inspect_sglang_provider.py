"""Unit tests for the SGLang-backed Inspect provider (``tl_bridge_sglang``).

Mocks ``sglang.srt.entrypoints.engine.Engine`` + :class:`CapturePuller` + the
``extract_hf_config`` walk so no install is needed. Live verification (real
SGLang, real GPU) is in the Colab walkthrough."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("inspect_ai")


def _install_sglang_mocks(monkeypatch, engine_class: Any) -> None:
    """Install fake ``sglang.srt.entrypoints.engine`` module so the provider's
    ``from sglang.srt.entrypoints.engine import Engine`` resolves to our mock."""
    sglang_pkg = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    entrypoints = types.ModuleType("sglang.srt.entrypoints")
    engine_mod = types.ModuleType("sglang.srt.entrypoints.engine")
    engine_mod.Engine = engine_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", engine_mod)


def _fake_tokenizer(monkeypatch) -> Any:
    tok = SimpleNamespace(chat_template=None, pad_token_id=0, eos_token_id=1)

    def encode(text: str) -> list[int]:
        return [ord(c) % 50 for c in str(text)[:32]]

    callable_tok = MagicMock(wraps=tok)
    callable_tok.side_effect = lambda text, **kw: SimpleNamespace(input_ids=encode(text))
    callable_tok.decode = lambda ids, **kw: "".join(chr(int(i)) for i in ids)
    callable_tok.chat_template = None
    callable_tok.pad_token_id = 0
    callable_tok.eos_token_id = 1

    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: callable_tok)
    return callable_tok


def _fake_hf_config(n_layers: int = 2, d_model: int = 8, vocab_size: int = 128) -> Any:
    return SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        hidden_size=d_model,
        num_hidden_layers=n_layers,
        vocab_size=vocab_size,
        torch_dtype="float16",
    )


def _patch_construction(monkeypatch, hf_config: Any, puller_messages: list | None = None):
    """Patch AutoConfig + assert_sglang_supported + extract_hf_config + CapturePuller.

    Returns a recording handle for the puller so tests can assert close/drain order.
    """
    from transformers import AutoConfig

    from transformer_lens.model_bridge.sources.sglang import internals

    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: hf_config)
    monkeypatch.setattr(internals, "assert_sglang_supported", lambda: None)
    monkeypatch.setattr(internals, "extract_hf_config", lambda engine: hf_config)

    puller_state: dict[str, Any] = {"messages": list(puller_messages or []), "closed": False}

    class _FakePuller:
        def __init__(self, channel: str) -> None:
            self.channel = channel

        def drain(self, timeout_ms: int = 1000):
            msgs = puller_state["messages"]
            puller_state["messages"] = []
            return msgs

        def close(self) -> None:
            puller_state["closed"] = True

    from transformer_lens.model_bridge.sources.inspect import sglang_provider
    from transformer_lens.model_bridge.sources.sglang import capture_puller as cp_module

    monkeypatch.setattr(
        sglang_provider, "__name__", sglang_provider.__name__
    )  # touch to ensure module loaded
    monkeypatch.setattr(cp_module, "CapturePuller", _FakePuller)

    return puller_state


def _make_provider(
    monkeypatch,
    engine_instance: Any,
    *,
    hf_config: Any | None = None,
    puller_messages: list | None = None,
):
    """Construct the provider with mocked sglang.Engine + tokenizer + CapturePuller."""
    _fake_tokenizer(monkeypatch)
    hf_config = hf_config or _fake_hf_config()
    puller_state = _patch_construction(monkeypatch, hf_config, puller_messages)
    engine_class = MagicMock(return_value=engine_instance, name="Engine")
    _install_sglang_mocks(monkeypatch, engine_class)
    from transformer_lens.model_bridge.sources.inspect.sglang_provider import (
        TransformerLensSGLangModelAPI,
    )

    provider = TransformerLensSGLangModelAPI("any/model", device="cpu")
    return provider, engine_class, puller_state


class TestModuleSafety:
    def test_module_imports_without_sglang(self):
        assert "sglang" not in sys.modules
        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401
            sglang_provider,
        )

        assert "sglang" not in sys.modules


class TestConstruction:
    def test_rejects_locked_kwarg_override(self, monkeypatch):
        _fake_tokenizer(monkeypatch)
        _patch_construction(monkeypatch, _fake_hf_config())
        _install_sglang_mocks(monkeypatch, MagicMock(name="Engine"))
        from transformer_lens.model_bridge.sources.inspect.sglang_provider import (
            TransformerLensSGLangModelAPI,
        )

        with pytest.raises(ValueError, match="tp_size"):
            TransformerLensSGLangModelAPI("any/model", device="cpu", sglang_kwargs={"tp_size": 2})

    def test_passes_locked_kwargs_and_forward_hooks(self, monkeypatch):
        provider, engine_class, _ = _make_provider(monkeypatch, MagicMock())
        kwargs = engine_class.call_args.kwargs
        assert kwargs["tp_size"] == 1
        assert kwargs["dp_size"] == 1
        assert kwargs["skip_tokenizer_init"] is True
        assert kwargs["model_path"] == "any/model"
        # forward_hooks list built from the overlay's capture_specs.
        assert isinstance(kwargs["forward_hooks"], list)
        assert kwargs["forward_hooks"]  # non-empty for a 2-layer config
        # Every entry points at our factory and carries a channel.
        for spec in kwargs["forward_hooks"]:
            assert spec["hook_factory"].endswith(":make_capture_hook")
            assert spec["config"]["channel"].startswith("ipc://")

    def test_disable_cuda_graph_default(self, monkeypatch):
        """Default True — full CUDA graph capture would silently skip our hooks."""
        provider, engine_class, _ = _make_provider(monkeypatch, MagicMock())
        assert engine_class.call_args.kwargs["disable_cuda_graph"] is True

    def test_supported_kinds_match_overlay(self, monkeypatch):
        provider, _, _ = _make_provider(monkeypatch, MagicMock())
        assert provider.supported_kinds() == frozenset({"resid_post", "attn_out", "mlp_out"})
        assert "resid_pre" in provider.capability_note()


class TestCapturePath:
    """TL-driven single-token capture: extra_args carries input_ids/capture/interventions;
    provider drives engine.collective_rpc + drains the puller."""

    def _config_with(self, extra_args: dict[str, Any]):
        from inspect_ai.model import GenerateConfig

        return GenerateConfig(extra_body={"extra_args": extra_args})

    def _make_capture_provider(self, monkeypatch, *, puller_messages=None, next_token: int = 65):
        engine = MagicMock()
        engine.generate.return_value = [
            {
                "output_ids": [next_token],
                "meta_info": {"output_top_logprobs": [[(-0.1, next_token, "A")]]},
            }
        ]
        provider, engine_class, puller_state = _make_provider(
            monkeypatch, engine, puller_messages=puller_messages
        )
        return provider, engine, puller_state

    def test_capture_rpc_sequence(self, monkeypatch):
        import asyncio

        import torch

        captures = [{"name": "blocks.0.hook_out", "tensor": torch.zeros(4, 8, dtype=torch.float32)}]
        provider, engine, _ = self._make_capture_provider(monkeypatch, puller_messages=captures)
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
        # tl_set_interventions → tl_set_capture_enabled(True) → generate → tl_set_capture_enabled(False).
        methods = [c.args[0] for c in engine.collective_rpc.call_args_list]
        assert methods == [
            "tl_set_interventions",
            "tl_set_capture_enabled",
            "tl_set_capture_enabled",
        ]
        # First set_interventions translates wire key → TL hook name.
        first = engine.collective_rpc.call_args_list[0]
        assert first.kwargs["specs"] == {"blocks.0.hook_out": {"op": "suppress"}}
        # Enable then disable.
        enabled = [c.kwargs["enabled"] for c in engine.collective_rpc.call_args_list[1:]]
        assert enabled == [True, False]

    def test_capture_gated_kind_raises(self, monkeypatch):
        import asyncio

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
        import asyncio

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
        import asyncio

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
        import asyncio

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
