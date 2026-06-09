"""Unit tests for the SGLang-backed Inspect provider (``tl_bridge_sglang``).

Mocks SGLang so no install is needed. Live end-to-end verification (real SGLang,
real GPU) is in the Colab walkthrough; the mocked tests pin the construction
sequence, generate→ModelOutput conversion, and validation surface.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# The provider lazy-imports inspect_ai inside method bodies; provider construction
# in the fixtures also pulls inspect_ai transitively. Skip when the ``inspect`` extra
# is absent rather than fail each test with ModuleNotFoundError.
pytest.importorskip("inspect_ai")


def _install_sglang_mocks(monkeypatch, engine_class: Any) -> None:
    """Install fake ``sglang``/``sglang.srt.entrypoints.engine`` modules in ``sys.modules``
    so the provider's lazy ``from sglang.srt.entrypoints.engine import Engine`` resolves."""
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


def _patch_construction(monkeypatch, hf_config: Any) -> None:
    """Patch AutoConfig + assert_sglang_supported + plugin.configure/register +
    extract_hf_config so ``__init__`` runs without sglang / real worker processes."""
    from transformers import AutoConfig

    from transformer_lens.model_bridge.sources.sglang import internals, plugin

    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: hf_config)
    monkeypatch.setattr(internals, "assert_sglang_supported", lambda: None)
    monkeypatch.setattr(plugin, "configure", lambda **kw: None)
    monkeypatch.setattr(plugin, "register", lambda: None)
    monkeypatch.setattr(internals, "extract_hf_config", lambda engine: hf_config)


def _make_provider(monkeypatch, engine_instance: Any, hf_config: Any | None = None):
    """Construct the provider with mocked sglang.Engine + fake tokenizer + patched
    construction; device='cpu' so intermediate tensors don't require CUDA."""
    _fake_tokenizer(monkeypatch)
    hf_config = hf_config or _fake_hf_config()
    _patch_construction(monkeypatch, hf_config)
    engine_class = MagicMock(return_value=engine_instance, name="Engine")
    _install_sglang_mocks(monkeypatch, engine_class)
    from transformer_lens.model_bridge.sources.inspect.sglang_provider import (
        TransformerLensSGLangModelAPI,
    )

    return TransformerLensSGLangModelAPI("any/model", device="cpu"), engine_class


class TestModuleSafety:
    def test_module_imports_without_sglang(self):
        # Confirm the lazy-sglang pattern: importing the module loads no sglang symbols.
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

    def test_passes_locked_kwargs_to_engine(self, monkeypatch):
        provider, engine_class = _make_provider(monkeypatch, MagicMock())
        kwargs = engine_class.call_args.kwargs
        assert kwargs["tp_size"] == 1
        assert kwargs["dp_size"] == 1
        assert kwargs["skip_tokenizer_init"] is True
        assert kwargs["model_path"] == "any/model"

    def test_supported_kinds_match_overlay(self, monkeypatch):
        provider, _ = _make_provider(monkeypatch, MagicMock())
        # Decoder-only overlay serves resid_post / attn_out / mlp_out; resid_pre and
        # the derived resid_mid are gated.
        assert provider.supported_kinds() == frozenset({"resid_post", "attn_out", "mlp_out"})
        assert "resid_pre" in provider.capability_note()


class TestCapturePath:
    """TL-driven single-forward capture: extra_args carries input_ids/capture/interventions;
    provider pushes specs → generates → reads captures via the RPC module."""

    def _config_with(self, extra_args: dict[str, Any]):
        from inspect_ai.model import GenerateConfig

        return GenerateConfig(extra_body={"extra_args": extra_args})

    def _make_capture_provider(self, monkeypatch, *, captures=None, next_token: int = 65):
        import torch

        engine = MagicMock()
        engine.generate.return_value = [
            {
                "token_ids": [next_token],
                "meta_info": {"output_top_logprobs": [[(-0.1, next_token, "A")]]},
            }
        ]
        captures = (
            captures
            if captures is not None
            else {"blocks.0.hook_out": torch.zeros(4, 8, dtype=torch.float32)}
        )

        # Patch the rpc module's methods directly on the provider instance after build.
        provider, engine_class = _make_provider(monkeypatch, engine)

        rpc_calls: list[tuple[str, Any]] = []

        def _set_interventions(engine, specs):
            rpc_calls.append(("set_interventions", specs))

        def _reset_flags(engine):
            rpc_calls.append(("reset_capture_flags", None))

        def _read(engine, method, prompt_lens, names):
            rpc_calls.append(("read_captures", (prompt_lens, names)))
            return captures

        # The provider's _rpc handle was set at __init__; patch the module functions.
        provider._rpc = SimpleNamespace(
            set_interventions=_set_interventions,
            reset_capture_flags=_reset_flags,
            call_with_prompt_lens=_read,
        )

        return provider, engine, rpc_calls

    def test_capture_pushes_interventions_and_reads(self, monkeypatch):
        import asyncio

        provider, engine, rpc_calls = self._make_capture_provider(monkeypatch)
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
        # set_interventions runs first (resets stale state), then reset_capture_flags, then read.
        methods = [name for name, _ in rpc_calls]
        assert methods == ["set_interventions", "reset_capture_flags", "read_captures"]
        # Specs translated wire key → TL hook name (worker is keyed by hook name).
        assert rpc_calls[0][1] == {"blocks.0.hook_out": {"op": "suppress"}}
        # Read takes (prompt_lens, hook names).
        assert rpc_calls[2][1] == ([4], ["blocks.0.hook_out"])

    def test_capture_gated_kind_raises(self, monkeypatch):
        import asyncio

        # resid_pre (blocks.{i}.hook_in) is gated by SGLang's fused execution.
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
