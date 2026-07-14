"""Orchestration tests for ``boot_vllm`` — mocks the HF / vLLM boundaries.

Covers locked-kwarg rejection, dtype resolution, HF_TOKEN plumbing, env-var
override warning, plugin-config lifecycle, and the happy path. Driver-level
behavior is covered separately in test_vllm_driver.py.
"""
from __future__ import annotations

import os
import sys
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources.vllm import plugin
from transformer_lens.model_bridge.sources.vllm.source import (
    _dtype_from_hf_config,
    boot_vllm,
)


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        torch_dtype=torch.float16,
        hidden_size=4,
        vocab_size=16,
        num_hidden_layers=2,
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


@pytest.fixture
def mocked_boot(monkeypatch):
    """Mock every external boundary boot_vllm crosses; yield handles for assertions."""
    plugin._config.clear()
    hf_config = _hf_config()
    cfg = _cfg()
    adapter = ArchitectureAdapter(cfg)
    adapter.component_mapping = {}

    auto_config = MagicMock(return_value=hf_config)
    auto_tokenizer = MagicMock(return_value=MagicMock(name="tokenizer"))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", auto_config)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", auto_tokenizer)

    fake_llm = MagicMock(name="llm")
    # The boot-time reconstruction probe fetches the unembedding via get_param, whose
    # return type is beartype-enforced — the RPC must yield a real tensor.
    fake_llm.collective_rpc = MagicMock(return_value=[torch.ones(16, 4)])
    fake_vllm = MagicMock()
    fake_vllm.LLM = MagicMock(return_value=fake_llm)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.get_hf_token",
        lambda: "fake-token",
    )
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.extract_hf_config",
        lambda llm: hf_config,
    )
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.build_bridge_config_from_hf",
        lambda hf, arch, name, dt: cfg,
    )
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source"
        ".ArchitectureAdapterFactory.select_architecture_adapter",
        lambda c: adapter,
    )
    # Skip the real monkey-patch of Worker.load_model — only entry-points discovery
    # would import vllm.v1.worker.gpu_worker. Leave configure() unmocked so the
    # clear() test exercises real state.
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.plugin.register",
        lambda: None,
    )

    # Real tokenizer setup/probing can't run on a MagicMock tokenizer.
    def _fake_configure_tokenizer(tokenizer, cfg_):
        cfg_.tokenizer_prepends_bos, cfg_.tokenizer_appends_eos = (False, True)
        return tokenizer

    configure_tok = MagicMock(side_effect=_fake_configure_tokenizer)
    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.vllm.source.configure_tokenizer",
        configure_tok,
    )

    yield {
        "auto_config": auto_config,
        "auto_tokenizer": auto_tokenizer,
        "vllm_llm": fake_vllm.LLM,
        "hf_config": hf_config,
        "cfg": cfg,
        "configure_tok": configure_tok,
    }

    plugin._config.clear()


def test_rejects_locked_kwarg_override():
    """tensor_parallel_size != 1 fails fast — before any I/O."""
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        boot_vllm("any-model", tensor_parallel_size=2)


def test_rejects_position_interventions_with_batching():
    """Position interventions need the compiled path's affine buffers — fails fast."""
    with pytest.raises(ValueError, match="incompatible with enable_batching"):
        boot_vllm("any-model", enable_position_interventions=True, enable_batching=True)


def test_position_interventions_flag_reaches_driver(mocked_boot):
    """boot_vllm threads enable_position_interventions through to the driver."""
    bridge = boot_vllm("any-model", enable_position_interventions=True)
    assert bridge._driver._enable_position_interventions is True
    # Default stays off.
    assert boot_vllm("any-model")._driver._enable_position_interventions is False


@pytest.mark.parametrize(
    "raw, expected",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("nonexistent_dtype", torch.float16),
        (None, torch.float16),
    ],
)
def test_dtype_resolution(raw, expected):
    assert _dtype_from_hf_config(SimpleNamespace(torch_dtype=raw)) == expected


def test_dtype_resolution_missing_attr():
    assert _dtype_from_hf_config(SimpleNamespace()) == torch.float16


def test_happy_path_returns_remote_bridge(mocked_boot):
    bridge = boot_vllm("any-model")
    assert isinstance(bridge, RemoteBridge)


def test_hf_token_passed_to_both_hf_calls(mocked_boot):
    boot_vllm("any-model")
    assert mocked_boot["auto_config"].call_args.kwargs["token"] == "fake-token"
    assert mocked_boot["auto_tokenizer"].call_args.kwargs["token"] == "fake-token"


def test_custom_tokenizer_skips_autotokenizer(mocked_boot):
    boot_vllm("any-model", tokenizer=MagicMock(name="custom"))
    mocked_boot["auto_tokenizer"].assert_not_called()


def test_env_var_override_warns(mocked_boot, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    with pytest.warns(UserWarning, match="VLLM_ENABLE_V1_MULTIPROCESSING"):
        boot_vllm("any-model")
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_env_var_zero_does_not_warn(mocked_boot, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        boot_vllm("any-model")
    assert not any("VLLM_ENABLE_V1_MULTIPROCESSING" in str(w.message) for w in caught)


def test_plugin_config_cleared_after_boot(mocked_boot):
    """No leak to non-TL vllm.LLM users in the same process."""
    boot_vllm("any-model")
    assert plugin._config == {}


def test_plugin_config_cleared_when_llm_construction_fails(mocked_boot):
    """A failed boot (OOM, gated repo) must not leave stale specs patched in —
    the next in-process vllm.LLM(...) would walk our dot-paths on a foreign model."""
    mocked_boot["vllm_llm"].side_effect = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        boot_vllm("any-model")
    assert plugin._config == {}


def test_rejects_prefix_caching_override():
    """Prefix caching breaks the row=position capture invariant — locked off."""
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        boot_vllm("any-model", enable_prefix_caching=True)


def test_bos_detection_written_to_config(mocked_boot):
    """boot_vllm must probe the tokenizer like boot_transformers — the dataclass
    default (prepends_bos=True) is wrong for Qwen-family tokenizers and shifts
    every activation by one position."""
    bridge = boot_vllm("any-model")
    assert mocked_boot["configure_tok"].called
    assert mocked_boot["cfg"].tokenizer_prepends_bos is False
    assert mocked_boot["cfg"].tokenizer_appends_eos is True
    assert bridge is not None


def test_logit_reconstruction_probed_at_boot(mocked_boot):
    """The unembedding is fetched once at boot, not re-cloned per forward."""
    bridge = boot_vllm("any-model")
    assert bridge._driver._unembed_probed is True
    assert bridge._driver._unembed is not None


def test_llm_construction_kwargs(mocked_boot):
    """Pin the kwargs boot_vllm passes to vllm.LLM(...). Catches regressions like
    forgetting worker_extension_cls (collective_rpc methods unreachable),
    max_num_batched_tokens (Dynamo symbolic-shape bound mismatch with buffer),
    or max_logprobs (driver synthesizes logits via full-vocab logprobs)."""
    boot_vllm("any-model", max_num_batched_tokens=1024)
    kwargs = mocked_boot["vllm_llm"].call_args.kwargs
    assert kwargs["model"] == "any-model"
    assert kwargs["max_num_batched_tokens"] == 1024
    assert kwargs["worker_extension_cls"] == (
        "transformer_lens.model_bridge.sources.vllm.worker_extension.TLWorkerExtension"
    )
    # Sized from the mocked hf_config.vocab_size in mocked_boot fixture.
    assert kwargs["max_logprobs"] == mocked_boot["hf_config"].vocab_size
    # Locked kwargs that must always reach LLM.
    assert kwargs["tensor_parallel_size"] == 1
    assert kwargs["pipeline_parallel_size"] == 1
    assert kwargs["skip_tokenizer_init"] is True
    assert kwargs["disable_log_stats"] is True
    assert kwargs["enable_prefix_caching"] is False
    # Always explicit — "auto" would downcast fp32 checkpoints under the buffers.
    assert kwargs["dtype"] == "float16"
