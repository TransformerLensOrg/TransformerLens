"""Shared mock scaffold for ``boot_vllm`` orchestration tests.

Mocks every external boundary boot_vllm crosses (HF Hub fetches, the ``vllm``
package, token/config/tokenizer plumbing) so tests exercise the real boot
orchestration without a GPU or the vllm extra. ``plugin.configure()`` stays
real by default so plugin-state lifecycle tests assert on real state; callers
own ``plugin.clear_config()`` setup/teardown.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from tests.mocks.mock_logits_bridge import mock_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter

_SOURCE = "transformer_lens.model_bridge.sources.vllm.source"


def boot_cfg(**overrides) -> TransformerBridgeConfig:
    """Tiny bridge config with the dims the boot tests assume (two layers, Llama)."""
    overrides.setdefault("n_layers", 2)
    overrides.setdefault("architecture", "LlamaForCausalLM")
    return mock_bridge_cfg(**overrides)


def boot_hf_config(**overrides) -> SimpleNamespace:
    """HF config stub matching boot_cfg's dims."""
    params = dict(
        architectures=["LlamaForCausalLM"],
        torch_dtype=torch.float16,
        hidden_size=4,
        vocab_size=16,
        num_hidden_layers=2,
    )
    params.update(overrides)
    return SimpleNamespace(**params)


def stub_adapter(cfg: TransformerBridgeConfig) -> ArchitectureAdapter:
    """Adapter with an empty component mapping — boot only walks its dot-paths."""
    adapter = ArchitectureAdapter(cfg)
    adapter.component_mapping = {}
    return adapter


def fake_collective_rpc(absent_per_rank):
    """Method-dispatch collective_rpc stub: tl_absent_hooks feeds the boot-time
    coverage check (list per rank); every other method returns a real tensor
    (the reconstruction probe's get_param is beartype-enforced)."""

    def rpc(method, *args, **kwargs):
        if method == "tl_absent_hooks":
            return absent_per_rank
        return [torch.ones(16, 4)]

    return MagicMock(side_effect=rpc)


def fake_configure_tokenizer(tokenizer, cfg_):
    """Real tokenizer setup/probing can't run on a MagicMock tokenizer."""
    cfg_.tokenizer_prepends_bos, cfg_.tokenizer_appends_eos = (False, True)
    return tokenizer


def mock_hf_hub(monkeypatch, hf_config):
    """Patch the HF Hub boundary: config fetch + tokenizer fetch."""
    auto_config = MagicMock(return_value=hf_config)
    auto_tokenizer = MagicMock(return_value=MagicMock(name="tokenizer"))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", auto_config)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", auto_tokenizer)
    return auto_config, auto_tokenizer


def mocked_vllm_boot(monkeypatch, *, hf_config=None, cfg=None, mock_plugin_configure=False):
    """Mock every external boundary boot_vllm crosses; return handles for assertions.

    ``plugin.register`` is always stubbed — only entry-points discovery would
    import vllm.v1.worker.gpu_worker. ``plugin.configure`` stays real unless
    ``mock_plugin_configure=True``, so clear-state tests exercise real state.
    """
    hf_config = boot_hf_config() if hf_config is None else hf_config
    cfg = boot_cfg() if cfg is None else cfg
    adapter = stub_adapter(cfg)
    auto_config, auto_tokenizer = mock_hf_hub(monkeypatch, hf_config)

    fake_llm = MagicMock(name="llm")
    fake_llm.collective_rpc = fake_collective_rpc([[]])
    fake_vllm = MagicMock()
    fake_vllm.LLM = MagicMock(return_value=fake_llm)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    monkeypatch.setattr(f"{_SOURCE}.get_hf_token", lambda: "fake-token")
    monkeypatch.setattr(f"{_SOURCE}.extract_hf_config", lambda llm: hf_config)
    build_cfg = MagicMock(return_value=cfg)
    monkeypatch.setattr(f"{_SOURCE}.build_bridge_config_from_hf", build_cfg)
    monkeypatch.setattr(
        f"{_SOURCE}.ArchitectureAdapterFactory.select_architecture_adapter",
        lambda c: adapter,
    )
    monkeypatch.setattr(f"{_SOURCE}.plugin.register", lambda: None)
    configure_tok = MagicMock(side_effect=fake_configure_tokenizer)
    monkeypatch.setattr(f"{_SOURCE}.configure_tokenizer", configure_tok)

    handles = {
        "auto_config": auto_config,
        "auto_tokenizer": auto_tokenizer,
        "vllm_llm": fake_vllm.LLM,
        "hf_config": hf_config,
        "cfg": cfg,
        "build_cfg": build_cfg,
        "configure_tok": configure_tok,
    }
    if mock_plugin_configure:
        plugin_configure = MagicMock()
        monkeypatch.setattr(f"{_SOURCE}.plugin.configure", plugin_configure)
        handles["plugin_configure"] = plugin_configure
    return handles
