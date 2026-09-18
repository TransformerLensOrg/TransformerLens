"""Adapter prepare_loading orchestration on fake remote modules (raven, rwkv7).

The shared patch mechanics are unit-tested in test_remote_code_compat.py; these
tests cover each adapter's orchestration — that a modeling module planted in
sys.modules gets both v5 patches applied (tied-weights dict rewrite, guarded
_init_weights) and that the flow is idempotent. Real trust_remote_code loads
are CI-invisible, so this is the only coverage those load-bearing patches get.
"""

import importlib
import sys
import types

import pytest
import torch


def _make_fake_modeling_module(module_name, causal_name, pretrained_name, tied_to):
    module = types.ModuleType(module_name)

    class FakePreTrainedModel:
        def _init_weights(self, mod):
            mod.initialized = True

    class FakeCausalLM(FakePreTrainedModel):
        _tied_weights_keys = ["lm_head.weight"]

    FakePreTrainedModel.__name__ = pretrained_name
    FakeCausalLM.__name__ = causal_name
    setattr(module, causal_name, FakeCausalLM)
    setattr(module, pretrained_name, FakePreTrainedModel)
    module._expected_tied_mapping = {"lm_head.weight": tied_to}
    return module


@pytest.fixture
def planted_module(request, monkeypatch):
    """Plant a fake remote modeling module for the requested arch fragment."""
    fragment, causal, pretrained, tied_to = request.param
    name = f"transformers_modules.fake.modeling_{fragment}"
    module = _make_fake_modeling_module(name, causal, pretrained, tied_to)
    monkeypatch.setitem(sys.modules, name, module)
    return module


_CASES = [
    pytest.param(
        ("raven", "RavenForCausalLM", "RavenPreTrainedModel", "transformer.wte.weight"),
        id="raven",
    ),
    pytest.param(
        ("rwkv7", "RWKV7ForCausalLM", "RWKV7PreTrainedModel", "model.embeddings.weight"),
        id="rwkv7",
    ),
]


def _adapter_for(fragment):
    from transformer_lens.config import TransformerBridgeConfig

    cfg = TransformerBridgeConfig(
        n_layers=1, d_model=8, d_head=2, n_heads=4, d_vocab=16, n_ctx=8, act_fn="gelu"
    )
    if fragment == "raven":
        from transformer_lens.model_bridge.supported_architectures.raven import (
            RavenArchitectureAdapter,
        )

        cfg.architecture = "RavenForCausalLM"
        return RavenArchitectureAdapter(cfg)
    from transformer_lens.model_bridge.supported_architectures.rwkv7 import (
        RWKV7ArchitectureAdapter,
    )

    cfg.architecture = "RWKV7ForCausalLM"
    return RWKV7ArchitectureAdapter(cfg)


@pytest.mark.parametrize("planted_module", _CASES, indirect=True)
def test_prepare_loading_applies_both_v5_patches(planted_module, monkeypatch):
    fragment = planted_module.__name__.rsplit("modeling_", 1)[-1]
    causal = [n for n in vars(planted_module) if n.endswith("ForCausalLM")][0]
    pretrained = [n for n in vars(planted_module) if n.endswith("PreTrainedModel")][0]

    # Force the fallback path deterministically: block a real `fla` install from
    # short-circuiting the rwkv7 case (and patching genuine fla classes), and
    # stub the Hub-dependent force-import (the module is already planted).
    monkeypatch.setitem(sys.modules, "fla", None)
    adapter_mod = importlib.import_module(
        f"transformer_lens.model_bridge.supported_architectures.{fragment}"
    )
    monkeypatch.setattr(adapter_mod, "force_import_remote_class", lambda *a, **k: object)

    adapter = _adapter_for(fragment)
    adapter.prepare_loading("fake/model", {})

    causal_cls = getattr(planted_module, causal)
    pretrained_cls = getattr(planted_module, pretrained)

    # Patch 1: list-form tied weights rewritten to the v5 dict form.
    assert causal_cls._tied_weights_keys == planted_module._expected_tied_mapping

    # Patch 2: _init_weights guarded — materialized modules are not re-initialized.
    assert getattr(pretrained_cls, "_tl_patched", False)
    loaded = torch.nn.Linear(2, 2)  # real (non-meta) params = "loaded from checkpoint"
    pretrained_cls._init_weights(pretrained_cls.__new__(pretrained_cls), loaded)
    assert not getattr(loaded, "initialized", False)

    # Idempotent: a second run must not double-wrap or crash.
    adapter.prepare_loading("fake/model", {})
    assert causal_cls._tied_weights_keys == planted_module._expected_tied_mapping
