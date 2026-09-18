"""Unit tests for the shared remote-code transformers-v5 compat helpers.

These patches only fire on real ``trust_remote_code`` loads, which CI never
performs, so the helpers are exercised directly with stand-in classes and tiny
``nn.Module``s that mirror how the patches detect loaded-ness (first parameter
on a non-meta device).
"""

import sys
from types import ModuleType
from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.supported_architectures._remote_code_compat import (
    compute_default_rope_inv_freq,
    disable_tied_weights_lookup,
    force_import_remote_class,
    iter_remote_modeling_modules,
    patch_init_weights_skip_loaded,
    retie_weights_keys_v5,
)


def _make_remote_class() -> tuple[type, list[Any]]:
    """A stand-in remote PreTrainedModel subclass whose _init_weights both
    records the module and clobbers its weight (the v5 re-randomisation bug)."""
    calls: list[Any] = []

    class FakeRemotePreTrained:
        def _init_weights(self, module: Any) -> None:
            calls.append(module)
            weight = getattr(module, "weight", None)
            if weight is not None:
                weight.data.fill_(0.0)

    return FakeRemotePreTrained, calls


class TestPatchInitWeightsSkipLoaded:
    def test_refuses_transformers_base_class(self) -> None:
        """Patching the HF base would disable init for every later model load."""
        from transformers import PreTrainedModel

        before = PreTrainedModel.__dict__["_init_weights"]
        with pytest.raises(ValueError, match="PreTrainedModel itself"):
            patch_init_weights_skip_loaded(PreTrainedModel)
        assert PreTrainedModel.__dict__["_init_weights"] is before
        assert not getattr(PreTrainedModel, "_tl_patched", False)

    def test_refuses_class_without_own_init_weights(self) -> None:
        """Wrapping an inherited implementation is never what the patch means."""

        class Parent:
            def _init_weights(self, module: Any) -> None:
                pass

        class Child(Parent):
            pass

        with pytest.raises(ValueError, match="defines no _init_weights of its own"):
            patch_init_weights_skip_loaded(Child)
        assert "_init_weights" not in Child.__dict__
        assert not getattr(Child, "_tl_patched", False)

    def test_skips_materialized_modules(self) -> None:
        """A module with real (non-meta) params was loaded from the checkpoint:
        the original init must never run on it."""
        cls, calls = _make_remote_class()
        patch_init_weights_skip_loaded(cls)

        loaded = nn.Linear(2, 2)
        checkpoint_values = loaded.weight.detach().clone()
        cls._init_weights(None, loaded)

        assert calls == []
        assert torch.equal(loaded.weight, checkpoint_values)

    def test_delegates_for_meta_modules(self) -> None:
        """Modules still on meta device are pre-materialisation: init proceeds."""
        cls, calls = _make_remote_class()
        patch_init_weights_skip_loaded(cls)

        meta = nn.Linear(2, 2, device="meta")
        cls._init_weights(None, meta)
        assert calls == [meta]

    def test_delegates_for_parameterless_modules(self) -> None:
        """No params means loaded-ness cannot be proven; the patch stays out
        of the way and lets the original decide."""
        cls, calls = _make_remote_class()
        patch_init_weights_skip_loaded(cls)

        container = nn.Module()
        cls._init_weights(None, container)
        assert calls == [container]

    def test_double_patch_wraps_once(self) -> None:
        cls, calls = _make_remote_class()
        patch_init_weights_skip_loaded(cls)
        wrapper = cls.__dict__["_init_weights"]

        patch_init_weights_skip_loaded(cls)
        assert cls.__dict__["_init_weights"] is wrapper
        assert cls._tl_patched is True

        # One delegation per call, not one per patch application.
        meta = nn.Linear(2, 2, device="meta")
        cls._init_weights(None, meta)
        assert calls == [meta]

    def test_real_subclass_is_patched_without_touching_base(self) -> None:
        """The intended target: a remote-code subclass of the HF base."""
        from transformers import PreTrainedModel

        calls: list[Any] = []

        class _RemoteSub(PreTrainedModel):
            def _init_weights(self, module: Any) -> None:
                calls.append(module)

        base_before = PreTrainedModel.__dict__["_init_weights"]
        patch_init_weights_skip_loaded(_RemoteSub)

        assert PreTrainedModel.__dict__["_init_weights"] is base_before
        assert not getattr(PreTrainedModel, "_tl_patched", False)

        meta = nn.Linear(2, 2, device="meta")
        _RemoteSub._init_weights(None, meta)
        assert calls == [meta]


class TestRetieWeightsKeysV5:
    def test_list_is_rewritten_to_mapping(self) -> None:
        class Model:
            _tied_weights_keys: Any = ["lm_head.weight"]

        retie_weights_keys_v5(Model, {"lm_head.weight": "model.embed.weight"})
        assert Model._tied_weights_keys == {"lm_head.weight": "model.embed.weight"}

    def test_dict_is_left_alone(self) -> None:
        existing = {"lm_head.weight": "already.v5.weight"}

        class Model:
            _tied_weights_keys: Any = existing

        retie_weights_keys_v5(Model, {"lm_head.weight": "other.weight"})
        assert Model._tied_weights_keys is existing

    def test_none_cls_and_absent_attribute_are_noops(self) -> None:
        retie_weights_keys_v5(None, {"a": "b"})

        class Model:
            pass

        retie_weights_keys_v5(Model, {"a": "b"})
        assert "_tied_weights_keys" not in Model.__dict__


class TestDisableTiedWeightsLookup:
    def test_sets_empty_mapping(self) -> None:
        class Model:
            pass

        disable_tied_weights_lookup(Model)
        assert Model.all_tied_weights_keys == {}  # type: ignore[attr-defined]


class TestIterRemoteModelingModules:
    def test_yields_only_modeling_modules_matching_a_fragment(self) -> None:
        names = {
            "match": "transformers_modules.acme.zorbo_modeling_minimal",
            "match_case": "transformers_modules.acme2.Modeling_ZORBO",
            "not_modeling": "transformers_modules.acme.configuration_zorbo",
            "other_arch": "transformers_modules.acme.modeling_florble",
        }
        for name in names.values():
            sys.modules[name] = ModuleType(name)
        try:
            found = {m.__name__ for m in iter_remote_modeling_modules("zorbo")}
            assert names["match"] in found
            assert names["match_case"] in found
            assert names["not_modeling"] not in found
            assert names["other_arch"] not in found

            both = {m.__name__ for m in iter_remote_modeling_modules("zorbo", "florble")}
            assert names["other_arch"] in both
            assert names["match"] in both
        finally:
            for name in names.values():
                del sys.modules[name]


class TestForceImportRemoteClass:
    def test_returns_class_and_forwards_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import transformers.dynamic_module_utils as dmu

        class Sentinel:
            pass

        seen: dict[str, Any] = {}

        def fake(ref: str, name: str, **kwargs: Any) -> type:
            seen["args"] = (ref, name, kwargs)
            return Sentinel

        monkeypatch.setattr(dmu, "get_class_from_dynamic_module", fake)
        result = force_import_remote_class("org/model", "modeling_x.XForCausalLM", revision="abc")
        assert result is Sentinel
        assert seen["args"] == ("modeling_x.XForCausalLM", "org/model", {"revision": "abc"})

    def test_returns_none_when_dynamic_module_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import transformers.dynamic_module_utils as dmu

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("offline / not a remote-code repo")

        monkeypatch.setattr(dmu, "get_class_from_dynamic_module", boom)
        assert force_import_remote_class("org/model", "modeling_x.XForCausalLM") is None


class TestComputeDefaultRopeInvFreq:
    def test_config_path_matches_v4_reference_formula(self) -> None:
        class Cfg:
            rope_theta = 10000.0
            hidden_size = 64
            num_attention_heads = 4

        inv_freq, scaling = compute_default_rope_inv_freq(Cfg(), device="cpu")
        dim = 16
        expected = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        torch.testing.assert_close(inv_freq, expected)
        assert scaling == 1.0

    def test_partial_rotary_factor_none_treated_as_full(self) -> None:
        """Some configs carry an explicit None; ouro's `or 1.0` guard must hold."""

        class Cfg:
            rope_theta = 10000.0
            hidden_size = 64
            num_attention_heads = 4
            partial_rotary_factor = None

        inv_freq, _ = compute_default_rope_inv_freq(Cfg())
        assert inv_freq.shape == (8,)

    def test_kwargs_only_path(self) -> None:
        """v4 also allowed configless calls with base/dim; dream registers the
        helper globally, so arbitrary remote code may use that form."""
        inv_freq, scaling = compute_default_rope_inv_freq(base=10000.0, dim=8)
        expected = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.int64).float() / 8))
        torch.testing.assert_close(inv_freq, expected)
        assert scaling == 1.0
