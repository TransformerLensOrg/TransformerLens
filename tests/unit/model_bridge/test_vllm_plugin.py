"""Unit tests for the eager batched capture hook (plugin._make_batched_hook).

The hook closure is pure torch + the segmentation helper, so it runs without a
vLLM install or GPU: a mock worker carries ``query_start_loc`` (segment_by_request's
backend-agnostic source) and the per-(req_id, hook) accumulator.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.sources.vllm import plugin
from transformer_lens.model_bridge.sources.vllm.plugin import _make_batched_hook
from transformer_lens.model_bridge.sources.vllm.worker_extension import resolve_dot_path


def _worker(specs):
    """Single-request worker mock: query_start_loc=[0, 2], one req id."""
    return SimpleNamespace(
        _tl_accum={},
        _tl_intervention_specs=specs,
        model_runner=SimpleNamespace(
            query_start_loc=torch.tensor([0, 2]),
            input_batch=SimpleNamespace(req_ids=["r0"]),
        ),
    )


class TestBatchedHookInterventionTargeting:
    """A batched hook applies ONLY the spec keyed to its own hook name."""

    def test_non_targeted_hook_is_unmodified(self):
        # Spec targets embed; a hook belonging to blocks.0.hook_out must NOT apply it.
        worker = _worker({"embed.hook_out": {"op": "suppress"}})
        counter = torch.zeros(1, dtype=torch.int64)
        hook = _make_batched_hook(worker, "blocks.0.hook_out", counter)

        out = hook(None, None, torch.ones(2, 4))
        assert torch.equal(out, torch.ones(2, 4)), "spec leaked onto a non-targeted hook"
        assert torch.equal(worker._tl_accum[("r0", "blocks.0.hook_out")][0], torch.ones(2, 4))

    def test_targeted_hook_is_modified(self):
        worker = _worker({"embed.hook_out": {"op": "suppress"}})
        counter = torch.zeros(1, dtype=torch.int64)
        hook = _make_batched_hook(worker, "embed.hook_out", counter)

        out = hook(None, None, torch.ones(2, 4))
        assert torch.equal(out, torch.zeros(2, 4)), "suppress did not apply to its target"
        assert torch.equal(worker._tl_accum[("r0", "embed.hook_out")][0], torch.zeros(2, 4))

    def test_no_spec_leaves_output_unchanged(self):
        worker = _worker({})
        counter = torch.zeros(1, dtype=torch.int64)
        hook = _make_batched_hook(worker, "embed.hook_out", counter)

        out = hook(None, None, torch.ones(2, 4) * 3)
        assert torch.equal(out, torch.ones(2, 4) * 3)


class TestSpecChannel:
    """The driver→worker spec channel must survive process boundaries (env var)."""

    def _sample_config(self):
        return {
            "capture_specs": {
                "embed.hook_out": ("model.embed_tokens", 4),
                "blocks.0.hook_out": ("model.layers.0", 4),
            },
            "max_num_batched_tokens": 2048,
            "dtype": torch.bfloat16,
            "enable_batching": False,
            "enable_position_interventions": True,
        }

    def test_serialize_round_trip(self):
        config = self._sample_config()
        restored = plugin._deserialize_config(plugin._serialize_config(config))
        assert restored == config
        assert restored["dtype"] is torch.bfloat16

    def test_deserialize_rejects_non_dtype(self):
        raw = plugin._serialize_config(self._sample_config())
        data = json.loads(raw)
        data["dtype"] = "nn"  # torch.nn exists but is not a dtype
        with pytest.raises(ValueError, match="not a torch dtype"):
            plugin._deserialize_config(json.dumps(data))

    def test_configure_sets_env_and_clear_config_removes_it(self):
        config = self._sample_config()
        try:
            plugin.configure(**config)
            assert plugin._ENV_CONFIG_KEY in os.environ
            assert plugin._active_config()["capture_specs"] == config["capture_specs"]
        finally:
            plugin.clear_config()
        assert plugin._ENV_CONFIG_KEY not in os.environ

    def test_active_config_reaches_spawned_workers(self, monkeypatch):
        """A spawned worker re-imports the module fresh but inherits the env var."""
        config = self._sample_config()
        monkeypatch.setenv(plugin._ENV_CONFIG_KEY, plugin._serialize_config(config))
        assert plugin._active_config() == config

    def test_active_config_none_when_no_channel(self, monkeypatch):
        monkeypatch.delenv(plugin._ENV_CONFIG_KEY, raising=False)
        assert plugin._active_config() is None


class TestResolveDotPath:
    """Per-rank module resolution: missing segments mean 'not on this rank', not a crash."""

    def _model(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(2, 2)])
        model.norm = nn.LayerNorm(2)
        return SimpleNamespace(model=model)

    def test_resolves_nested_and_indexed(self):
        root = self._model()
        assert resolve_dot_path(root, "model.norm") is root.model.norm
        assert resolve_dot_path(root, "model.layers.0") is root.model.layers[0]

    def test_missing_attribute_returns_none(self):
        assert resolve_dot_path(self._model(), "model.mlp") is None

    def test_out_of_range_index_returns_none(self):
        """A PP rank owning layers [0..k) legally lacks the later indices."""
        assert resolve_dot_path(self._model(), "model.layers.7") is None

    def test_pp_missing_layer_stub_treated_as_absent(self):
        """vLLM PP fills non-owned slots (layers, embed_tokens, norm) with
        PPMissingLayer identity stubs the forward never calls — resolving one would
        install a hook that serves its dead zero buffer as a real capture."""

        class PPMissingLayer(nn.Module):  # matched by name, as vLLM's real class is
            pass

        root = self._model()
        root.model.layers.append(PPMissingLayer())
        root.model.norm = PPMissingLayer()
        assert resolve_dot_path(root, "model.layers.0") is root.model.layers[0]
        assert resolve_dot_path(root, "model.layers.1") is None
        assert resolve_dot_path(root, "model.norm") is None


class TestCompiledHookInstrumentationSafety:
    """Compile-traced hook internals must stay annotation-free: pytest's jaxtyping
    hook wraps annotated functions, and dynamo tracing the wrapper corrupts
    jaxtyping's thread-local memo stack process-wide (GPU-verified)."""

    def test_gated_capture_has_no_annotations(self):
        from transformer_lens.model_bridge.sources.vllm.plugin import _gated_capture

        assert (
            _gated_capture.__wrapped__.__annotations__ == {}
            if hasattr(_gated_capture, "__wrapped__")
            else _gated_capture.__annotations__ == {}
        )


class TestVllmPluginEntryPoint:
    def test_general_plugin_entry_point_declared(self):
        """Spawned TP workers only get the load_model patch via vllm.general_plugins —
        without this entry point, workers boot hookless and captures are empty."""
        from importlib.metadata import entry_points

        eps = entry_points(group="vllm.general_plugins")
        assert any(
            ep.value == "transformer_lens.model_bridge.sources.vllm.plugin:register" for ep in eps
        )
