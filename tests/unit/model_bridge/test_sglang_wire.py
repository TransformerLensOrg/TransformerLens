"""Unit tests for the overlay → ``ServerArgs.forward_hooks`` spec translation."""
from __future__ import annotations

import json

from transformer_lens.model_bridge.sources.sglang import wire


class TestFreshChannel:
    def test_returns_unique_ipc_addresses(self):
        a = wire.fresh_channel()
        b = wire.fresh_channel()
        assert a != b
        assert a.startswith("ipc:///tmp/tl_sglang_") and a.endswith(".sock")


class TestBuildForwardHooks:
    """Each capture-spec entry → one forward-hooks dict with the right factory path."""

    def _specs(self):
        # Mimics DecoderOnlyOverlay.capture_specs(hf_config) shape.
        return {
            "embed.hook_out": ("model.embed_tokens", 16),
            "blocks.0.hook_out": ("model.layers.0", 16),
            "blocks.0.attn.hook_out": ("model.layers.0.self_attn", 16),
            "blocks.0.mlp.hook_out": ("model.layers.0.mlp", 16),
        }

    def test_one_entry_per_spec(self):
        hooks = wire.build_forward_hooks(self._specs(), "ipc:///tmp/ch.sock")
        assert len(hooks) == 4

    def test_factory_path_points_at_make_capture_hook(self):
        hooks = wire.build_forward_hooks(self._specs(), "ipc:///tmp/ch.sock")
        for spec in hooks:
            assert spec["hook_factory"] == (
                "transformer_lens.model_bridge.sources.sglang.hooks:make_capture_hook"
            )

    def test_target_module_is_the_dot_path(self):
        hooks = wire.build_forward_hooks(self._specs(), "ipc:///tmp/ch.sock")
        by_name = {h["config"]["canonical_name"]: h for h in hooks}
        assert by_name["blocks.0.attn.hook_out"]["target_modules"] == ["model.layers.0.self_attn"]

    def test_decoder_layer_path_flagged_materialize(self):
        hooks = wire.build_forward_hooks(self._specs(), "ipc:///tmp/ch.sock")
        by_name = {h["config"]["canonical_name"]: h for h in hooks}
        # Only ``model.layers.{i}`` exactly should be materialize=True.
        assert by_name["blocks.0.hook_out"]["config"]["materialize"] is True
        assert by_name["embed.hook_out"]["config"]["materialize"] is False
        assert by_name["blocks.0.attn.hook_out"]["config"]["materialize"] is False
        assert by_name["blocks.0.mlp.hook_out"]["config"]["materialize"] is False

    def test_channel_propagates_to_every_config(self):
        ch = "ipc:///tmp/specific.sock"
        hooks = wire.build_forward_hooks(self._specs(), ch)
        for spec in hooks:
            assert spec["config"]["channel"] == ch

    def test_entries_are_json_serializable(self):
        # SGLang pickles ServerArgs across spawn; JSON-roundtrippable is a strong
        # proxy for "pure primitives, no callables sneaking in".
        hooks = wire.build_forward_hooks(self._specs(), "ipc:///tmp/ch.sock")
        rendered = json.dumps(hooks)
        assert "make_capture_hook" in rendered
