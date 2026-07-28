"""Non-persistent buffer restoration after meta-device loading.

transformers 5.x replaces every non-persistent buffer with uninitialized memory
(`_move_missing_keys_from_meta_to_device`), then restores rotary tables only for
modules exposing ``original_inv_freq``. Remote code predating that attribute
silently rotates positions by random values, so adapters restore it themselves.
"""

import sys
from types import ModuleType, SimpleNamespace

import torch


class TestInternLM2RotaryRestoration:
    def _fake_model(self, garbage: torch.Tensor):
        class _Rotary(torch.nn.Module):
            # Name must contain "RotaryEmbedding" — that is how the hook finds it.
            pass

        _Rotary.__name__ = "InternLM2RotaryEmbedding"
        rotary = _Rotary()
        rotary.inv_freq = garbage
        rotary.dim = 8
        rotary.base = 10000

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = rotary

        model = _Model()
        model.config = SimpleNamespace(pretraining_tp=1)
        return model, rotary

    def _adapter(self):
        from transformer_lens.config import TransformerBridgeConfig
        from transformer_lens.model_bridge.supported_architectures.internlm2 import (
            InternLM2ArchitectureAdapter,
        )

        cfg = TransformerBridgeConfig(
            d_model=32,
            d_head=8,
            n_layers=1,
            n_ctx=64,
            n_heads=4,
            d_vocab=64,
            architecture="InternLM2ForCausalLM",
        )
        return InternLM2ArchitectureAdapter(cfg)

    def test_garbage_inv_freq_is_recomputed(self) -> None:
        """A NaN table is what meta materialization actually leaves behind."""
        garbage = torch.full((4,), float("nan"))
        model, rotary = self._fake_model(garbage)
        self._adapter().prepare_model(model)

        expected = 1.0 / (10000 ** (torch.arange(0, 8, 2, dtype=torch.int64).float() / 8))
        assert torch.isfinite(rotary.inv_freq).all()
        assert torch.allclose(rotary.inv_freq, expected)

    def test_modules_without_rotary_metadata_are_skipped(self) -> None:
        """Never guess: a module missing dim/base is left untouched."""
        model, rotary = self._fake_model(torch.full((4,), float("nan")))
        del rotary.dim
        self._adapter().prepare_model(model)
        assert torch.isnan(rotary.inv_freq).all()


class TestBaichuanDoesNotPatchBaseClass:
    def test_transformers_base_class_is_never_patched(self) -> None:
        """Baichuan remote code does `from transformers import PreTrainedModel`.

        Patching that name would disable _init_weights — including HF's rotary
        restoration — for every model loaded later in the same process.
        """
        from transformers import PreTrainedModel

        from transformer_lens.model_bridge.supported_architectures.baichuan import (
            _patch_init_weights_for_baichuan,
        )

        fake = ModuleType("transformers_modules.fake.modeling_baichuan")
        fake.PreTrainedModel = PreTrainedModel  # type: ignore[attr-defined]
        sys.modules["transformers_modules.fake.modeling_baichuan"] = fake
        before = PreTrainedModel.__dict__["_init_weights"]
        try:
            _patch_init_weights_for_baichuan()
            after = PreTrainedModel.__dict__["_init_weights"]
            assert after is before, "transformers base _init_weights was replaced"
            assert not getattr(PreTrainedModel, "_tl_patched", False)
        finally:
            del sys.modules["transformers_modules.fake.modeling_baichuan"]


class TestSharedRotaryRestore:
    """One helper serves adapters and the benchmark reference so they cannot drift."""

    def _module(self, inv_freq, dim=8, base=10000):
        class _Rotary(torch.nn.Module):
            pass

        _Rotary.__name__ = "FakeRotaryEmbedding"
        mod = _Rotary()
        mod.inv_freq = inv_freq
        mod.dim = dim
        mod.base = base

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = mod

        return _Model(), mod

    def test_zero_table_is_repaired(self) -> None:
        """internlm2's real failure mode: all zeros, which makes RoPE identity."""
        from transformer_lens.model_bridge.buffer_restore import restore_rotary_inv_freq

        model, mod = self._module(torch.zeros(4))
        assert restore_rotary_inv_freq(model) == 1
        expected = 1.0 / (10000 ** (torch.arange(0, 8, 2, dtype=torch.int64).float() / 8))
        assert torch.allclose(mod.inv_freq, expected)

    def test_nan_table_is_repaired(self) -> None:
        from transformer_lens.model_bridge.buffer_restore import restore_rotary_inv_freq

        model, mod = self._module(torch.full((4,), float("nan")))
        assert restore_rotary_inv_freq(model) == 1
        assert torch.isfinite(mod.inv_freq).all()

    def test_valid_table_is_left_alone(self) -> None:
        """A correct table must never be overwritten."""
        from transformer_lens.model_bridge.buffer_restore import restore_rotary_inv_freq

        good = 1.0 / (10000 ** (torch.arange(0, 8, 2, dtype=torch.int64).float() / 8))
        model, mod = self._module(good.clone())
        assert restore_rotary_inv_freq(model) == 0
        assert torch.allclose(mod.inv_freq, good)

    def test_scaled_table_is_left_alone(self) -> None:
        """Linear/NTK scaling keeps the decreasing-in-(0,1] shape; do not clobber."""
        from transformer_lens.model_bridge.buffer_restore import restore_rotary_inv_freq

        scaled = 1.0 / (500000 ** (torch.arange(0, 8, 2, dtype=torch.int64).float() / 8))
        model, mod = self._module(scaled.clone(), base=10000)
        assert restore_rotary_inv_freq(model) == 0
        assert torch.allclose(mod.inv_freq, scaled)


class TestLegacyCosSinCacheRebuild:
    """Legacy rotary indexes cos_cached directly, so it must be rebuilt, not cleared."""

    def _legacy_module(self):
        class _Rotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 8
                self.base = 10000
                self.max_seq_len_cached = 16
                self.inv_freq = torch.zeros(4)  # the meta-load failure mode
                self.cos_cached = torch.ones(16, 8)
                self.sin_cached = torch.zeros(16, 8)
                self.rebuilt_with = None

            def _set_cos_sin_cache(self, seq_len, device, dtype):
                self.rebuilt_with = seq_len
                freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), self.inv_freq)
                self.cos_cached = freqs.cos().to(dtype)
                self.sin_cached = freqs.sin().to(dtype)

            def forward(self, x, seq_len):
                # Mirrors the real legacy implementation's direct indexing.
                if seq_len > self.max_seq_len_cached:
                    self._set_cos_sin_cache(seq_len, x.device, torch.float32)
                return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

        _Rotary.__name__ = "LegacyRotaryEmbedding"
        rotary = _Rotary()

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = rotary

        return _Model(), rotary

    def test_cache_is_rebuilt_not_cleared(self) -> None:
        """Setting the cache to None would raise on the next forward."""
        from transformer_lens.model_bridge.buffer_restore import restore_rotary_inv_freq

        model, rotary = self._legacy_module()
        assert restore_rotary_inv_freq(model) == 1
        assert rotary.cos_cached is not None and rotary.sin_cached is not None
        # The decisive check: a short forward must not raise.
        cos, _ = rotary(torch.zeros(1), seq_len=4)
        assert cos.shape[0] == 4

    def test_rebuild_uses_the_repaired_table(self) -> None:
        from transformer_lens.model_bridge.buffer_restore import restore_rotary_inv_freq

        model, rotary = self._legacy_module()
        restore_rotary_inv_freq(model)
        expected = 1.0 / (10000 ** (torch.arange(0, 8, 2, dtype=torch.int64).float() / 8))
        assert torch.allclose(rotary.inv_freq, expected)
        # cos of nonzero freqs is no longer the all-ones placeholder.
        assert not torch.allclose(rotary.cos_cached, torch.ones_like(rotary.cos_cached))


class TestInternLM2PositionEmbeddingInjection:
    """internlm2 keeps rotary per attention module, so its decoder layer passes
    no position_embeddings — the bridge must supply them or RoPE is skipped."""

    def _bridge(self):
        from transformer_lens.config import TransformerBridgeConfig
        from transformer_lens.model_bridge.supported_architectures.internlm2 import (
            _InternLM2AttentionBridge,
        )

        cfg = TransformerBridgeConfig(
            d_model=32,
            d_head=8,
            n_layers=1,
            n_ctx=64,
            n_heads=4,
            d_vocab=64,
            architecture="InternLM2ForCausalLM",
        )
        bridge = _InternLM2AttentionBridge(name="attention", config=cfg)

        class _Rotary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, x, position_ids):
                self.calls.append(position_ids)
                seq = x.shape[1]
                return (torch.ones(1, seq, 8), torch.zeros(1, seq, 8))

        class _HFAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = _Rotary()

        hf_attn = _HFAttn()
        object.__setattr__(bridge, "_original_component_ref", hf_attn)
        return bridge, hf_attn

    def test_injects_from_the_layers_own_rotary(self) -> None:
        """Without this the reconstruction skips RoPE entirely and silently."""
        bridge, hf_attn = self._bridge()
        captured = {}

        def fake_super_forward(*args, **kwargs):
            captured.update(kwargs)
            return "ok"

        # Exercise only the injection logic, not the full reconstruction.
        import transformer_lens.model_bridge.supported_architectures.internlm2 as mod

        original = mod.JointQKVPositionEmbeddingsAttentionBridge.forward
        mod.JointQKVPositionEmbeddingsAttentionBridge.forward = fake_super_forward
        try:
            type(bridge).original_component = property(lambda self: hf_attn)  # type: ignore[assignment]
            bridge(hidden_states=torch.zeros(1, 5, 32))
        finally:
            mod.JointQKVPositionEmbeddingsAttentionBridge.forward = original
            del type(bridge).original_component

        pos = captured.get("position_embeddings")
        assert pos is not None, "position_embeddings were not injected"
        assert len(hf_attn.rotary_emb.calls) == 1
        assert hf_attn.rotary_emb.calls[0].shape == (1, 5), "position_ids must span the sequence"
