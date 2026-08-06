"""Multi-GPU support tests for TransformerBridge — the CPU/mocked tier.

Exercises the resolver / param-plumbing / .to() guard / validation logic without
hardware. Real >= 2-GPU coverage lives in test_bridge_multigpu.py and
test_bridge_multigpu_device_map.py (`-m multigpu`).
"""

from typing import Dict, Union

import pytest
import torch
from accelerate.hooks import attach_align_device_hook

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)
from transformer_lens.utilities.multi_gpu import (
    cast_floating_params_to_dtype,
    count_unique_devices,
    find_embedding_device,
    find_misplaced_modules,
    resolve_device_map,
)

# ---------- CPU-runnable tests ----------


class TestResolveDeviceMap:
    def test_no_multi_device_returns_none(self):
        dm, mm = resolve_device_map(None, None, None)
        assert dm is None and mm is None
        dm, mm = resolve_device_map(1, None, None)
        assert dm is None and mm is None
        dm, mm = resolve_device_map(0, None, None)
        assert dm is None and mm is None

    def test_explicit_device_map_string_passes_through(self):
        dm, mm = resolve_device_map(None, "auto", None)
        assert dm == "auto"
        assert mm is None

    def test_explicit_device_map_dict_passes_through(self):
        explicit: Dict[str, Union[str, int]] = {"transformer.h.0": 0}
        dm, mm = resolve_device_map(None, explicit, None)
        assert dm is explicit
        assert mm is None

    def test_user_max_memory_passes_through(self):
        user_mm: Dict[Union[str, int], Union[str, int]] = {0: "20GiB"}
        dm, mm = resolve_device_map(None, "auto", None, max_memory=user_mm)
        assert dm == "auto"
        assert mm is user_mm

    def test_device_and_device_map_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_device_map(None, "auto", "cuda")

    def test_n_devices_without_cuda_raises(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA available; this test targets the no-CUDA path.")
        with pytest.raises(ValueError, match="requires CUDA"):
            resolve_device_map(2, None, None)

    def test_n_devices_exceeds_visible_raises(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required.")
        too_many = torch.cuda.device_count() + 1
        with pytest.raises(ValueError, match="only"):
            resolve_device_map(too_many, None, None)

    def test_n_devices_returns_balanced_string_and_max_memory(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Requires 2+ CUDA devices.")
        dm, mm = resolve_device_map(2, None, None)
        # device_map must be a string directive (HF device_map dicts are keyed by
        # submodule path — int keys would fail to match any submodule).
        assert dm == "balanced"
        assert isinstance(mm, dict)
        assert set(mm.keys()) == {0, 1}

    def test_n_devices_uses_concrete_cuda_memory_caps(self, monkeypatch):
        free_memory = {0: 11_000_000_000, 1: 22_000_000_000, 2: 33_000_000_000}

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
        monkeypatch.setattr(
            torch.cuda,
            "mem_get_info",
            lambda index: (free_memory[index], free_memory[index] + 1_000_000_000),
        )

        dm, mm = resolve_device_map(2, None, None)

        assert dm == "balanced"
        assert mm == {0: free_memory[0], 1: free_memory[1]}
        assert "auto" not in mm.values()

    def test_n_devices_respects_user_max_memory(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Requires 2+ CUDA devices.")
        user_mm: Dict[Union[str, int], Union[str, int]] = {0: "10GiB", 1: "10GiB"}
        dm, mm = resolve_device_map(2, None, None, max_memory=user_mm)
        assert dm == "balanced"
        assert mm == user_mm

    def test_cpu_value_in_device_map_passes_through(self):
        explicit: Dict[str, Union[str, int]] = {"transformer.h.0": "cpu"}
        dm, mm = resolve_device_map(None, explicit, None)
        assert dm is explicit
        assert mm is None

    def test_disk_value_in_device_map_passes_through(self):
        explicit: Dict[str, Union[str, int]] = {"transformer.h.0": "disk"}
        dm, mm = resolve_device_map(None, explicit, None)
        assert dm is explicit
        assert mm is None

    def test_meta_value_in_device_map_passes_through(self):
        device_map: Dict[str, Union[str, int]] = {"transformer.h.0": "meta"}
        dm, mm = resolve_device_map(None, device_map, None)
        assert dm is device_map
        assert mm is None


class TestMixedCpuGpuMapRejection:
    """Mixed CPU+GPU maps mean accelerate CPU offload — meta placeholders that
    param-reading Bridge components never materialize. Rejected at the resolver
    (explicit dicts) so no weights are downloaded before the error."""

    def test_explicit_mixed_dict_rejected(self):
        with pytest.raises(ValueError, match="offload"):
            resolve_device_map(None, {"transformer.wte": 0, "transformer.ln_f": "cpu"}, None)

    def test_all_cpu_dict_passes(self):
        dm, _ = resolve_device_map(None, {"transformer.wte": "cpu", "lm_head": "cpu"}, None)
        assert dm == {"transformer.wte": "cpu", "lm_head": "cpu"}

    def test_all_gpu_dict_passes(self):
        dm, _ = resolve_device_map(None, {"transformer.wte": 0, "lm_head": "cuda:1"}, None)
        assert dm == {"transformer.wte": 0, "lm_head": "cuda:1"}


class TestFindMisplacedModules:
    """Realized-placement check: a map entry whose loaded params sit elsewhere is a
    split tie group (accelerate places a tied parameter once) — boot must fail loud
    instead of crashing mid-forward on a cross-device kernel."""

    def _model(self, device_map, param_device="cpu"):
        import torch.nn as nn

        model = nn.Module()
        model.emb = nn.Linear(2, 2)
        model.head = nn.Linear(2, 2)
        if param_device == "meta":
            model.head = model.head.to_empty(device="meta")
        model.hf_device_map = device_map
        return model

    def test_consistent_map_passes(self):
        model = self._model({"emb": "cpu", "head": "cpu"})
        assert find_misplaced_modules(model) == []

    def test_violated_entry_flagged(self):
        # Params physically on cpu but mapped to cuda:0 — the tied-split signature.
        model = self._model({"emb": 0, "head": "cpu"})
        assert find_misplaced_modules(model) == [("emb", "0", "cpu")]

    def test_meta_params_skipped_as_offload(self):
        # Offloaded modules hold meta placeholders; accelerate manages them.
        model = self._model({"emb": "cpu", "head": "cpu"}, param_device="meta")
        assert find_misplaced_modules(model) == []

    def test_no_map_returns_empty(self):
        import torch.nn as nn

        assert find_misplaced_modules(nn.Linear(2, 2)) == []


class TestFindEmbeddingDevice:
    def test_returns_none_for_no_device_map(self):
        class Stub:
            pass

        assert find_embedding_device(Stub()) is None

    def test_uses_get_input_embeddings_when_available(self):
        # A stub model with both hf_device_map AND get_input_embeddings should
        # consult the embedding module, not the first dict entry. This is the key
        # difference from the insertion-order heuristic — covers the multimodal /
        # encoder-decoder case where the first map entry is the vision tower.
        embed = torch.nn.Embedding(10, 4)
        embed = embed.to("cpu")

        class Stub:
            hf_device_map = {"vision_tower.stuff": 1, "language_model.embed_tokens": "cpu"}

            def get_input_embeddings(self):
                return embed

        result = find_embedding_device(Stub())
        assert result is not None
        assert result.type == "cpu"

    def test_falls_back_to_first_entry_when_get_input_embeddings_unavailable(self):
        class Stub:
            hf_device_map = {"embed_tokens": "cpu", "layers.0": "cpu"}

        assert find_embedding_device(Stub()) == torch.device("cpu")

    def test_handles_int_device_ids_in_fallback(self):
        class Stub:
            hf_device_map = {"embed_tokens": 0, "layers.0": 1}

        result = find_embedding_device(Stub())
        assert result is not None
        assert result.type == "cuda"
        assert result.index == 0

    def test_handles_get_input_embeddings_returning_none(self):
        class Stub:
            hf_device_map = {"embed_tokens": "cpu"}

            def get_input_embeddings(self):
                return None

        assert find_embedding_device(Stub()) == torch.device("cpu")


class TestCountUniqueDevices:
    def test_no_map_returns_1(self):
        class Stub:
            pass

        assert count_unique_devices(Stub()) == 1

    def test_counts_unique_values(self):
        class Stub:
            hf_device_map = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 2}

        assert count_unique_devices(Stub()) == 3


class TestCastFloatingParamsToDtype:
    def test_casts_cpu_and_disk_offloaded_params(self, tmp_path):
        from accelerate.big_modeling import dispatch_model
        from accelerate.utils import align_module_device

        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(3, 3, dtype=torch.float32)
                self.second = torch.nn.Linear(3, 2, dtype=torch.float32)

        model = StubModel()
        dispatch_model(
            model,
            device_map={"first": "cpu", "second": "disk"},
            offload_dir=str(tmp_path),
        )

        assert model.first.weight.device.type == "cpu"
        assert model.second.weight.device.type == "meta"

        cast_floating_params_to_dtype(model, torch.float16)

        assert model.first.weight.dtype == torch.float16
        assert model.second.weight.device.type == "meta"
        assert model.second.weight.dtype == torch.float16
        with align_module_device(model.second):
            assert model.second.weight.device.type == "cpu"
            assert model.second.weight.dtype == torch.float16

    def test_skips_plain_meta_params(self):
        module = torch.nn.Linear(3, 2, device="meta", dtype=torch.float32)

        cast_floating_params_to_dtype(module, torch.float16)

        assert module.weight.device.type == "meta"
        assert module.weight.dtype == torch.float32


class TestBootParamValidation:
    def test_device_and_device_map_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            TransformerBridge.boot_transformers("gpt2", device="cpu", device_map="auto")

    def test_explicit_cpu_device_map_boots_and_runs(self):
        bridge = TransformerBridge.boot_transformers("gpt2", device_map={"": "cpu"})

        assert bridge.cfg.device == "cpu"
        assert bridge.cfg.n_devices == 1
        tokens = torch.tensor([[bridge.tokenizer.eos_token_id]])
        with torch.no_grad():
            logits = bridge(tokens)
        assert logits.shape == (1, 1, bridge.cfg.d_vocab_out)
        assert logits.device.type == "cpu"

    def test_preloaded_with_device_map_rejected(self, gpt2_bridge):
        # Passing both hf_model= and device_map/n_devices is ambiguous — the device_map
        # would be silently ignored. We raise so the caller isn't surprised.
        with pytest.raises(ValueError, match="only supported when the bridge loads"):
            TransformerBridge.boot_transformers(
                "gpt2", hf_model=gpt2_bridge.original_model, device_map="auto"
            )

    def test_preloaded_with_n_devices_rejected(self, gpt2_bridge):
        with pytest.raises(ValueError, match="only supported when the bridge loads"):
            TransformerBridge.boot_transformers(
                "gpt2", hf_model=gpt2_bridge.original_model, n_devices=2
            )

    def test_meta_device_map_rejected_at_boot(self):
        with pytest.raises(ValueError, match="meta target"):
            TransformerBridge.boot_transformers("gpt2", device_map={"": "meta"})

    def test_meta_device_map_allowed_with_load_weights_false(self):
        bridge = TransformerBridge.boot_transformers(
            "gpt2", device_map={"": "meta"}, load_weights=False
        )
        assert bridge is not None
        assert bridge.cfg.d_model == 768


class TestToMethodGuardsMultiDevice:
    def test_to_warns_and_drops_device_when_n_devices_gt_1(self, gpt2_bridge):
        # Simulate a dispatched model by bumping n_devices — we don't need multi-GPU
        # hardware to verify the .to() guard path.
        original_n_devices = gpt2_bridge.cfg.n_devices
        gpt2_bridge.cfg.n_devices = 2
        try:
            with pytest.warns(UserWarning, match="ignored"):
                gpt2_bridge.to("cpu")
            assert next(gpt2_bridge.original_model.parameters()).device.type == "cpu"
        finally:
            gpt2_bridge.cfg.n_devices = original_n_devices

    def test_to_still_honors_dtype_under_multi_device(self, gpt2_bridge):
        original_n_devices = gpt2_bridge.cfg.n_devices
        original_dtype = next(gpt2_bridge.original_model.parameters()).dtype
        gpt2_bridge.cfg.n_devices = 2
        try:
            with pytest.warns(UserWarning, match="ignored"):
                gpt2_bridge.to("cpu", torch.float64)
            assert next(gpt2_bridge.original_model.parameters()).dtype == torch.float64
        finally:
            gpt2_bridge.cfg.n_devices = original_n_devices
            gpt2_bridge.original_model.to(original_dtype)


class TestRunWithCacheGuardsMultiDevice:
    def test_run_with_cache_device_arg_warns_under_multi_device(self, gpt2_bridge):
        original_n_devices = gpt2_bridge.cfg.n_devices
        gpt2_bridge.cfg.n_devices = 2
        try:
            with pytest.warns(UserWarning, match="ignored"):
                gpt2_bridge.run_with_cache(torch.tensor([[1, 2, 3]]), device="cpu")
        finally:
            gpt2_bridge.cfg.n_devices = original_n_devices


class TestStackedWeightsHandleCrossDevice:
    def test_stack_gathers_across_devices(self, gpt2_bridge):
        # Fake multi-device state by flipping cfg.n_devices. The GPT-2 bridge's weights
        # still live on CPU, so gathering to cfg.device (also CPU) is a no-op — but the
        # code path we care about (the `if n_devices > 1` branch) is exercised.
        original_n_devices = gpt2_bridge.cfg.n_devices
        gpt2_bridge.cfg.n_devices = 2
        try:
            # None of these should raise, even with n_devices>1.
            W_Q = gpt2_bridge.W_Q
            W_K = gpt2_bridge.W_K
            W_V = gpt2_bridge.W_V
            W_O = gpt2_bridge.W_O
            assert W_Q.shape[0] == gpt2_bridge.cfg.n_layers
            assert W_K.shape[0] == gpt2_bridge.cfg.n_layers
            assert W_V.shape[0] == gpt2_bridge.cfg.n_layers
            assert W_O.shape[0] == gpt2_bridge.cfg.n_layers
        finally:
            gpt2_bridge.cfg.n_devices = original_n_devices

    def test_accumulated_bias_handles_cross_device(self, gpt2_bridge):
        original_n_devices = gpt2_bridge.cfg.n_devices
        gpt2_bridge.cfg.n_devices = 2
        try:
            # Exercises the .to(accumulated.device) branch without requiring real GPUs.
            bias = gpt2_bridge.accumulated_bias(layer=gpt2_bridge.cfg.n_layers - 1)
            assert bias.shape == (gpt2_bridge.cfg.d_model,)
        finally:
            gpt2_bridge.cfg.n_devices = original_n_devices


class TestGeneralizedComponentOffloadMaterialization:
    """CPU/disk offload (#1280, #1459, #1493) hits meta tensors inside Bridge
    components because Accelerate only materializes real data around a hooked
    module's own forward() - components that read raw params directly (or via
    a synthetic sub-module built once at setup, e.g. split QKV) bypass that
    window. GeneralizedComponent.__call__ wraps every component call in
    Accelerate's own align_module_device, so this covers the leaf case without
    needing a real (slow) model download."""

    def _offloaded_layer_norm(self) -> tuple[torch.nn.LayerNorm, dict[str, torch.Tensor]]:
        module = torch.nn.LayerNorm(4)
        weights_map = {k: v.clone() for k, v in module.state_dict().items()}
        attach_align_device_hook(
            module, execution_device=torch.device("cpu"), offload=True, weights_map=weights_map
        )
        return module, weights_map

    def test_offloaded_leaf_module_starts_as_meta(self):
        # Establishes the failure mode this test guards against: Accelerate
        # offload really does leave the module's own weight as a meta
        # placeholder outside of a forward call.
        module, _ = self._offloaded_layer_norm()
        assert module.weight.is_meta

    def test_normalization_bridge_materializes_offloaded_params(self):
        module, weights_map = self._offloaded_layer_norm()

        class _Cfg:
            eps = 1e-5
            rmsnorm_uses_offset = False

        bridge = NormalizationBridge(name="ln", config=_Cfg(), uses_rms_norm=False)
        bridge.set_original_component(module)
        bridge.eval()

        x = torch.randn(2, 3, 4)
        with torch.no_grad():
            out = bridge(x)
        assert not torch.isnan(out).any()

        expected = torch.nn.functional.layer_norm(
            x, (4,), weight=weights_map["weight"], bias=weights_map["bias"], eps=1e-5
        )
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestDiskOffloadEndToEnd:
    """Real gpt2 end-to-end: disk device_map used to be rejected outright
    (#1280); now it should produce identical numerics to a non-offloaded load,
    including through gpt2's split-QKV path (JointQKVAttentionBridge builds
    q/k/v as one-time slices of c_attn's raw weight at setup - a case the
    generic per-call materialization alone doesn't cover, see
    set_original_component)."""

    def _device_map(self, disk_layers: set[int]) -> Dict[str, str]:
        device_map: Dict[str, str] = {
            "transformer.wte": "cpu",
            "transformer.wpe": "cpu",
            "transformer.ln_f": "cpu",
            "lm_head": "cpu",
        }
        for i in range(12):
            device_map[f"transformer.h.{i}"] = "disk" if i in disk_layers else "cpu"
        return device_map

    def test_disk_offload_matches_non_offloaded_forward_pass(self, gpt2_bridge, tmp_path):
        gpt2_bridge.eval()
        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (2, 8))
        with torch.no_grad():
            baseline_logits = gpt2_bridge(tokens).clone()
            baseline_loss = gpt2_bridge(tokens, return_type="loss").item()

        offloaded_bridge = TransformerBridge.boot_transformers(
            "gpt2",
            device_map=self._device_map({0, 3, 6, 9}),
            offload_folder=str(tmp_path),
        )
        offloaded_bridge.eval()
        with torch.no_grad():
            offloaded_logits = offloaded_bridge(tokens)
            offloaded_loss = offloaded_bridge(tokens, return_type="loss").item()

        torch.testing.assert_close(offloaded_logits, baseline_logits, atol=1e-4, rtol=1e-4)
        assert abs(offloaded_loss - baseline_loss) < 1e-4

    def test_disk_value_no_longer_raises_at_boot(self, tmp_path):
        # Regression guard for the ValueError this used to raise unconditionally.
        bridge = TransformerBridge.boot_transformers(
            "gpt2",
            device_map=self._device_map({0}),
            offload_folder=str(tmp_path),
        )
        assert bridge.original_model.hf_device_map["transformer.h.0"] == "disk"


# Real-hardware multi-GPU coverage lives in test_bridge_multigpu.py and
# test_bridge_multigpu_device_map.py (`-m multigpu`, >= 2 CUDA devices) — this file
# is the CPU/mocked tier: resolver, validation gates, and spoofed-n_devices guards.
