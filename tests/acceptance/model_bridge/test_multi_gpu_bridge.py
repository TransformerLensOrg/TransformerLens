"""Multi-GPU support tests for TransformerBridge.

CPU-runnable tests exercise the resolver / param-plumbing / .to() guard /
validation logic. Tests requiring real multi-GPU hardware are marked skipif.
"""

from typing import Dict, Union

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities.multi_gpu import (
    count_unique_devices,
    find_embedding_device,
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
        user_mm: Dict[Union[str, int], str] = {0: "20GiB"}
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

    def test_n_devices_respects_user_max_memory(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Requires 2+ CUDA devices.")
        user_mm: Dict[Union[str, int], str] = {0: "10GiB", 1: "10GiB"}
        dm, mm = resolve_device_map(2, None, None, max_memory=user_mm)
        assert dm == "balanced"
        assert mm == user_mm

    def test_cpu_value_in_device_map_rejected(self):
        bad: Dict[str, Union[str, int]] = {"transformer.h.0": "cpu"}
        with pytest.raises(ValueError, match="not supported"):
            resolve_device_map(None, bad, None)

    def test_disk_value_in_device_map_rejected(self):
        bad: Dict[str, Union[str, int]] = {"transformer.h.0": "disk"}
        with pytest.raises(ValueError, match="not supported"):
            resolve_device_map(None, bad, None)


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


class TestBootParamValidation:
    def test_device_and_device_map_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            TransformerBridge.boot_transformers("gpt2", device="cpu", device_map="auto")

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


class TestSingleDevicePathUnchanged:
    def test_cpu_load_default_unchanged(self, gpt2_bridge):
        # If any of our changes broke the baseline path, existing bridge tests would
        # catch it too — this is a smoke check that n_devices stays 1 on the default path.
        assert gpt2_bridge.cfg.n_devices == 1
        assert gpt2_bridge.cfg.device is not None


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


# ---------- Multi-GPU tests (require real hardware) ----------


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ CUDA devices")
class TestMultiDeviceIntegration:
    def test_n_devices_matches_single_device_logits(self):
        single = TransformerBridge.boot_transformers("gpt2", device="cuda:0")
        multi = TransformerBridge.boot_transformers("gpt2", n_devices=2)

        assert multi.cfg.n_devices == 2
        assert single.cfg.n_devices == 1

        tokens = torch.tensor([[1, 2, 3, 4]])
        logits_single = single(tokens).to("cpu")
        logits_multi = multi(tokens).to("cpu")
        assert torch.allclose(logits_single, logits_multi, atol=1e-4, rtol=1e-4)

    def test_parameters_distributed_across_devices(self):
        bridge = TransformerBridge.boot_transformers("gpt2", n_devices=2)
        cuda_indices = {
            p.device.index for p in bridge.original_model.parameters() if p.device.type == "cuda"
        }
        assert cuda_indices == {0, 1}

    def test_generate_works_with_multi_device(self):
        bridge = TransformerBridge.boot_transformers("gpt2", n_devices=2)
        out = bridge.generate("Hello", max_new_tokens=3, do_sample=False)
        assert isinstance(out, str)
        assert len(out) > len("Hello")

    def test_stacked_weights_work_across_devices(self):
        # Real multi-device exercise of _stack_block_params (no spoofed n_devices).
        bridge = TransformerBridge.boot_transformers("gpt2", n_devices=2)
        W_Q = bridge.W_Q
        assert W_Q.shape[0] == bridge.cfg.n_layers
        # After stacking, all elements should be on cfg.device (the embedding device).
        assert bridge.cfg.device is not None
        assert W_Q.device == torch.device(bridge.cfg.device)

    def test_preloaded_device_map_model(self):
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
        bridge = TransformerBridge.boot_transformers("gpt2", hf_model=hf_model)
        assert bridge.cfg.n_devices >= 1
        assert bridge.cfg.device is not None
