"""Unit tests for InspectDriver — mocks the inspect_ai Model so no provider runs.

The driver is the torch-free consumer side; these tests exercise it with a fake
async model returning a wire-encoded ModelOutput. Real end-to-end parity (gpt2
through our provider) lives in tests/acceptance/model_bridge/test_inspect_provider.py.
"""
from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# The driver lazily imports inspect_ai inside ``_generate``; without the ``inspect`` extra
# the import fires at first call and tests fail with ModuleNotFoundError. Skip-collect here.
pytest.importorskip("inspect_ai")

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.driver_protocol import (
    Driver,
    ForwardResult,
    validate_driver,
)
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources.inspect import (
    hooks,
    intervention,
    profiles,
    wire,
)
from transformer_lens.model_bridge.sources.inspect.driver import InspectDriver

N_LAYERS, D_MODEL, D_VOCAB = 2, 4, 16


def _cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=2,
        n_layers=N_LAYERS,
        n_ctx=8,
        n_heads=2,
        d_vocab=D_VOCAB,
        d_mlp=8,
        architecture="GPT2LMHeadModel",
    )


def _adapter() -> ArchitectureAdapter:
    adapter = ArchitectureAdapter(_cfg())
    adapter.component_mapping = {}
    return adapter


def _fake_model(*, captures=None, logits=None, raises=None):
    """Stand-in inspect_ai Model. ``captures``: {wire_key: (seq,d)}; ``logits``: (seq,vocab)."""
    metadata: dict = {}
    if captures is not None:
        metadata["activations"] = wire.encode_activations(captures)
    if logits is not None:
        metadata["tl_logits"] = wire.encode_array(logits)
    output = SimpleNamespace(metadata=metadata)

    model = SimpleNamespace()

    async def generate(_input, config=None):
        if raises is not None:
            raise raises
        return output

    model.generate = generate
    return model


def _driver(model=None) -> InspectDriver:
    return InspectDriver(model=model or _fake_model(), adapter=_adapter(), tokenizer=None)


class TestProtocolConformance:
    def test_is_driver_and_validates(self):
        driver = _driver()
        assert isinstance(driver, Driver)
        validate_driver(driver)

    def test_no_torch_capabilities(self):
        driver = _driver()
        for feature in ("parameters", "state_dict", "gradients", "weight_access"):
            assert driver.supports(feature) is False

    def test_provides_sequence_logits_true(self):
        # Provider returns full-sequence logits, so loss/both are available.
        assert _driver().provides_sequence_logits is True

    def test_tl_bridge_profile_can_disable_sequence_logits(self):
        # The vLLM provider routes through TLBridgeProfile with provides_sequence_logits=False
        # (set in source.py from api.provides_sequence_logits) so RemoteBridge.forward rejects
        # loss/both — synthesized logits only cover the gen position; earlier positions are -inf.
        driver = InspectDriver(
            _fake_model(),
            _adapter(),
            tokenizer=None,
            profile=profiles.TLBridgeProfile(provides_sequence_logits=False),
        )
        assert driver.provides_sequence_logits is False


class TestHookSets:
    def test_full_residual_attn_mlp_set(self):
        supported = _driver().supported_hook_points
        assert len(supported) == 5 * N_LAYERS  # one canonical name per boundary, no aliases
        for name in [
            "blocks.0.hook_in",  # resid_pre
            "blocks.0.ln2.hook_in",  # resid_mid
            "blocks.0.hook_out",  # resid_post
            "blocks.0.attn.hook_out",  # attn_out
            "blocks.0.mlp.hook_out",  # mlp_out
        ]:
            assert name in supported
        assert "blocks.0.hook_attn_out" not in supported  # HookedTransformer alias not duplicated

    def test_nonfireable_disjoint_and_includes_headsplit(self):
        driver = _driver()
        assert driver.supported_hook_points.isdisjoint(driver.non_fireable_hook_points)
        assert "blocks.0.attn.hook_pattern" in driver.non_fireable_hook_points
        assert "ln_final.hook_normalized" in driver.non_fireable_hook_points

    def test_vllm_lens_profile_narrows_to_residual(self):
        driver = InspectDriver(
            _fake_model(), _adapter(), tokenizer=None, profile=profiles.VLLMLensProfile()
        )
        assert driver.supported_hook_points == frozenset(
            f"blocks.{i}.hook_out" for i in range(N_LAYERS)
        )
        assert driver.provides_sequence_logits is False  # no full logits via that path
        # attn/mlp hooks our provider could do are non-fireable for the residual-only peer.
        assert "blocks.0.attn.hook_out" in driver.non_fireable_hook_points


class TestKindGating:
    """The provider's structural self-check feeds a kind set through to the driver."""

    def test_supported_hook_points_filters_kinds(self):
        only = hooks.supported_hook_points(2, kinds={"resid_pre", "resid_post"})
        assert only == frozenset(
            f"blocks.{i}.{suffix}" for i in range(2) for suffix in ("hook_in", "hook_out")
        )

    def test_profile_restricts_to_detected_kinds(self):
        # resid_mid gated (e.g. parallel-residual arch) → ln2.hook_in absent, rest present.
        prof = profiles.TLBridgeProfile(
            supported_kinds={"resid_pre", "resid_post", "attn_out", "mlp_out"}
        )
        names = prof.supported_hooks(N_LAYERS)
        assert "blocks.0.ln2.hook_in" not in names
        assert "blocks.0.attn.hook_out" in names
        assert len(names) == 4 * N_LAYERS

    def test_driver_moves_gated_kind_to_nonfireable(self):
        prof = profiles.TLBridgeProfile(
            supported_kinds={"resid_pre", "resid_post", "attn_out", "mlp_out"}
        )
        driver = InspectDriver(_fake_model(), _adapter(), tokenizer=None, profile=prof)
        assert "blocks.0.ln2.hook_in" not in driver.supported_hook_points
        assert "blocks.0.ln2.hook_in" in driver.non_fireable_hook_points


class TestNormalizeInputIds:
    def test_1d_tensor(self):
        assert InspectDriver._normalize_input_ids(torch.tensor([1, 2, 3])) == [1, 2, 3]

    def test_2d_single_row(self):
        assert InspectDriver._normalize_input_ids(torch.tensor([[1, 2, 3]])) == [1, 2, 3]

    def test_batch_gt_one_raises(self):
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            InspectDriver._normalize_input_ids(torch.tensor([[1, 2], [3, 4]]))

    def test_cuda_like_tensor_moved_to_cpu(self):
        # np.asarray can't read CUDA memory; the torch-free driver must duck-type
        # .detach().cpu() first. Simulate: __array__ raises (as a CUDA tensor would),
        # cpu() yields a real ndarray.
        class FakeCuda:
            def __init__(self, data):
                self._data = data
                self.moved = False

            def detach(self):
                return self

            def cpu(self):
                self.moved = True
                return np.asarray(self._data)

            def __array__(self, *a, **k):
                raise TypeError("can't convert cuda:0 device tensor to numpy")

        fake = FakeCuda([[1, 2, 3]])
        assert InspectDriver._normalize_input_ids(fake) == [1, 2, 3]
        assert fake.moved


class TestForward:
    def test_assembles_named_captures_and_full_logits(self):
        caps = {
            "0:resid_post": np.ones((3, D_MODEL), np.float32),
            "0:attn_out": np.full((3, D_MODEL), 2.0, np.float32),
        }
        logits = np.zeros((3, D_VOCAB), np.float32)
        logits[-1, 7] = 5.0
        driver = _driver(_fake_model(captures=caps, logits=logits))

        result = driver.forward(
            torch.tensor([[1, 2, 3]]),
            capture=("blocks.0.hook_out", "blocks.0.attn.hook_out"),
        )
        assert isinstance(result, ForwardResult)
        assert tuple(result.captured["blocks.0.hook_out"].shape) == (1, 3, D_MODEL)  # resid_post
        assert tuple(result.captured["blocks.0.attn.hook_out"].shape) == (1, 3, D_MODEL)  # attn_out
        assert tuple(result.logits.shape) == (1, 3, D_VOCAB)  # full sequence
        assert int(result.logits[0, -1].argmax()) == 7

    def test_empty_capture_captures_nothing(self):
        # capture=() means "logits only" — the driver requests no activations.
        driver = _driver(
            _fake_model(
                captures={"0:resid_post": np.ones((2, D_MODEL), np.float32)},
                logits=np.zeros((2, D_VOCAB), np.float32),
            )
        )
        result = driver.forward(torch.tensor([[1, 2]]))
        assert result.captured == {}
        assert result.logits is not None

    def test_missing_hook_warns(self):
        # Provider returns no activation for a requested+supported hook → warn, key absent.
        driver = _driver(_fake_model(captures={}, logits=np.zeros((2, D_VOCAB), np.float32)))
        with pytest.warns(UserWarning, match="no activation"):
            result = driver.forward(torch.tensor([[1, 2]]), capture=("blocks.0.hook_out",))
        assert "blocks.0.hook_out" not in result.captured

    def test_resid_mid_consumed_under_its_name(self):
        # The provider derives resid_mid (= resid_pre + attn_out) and sends it under
        # its wire key; the driver surfaces it as blocks.{i}.ln2.hook_in.
        caps = {"0:resid_mid": np.full((2, D_MODEL), 4.0, np.float32)}
        driver = _driver(_fake_model(captures=caps, logits=np.zeros((2, D_VOCAB), np.float32)))
        result = driver.forward(torch.tensor([[1, 2]]), capture=("blocks.0.ln2.hook_in",))
        assert np.allclose(result.captured["blocks.0.ln2.hook_in"][0], 4.0)

    def test_provider_error_propagates(self):
        driver = _driver(_fake_model(raises=RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            driver.forward(torch.tensor([[1, 2, 3]]))

    def test_rejects_max_new_tokens_gt_one(self):
        with pytest.raises(NotImplementedError, match="max_new_tokens=1"):
            _driver().forward(torch.tensor([[1, 2]]), max_new_tokens=2)

    def test_closed_driver_raises(self):
        driver = _driver()
        driver.close()
        with pytest.raises(RuntimeError, match="closed"):
            driver.forward(torch.tensor([[1, 2]]))


class TestAsyncWrapper:
    def test_reuses_one_loop_thread(self):
        caps = {"0:resid_post": np.ones((1, D_MODEL), np.float32)}
        driver = _driver(_fake_model(captures=caps, logits=np.zeros((1, D_VOCAB), np.float32)))
        driver.forward(torch.tensor([[1]]))
        first = driver._loop_thread
        driver.forward(torch.tensor([[1]]))
        assert driver._loop_thread is first and first.is_alive()

    def test_works_inside_running_loop(self):
        """The Jupyter case: a loop already running on the calling thread."""
        import asyncio

        caps = {"0:resid_post": np.ones((1, D_MODEL), np.float32)}
        driver = _driver(_fake_model(captures=caps, logits=np.zeros((1, D_VOCAB), np.float32)))

        async def run():
            return driver.forward(torch.tensor([[1]]))

        assert isinstance(asyncio.run(run()), ForwardResult)


class TestWire:
    def test_array_round_trip(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert np.array_equal(wire.decode_array(wire.encode_array(arr)), arr)

    def test_flat_activations_round_trip(self):
        caps = {
            "0:resid_post": np.ones((2, 4), np.float32),
            "1:attn_out": np.full((2, 4), 3.0, np.float32),
        }
        meta = {"activations": wire.encode_activations(caps)}
        out = wire.decode_activations(meta, ["0:resid_post", "1:attn_out"])
        assert np.array_equal(out["1:attn_out"], caps["1:attn_out"])

    def test_vllm_lens_residual_stream_fallback(self):
        # A peer provider's nested residual_stream decodes for resid_post keys.
        meta = {
            "activations": {
                "residual_stream": {"3": wire.encode_array(np.ones((2, 4), np.float32))}
            }
        }
        out = wire.decode_activations(meta, ["3:resid_post"])
        assert tuple(out["3:resid_post"].shape) == (2, 4)


class TestHooksRegistry:
    def test_resolve(self):
        assert hooks.resolve("blocks.2.attn.hook_out") == (2, "attn_out")
        assert hooks.resolve("blocks.2.mlp.hook_out") == (2, "mlp_out")
        assert hooks.resolve("blocks.0.ln2.hook_in") == (0, "resid_mid")
        assert hooks.resolve("blocks.0.hook_in") == (0, "resid_pre")
        assert hooks.resolve("blocks.0.hook_out") == (0, "resid_post")
        assert hooks.resolve("embed.hook_out") is None
        assert (
            hooks.resolve("blocks.2.hook_attn_out") is None
        )  # HookedTransformer alias not exposed


class TestInterventionTranslation:
    def _supported(self):
        return hooks.supported_hook_points(N_LAYERS)

    def test_op_translates_to_wire_key(self):
        out = intervention.build_interventions(
            {"blocks.1.hook_out": {"op": "suppress"}}, self._supported()
        )
        assert out == {"1:resid_post": {"op": "suppress"}}

    def test_attn_out_intervention(self):
        out = intervention.build_interventions(
            {"blocks.0.attn.hook_out": {"op": "scale", "factor": 0.5}}, self._supported()
        )
        assert out == {"0:attn_out": {"op": "scale", "factor": 0.5}}

    def test_resid_mid_is_capture_only(self):
        with pytest.raises(ValueError, match="capture-only"):
            intervention.build_interventions(
                {"blocks.0.ln2.hook_in": {"op": "suppress"}}, self._supported()
            )

    def test_callable_rejected(self):
        with pytest.raises(NotImplementedError, match="specs"):
            intervention.build_interventions({"blocks.0.hook_out": lambda a: a}, self._supported())

    def test_unknown_hook_rejected(self):
        with pytest.raises(ValueError, match="not in supported_hook_points"):
            intervention.build_interventions(
                {"blocks.9.hook_out": {"op": "suppress"}}, self._supported()
            )

    def test_bad_op_rejected(self):
        with pytest.raises(ValueError, match="Unsupported intervention op"):
            intervention.build_interventions(
                {"blocks.0.hook_out": {"op": "clamp"}}, self._supported()
            )


class TestVLLMLensProfile:
    """The vllm-lens request/response codec (live path unverified; logic is testable)."""

    def test_build_request_uses_output_residual_stream(self):
        tokenizer = SimpleNamespace(decode=lambda ids: "hello world")
        prompt, extra = profiles.VLLMLensProfile().build_request(
            [1, 2, 3], ["0:resid_post", "1:resid_post"], [], True, tokenizer
        )
        assert prompt == "hello world"  # detokenized — vllm-lens re-tokenizes
        assert extra == {"output_residual_stream": [0, 1]}

    def test_decode_logits_one_hot_from_completion(self):
        tokenizer = SimpleNamespace(encode=lambda text: [9])
        output = SimpleNamespace(completion="x", metadata={})
        logits = profiles.VLLMLensProfile().decode_logits(output, 3, D_VOCAB, tokenizer)
        assert tuple(logits.shape) == (1, 3, D_VOCAB)
        assert int(logits[0, -1].argmax()) == 9

    def test_non_additive_intervention_rejected_without_vllm_lens(self):
        # Validation happens before the lazy vllm_lens import, so this works uninstalled.
        with pytest.raises(NotImplementedError, match="additive steering"):
            profiles.VLLMLensProfile().translate_interventions(
                {"blocks.0.hook_out": {"op": "suppress"}},
                frozenset({"blocks.0.hook_out"}),
            )

    def test_no_interventions_needs_no_vllm_lens(self):
        assert profiles.VLLMLensProfile().translate_interventions({}, frozenset()) == []


class TestThroughBridge:
    def test_replays_as_torch(self):
        caps = {"0:resid_post": np.ones((3, D_MODEL), np.float32)}
        driver = _driver(_fake_model(captures=caps, logits=np.zeros((3, D_VOCAB), np.float32)))
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)

        fired: list = []
        bridge.add_hook("blocks.0.hook_out", lambda act, hook: fired.append(act))
        bridge.forward(torch.tensor([[1, 2, 3]]))
        assert len(fired) == 1 and isinstance(fired[0], torch.Tensor)  # numpy→torch at the bridge
        assert tuple(fired[0].shape) == (1, 3, D_MODEL)

    def test_loss_now_supported(self):
        # Full-sequence logits ⇒ the bridge computes loss (no longer rejected).
        logits = np.zeros((3, D_VOCAB), np.float32)
        driver = _driver(_fake_model(captures={}, logits=logits))
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)
        loss = bridge.forward(torch.tensor([[1, 2, 3]]), return_type="loss")
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0

    def test_context_manager_closes(self):
        driver = _driver()
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)
        with bridge:
            pass
        assert driver._model is None


class TestHeadSplitKinds:
    """Head-split q/k/v/z/pattern: registry entries, driver assembly, intervention gating.

    The provider serves these only when its structural probe finds the projections; here
    a profile with an explicit kind set stands in for that detection.
    """

    def _head_profile(self):
        return profiles.TLBridgeProfile(supported_kinds=hooks.ALL_KINDS | hooks.HEAD_KINDS)

    def test_resolve_head_names(self):
        assert hooks.resolve("blocks.3.attn.hook_q") == (3, "q")
        assert hooks.resolve("blocks.0.attn.hook_z") == (0, "z")
        assert hooks.resolve("blocks.1.attn.hook_pattern") == (1, "pattern")
        assert hooks.resolve("blocks.0.attn.hook_attn_scores") is None  # never served

    def test_default_supported_excludes_head_kinds(self):
        """kinds=None stays boundary-only — head kinds are opt-in via detection, so
        profiles that pass None (vllm-lens, defaults) don't silently claim them."""
        default = hooks.supported_hook_points(N_LAYERS)
        assert "blocks.0.attn.hook_q" not in default
        assert "blocks.0.attn.hook_pattern" not in default

    def test_all_hook_points_is_boundary_plus_head(self):
        universe = hooks.all_hook_points(N_LAYERS)
        assert hooks.supported_hook_points(N_LAYERS) < universe
        assert "blocks.0.attn.hook_q" in universe
        assert len(universe) == (len(hooks.ALL_KINDS) + len(hooks.HEAD_KINDS)) * N_LAYERS

    def test_interveneable_includes_qkvz_not_pattern(self):
        assert {"q", "k", "v", "z"} <= hooks.INTERVENEABLE_KINDS
        assert "pattern" not in hooks.INTERVENEABLE_KINDS

    def test_attn_scores_always_nonfireable(self):
        driver = InspectDriver(
            _fake_model(), _adapter(), tokenizer=None, profile=self._head_profile()
        )
        assert "blocks.0.attn.hook_attn_scores" in driver.non_fireable_hook_points

    def test_driver_serves_head_kinds_via_profile(self):
        driver = InspectDriver(
            _fake_model(), _adapter(), tokenizer=None, profile=self._head_profile()
        )
        for name in ("blocks.0.attn.hook_q", "blocks.1.attn.hook_z", "blocks.0.attn.hook_pattern"):
            assert name in driver.supported_hook_points
        assert driver.supported_hook_points.isdisjoint(driver.non_fireable_hook_points)

    def test_default_driver_moves_head_kinds_to_nonfireable(self):
        driver = _driver()  # default profile: boundary kinds only
        assert "blocks.0.attn.hook_q" in driver.non_fireable_hook_points
        assert "blocks.0.attn.hook_pattern" in driver.non_fireable_hook_points

    def test_head_captures_get_batch_dim(self):
        """3-D wire arrays (seq,heads,d_head) / (heads,q,k) unsqueeze to exactly one batch dim."""
        heads, d_head, seq = 2, 2, 3
        caps = {
            "0:q": np.ones((seq, heads, d_head), np.float32),
            "0:pattern": np.ones((heads, seq, seq), np.float32),
            "0:resid_post": np.ones((seq, D_MODEL), np.float32),
        }
        driver = InspectDriver(
            _fake_model(captures=caps, logits=np.zeros((seq, D_VOCAB), np.float32)),
            _adapter(),
            tokenizer=None,
            profile=self._head_profile(),
        )
        result = driver.forward(
            torch.tensor([[1, 2, 3]]),
            capture=("blocks.0.attn.hook_q", "blocks.0.attn.hook_pattern", "blocks.0.hook_out"),
        )
        assert tuple(result.captured["blocks.0.attn.hook_q"].shape) == (1, seq, heads, d_head)
        assert tuple(result.captured["blocks.0.attn.hook_pattern"].shape) == (1, heads, seq, seq)
        assert tuple(result.captured["blocks.0.hook_out"].shape) == (1, seq, D_MODEL)

    def test_qkvz_intervention_translates_to_wire_key(self):
        supported = hooks.supported_hook_points(N_LAYERS, hooks.ALL_KINDS | hooks.HEAD_KINDS)
        out = intervention.build_interventions(
            {"blocks.0.attn.hook_v": {"op": "suppress"}}, supported
        )
        assert out == {"0:v": {"op": "suppress"}}

    def test_pattern_intervention_rejected(self):
        supported = hooks.supported_hook_points(N_LAYERS, hooks.ALL_KINDS | hooks.HEAD_KINDS)
        with pytest.raises(ValueError, match="capture-only"):
            intervention.build_interventions(
                {"blocks.0.attn.hook_pattern": {"op": "suppress"}}, supported
            )

    def test_head_hook_rejected_when_gated(self):
        """A gated head hook (default boundary-only profile) can't be intervened on."""
        supported = hooks.supported_hook_points(N_LAYERS)  # no head kinds
        with pytest.raises(ValueError, match="not in supported_hook_points"):
            intervention.build_interventions(
                {"blocks.0.attn.hook_q": {"op": "suppress"}}, supported
            )

    def test_turn_activations_batch_dim_is_rank_aware(self):
        """Mixed 2-D boundary and 3-D head-split arrays in one turn all get exactly one
        batch dim — a rank-2-only rule would leave head-split arrays batchless, and a
        batchless (seq, heads, d_head) is shape-indistinguishable from a batched 2-D."""
        from transformer_lens.model_bridge.sources.inspect.eval import turn_activations

        payload = wire.encode_activations(
            {
                "0:resid_post": np.ones((3, D_MODEL), np.float32),
                "0:q": np.ones((3, 2, 2), np.float32),
                "0:pattern": np.ones((2, 3, 3), np.float32),
            }
        )
        sample = SimpleNamespace(
            events=[SimpleNamespace(output=SimpleNamespace(metadata={"activations": payload}))]
        )
        (turn,) = turn_activations(sample)
        assert turn["blocks.0.hook_out"].shape == (1, 3, D_MODEL)
        assert turn["blocks.0.attn.hook_q"].shape == (1, 3, 2, 2)
        assert turn["blocks.0.attn.hook_pattern"].shape == (1, 2, 3, 3)


class TestHeterogeneousLayers:
    """Hybrid attn/SSM stacks: capability detection probes layer 0 only, so a layer
    missing the targeted submodule must fail loud instead of installing nothing."""

    def _provider(self):
        import torch.nn as nn

        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            TransformerLensTransformersModelAPI,
        )

        class AttnBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(4, 4)
                self.mlp = nn.Linear(4, 4)

        class SSMBlock(nn.Module):  # no attention / MLP submodule
            def __init__(self):
                super().__init__()
                self.mixer = nn.Linear(4, 4)

        api = object.__new__(TransformerLensTransformersModelAPI)
        api._layers = nn.ModuleList([AttnBlock(), SSMBlock()])
        api._geometry = (2, 2, 2)
        return api

    def test_capture_on_layer_missing_attn_raises(self):
        with pytest.raises(RuntimeError, match="no attention submodule"):
            self._provider()._install_hooks({1: {"attn_out"}}, {}, {}, "cid")

    def test_intervention_on_layer_missing_attn_raises(self):
        with pytest.raises(RuntimeError, match="no attention submodule"):
            self._provider()._install_hooks({}, {1: {"attn_out": {"op": "suppress"}}}, {}, "cid")

    def test_capture_on_layer_missing_mlp_raises(self):
        with pytest.raises(RuntimeError, match="no MLP submodule"):
            self._provider()._install_hooks({1: {"mlp_out"}}, {}, {}, "cid")

    def test_head_kind_on_layer_missing_attn_raises(self):
        with pytest.raises(RuntimeError, match="no attention submodule"):
            self._provider()._install_hooks({1: {"q"}}, {}, {}, "cid")

    def test_layer_with_modules_still_installs(self):
        handles = self._provider()._install_hooks({0: {"attn_out", "mlp_out"}}, {}, {}, "cid")
        assert len(handles) == 2
        for handle in handles:
            handle.remove()

    def test_resid_kinds_on_ssm_layer_still_install(self):
        # Block in/out boundaries exist on every layer; no attn/mlp needed.
        handles = self._provider()._install_hooks({1: {"resid_pre", "resid_post"}}, {}, {}, "cid")
        assert len(handles) == 2
        for handle in handles:
            handle.remove()


class TestTimeoutRecovery:
    def test_timeout_abandons_wedged_loop_and_rebuilds(self, monkeypatch):
        """A hung provider forward blocks the loop thread; after the TimeoutError the
        driver must serve later forwards on a fresh loop, not queue behind the wedge."""
        import threading

        import transformer_lens.model_bridge.sources.inspect.driver as drv_mod

        monkeypatch.setattr(drv_mod, "_PROVIDER_TIMEOUT_S", 0.2)

        release = threading.Event()
        calls = {"n": 0}
        metadata = {"tl_logits": wire.encode_array(np.zeros((1, D_VOCAB), np.float32))}
        output = SimpleNamespace(metadata=metadata)

        async def generate(_input, config=None):
            calls["n"] += 1
            if calls["n"] == 1:
                release.wait(30)  # sync-block the loop thread, like a hung torch forward
            return output

        driver = _driver(SimpleNamespace(generate=generate))
        try:
            with pytest.warns(UserWarning, match="abandon"):
                with pytest.raises(TimeoutError, match="exceeded"):
                    driver.forward(torch.tensor([[1]]))
            wedged = driver._loop is None  # poisoned state cleared for rebuild
            assert wedged
            result = driver.forward(torch.tensor([[1]]))  # fresh loop, must not time out
            assert isinstance(result, ForwardResult)
            assert calls["n"] == 2
        finally:
            release.set()  # let the abandoned daemon thread finish

    def test_ensure_loop_is_thread_safe(self):
        import threading

        driver = _driver()
        loops: list = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            loops.append(driver._ensure_loop())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(loop) for loop in loops}) == 1
        driver.close()


class TestForProvider:
    def test_known_providers_resolve(self):
        assert isinstance(profiles.for_provider("vllm-lens"), profiles.VLLMLensProfile)
        assert isinstance(profiles.for_provider("vllm-lens-v2"), profiles.VLLMLensProfile)
        assert isinstance(profiles.for_provider("tl_bridge"), profiles.TLBridgeProfile)
        assert isinstance(profiles.for_provider("tl_bridge_vllm"), profiles.TLBridgeProfile)

    def test_unknown_provider_raises(self):
        # Falling back to the full-capability codec would boot then NaN downstream.
        with pytest.raises(ValueError, match="tl_bridge.*vllm-lens"):
            profiles.for_provider("hf")


class TestEvalArtifactEpochs:
    def test_epoch_in_artifact_filename(self, tmp_path, monkeypatch):
        """Multi-epoch evals reuse sample_ids; artifacts must not overwrite across epochs."""
        import asyncio

        from transformer_lens.model_bridge.sources.inspect import eval as eval_mod

        payload = wire.encode_activations({"0:resid_post": np.ones((2, D_MODEL), np.float32)})
        output = SimpleNamespace(metadata={"activations": payload})

        async def generate(_messages, config=None):
            return output

        monkeypatch.setattr(eval_mod, "get_model", lambda: SimpleNamespace(generate=generate))
        stored: dict = {}
        monkeypatch.setattr(
            eval_mod,
            "store",
            lambda: SimpleNamespace(set=lambda k, v: stored.__setitem__(k, v)),
        )

        solve = eval_mod.capture_activations(["blocks.0.hook_out"], output_dir=str(tmp_path))
        for epoch in (1, 2):
            state = SimpleNamespace(messages=[], sample_id="s1", epoch=epoch)
            asyncio.run(solve(state, None))
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "s1_epoch1.npz",
            "s1_epoch2.npz",
        ]
        assert stored["tl_activations_path"].endswith("s1_epoch2.npz")


class TestHFProviderGenerateConfig:
    """stop_seqs wiring + warn-once for unsupported GenerateConfig fields (HF provider)."""

    def _api(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            TransformerLensTransformersModelAPI,
        )

        api = object.__new__(TransformerLensTransformersModelAPI)
        api.model_name = "fake"
        api._device = "cpu"
        api._eval_capture = {}

        def tokenize(text, **kw):
            return SimpleNamespace(input_ids=[ord(c) % 50 for c in text])

        tok = SimpleNamespace(
            chat_template=None, pad_token_id=0, eos_token_id=1, decode=lambda ids, **kw: "x"
        )
        from unittest.mock import MagicMock

        callable_tok = MagicMock(wraps=tok)
        callable_tok.side_effect = tokenize
        callable_tok.chat_template = None
        callable_tok.pad_token_id = 0
        callable_tok.eos_token_id = 1
        callable_tok.decode = tok.decode
        api._tokenizer = callable_tok

        hf = MagicMock()
        hf.generate.return_value = SimpleNamespace(
            sequences=torch.tensor([[5, 6, 65]]), scores=None
        )
        api._hf = hf
        return api

    def _config(self, **overrides):
        from inspect_ai.model import GenerateConfig

        return GenerateConfig(**overrides)

    def test_stop_seqs_wired_to_hf_generate(self):
        api = self._api()
        api._generate_eval("hi", self._config(max_tokens=1, stop_seqs=["END"]), [])
        gen_kwargs = api._hf.generate.call_args.kwargs
        assert gen_kwargs["stop_strings"] == ["END"]
        assert gen_kwargs["tokenizer"] is api._tokenizer

    def test_unsupported_config_field_warns_once(self, monkeypatch):
        import warnings as warnings_mod

        from transformer_lens.model_bridge.sources.inspect import _provider_base

        monkeypatch.setattr(_provider_base, "_WARNED_UNSUPPORTED", set())
        api = self._api()
        with pytest.warns(UserWarning, match="frequency_penalty"):
            api._generate_eval("hi", self._config(max_tokens=1, frequency_penalty=0.5), [])
        with warnings_mod.catch_warnings(record=True) as rec:
            warnings_mod.simplefilter("always")
            api._generate_eval("hi", self._config(max_tokens=1, frequency_penalty=0.5), [])
        assert not [w for w in rec if "frequency_penalty" in str(w.message)]


class TestParityScript:
    SCRIPT = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "inspect_parity_report.py"

    def _load(self, monkeypatch, env=None):
        import importlib.util
        import uuid

        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        spec = importlib.util.spec_from_file_location(
            f"inspect_parity_{uuid.uuid4().hex}", self.SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_tolerance_env_overrides(self, monkeypatch):
        module = self._load(monkeypatch, {"TL_INSPECT_ATOL": "5e-2", "TL_INSPECT_RTOL": "7e-2"})
        assert module.ATOL == 5e-2 and module.RTOL == 7e-2

    def test_default_tolerances(self, monkeypatch):
        monkeypatch.delenv("TL_INSPECT_ATOL", raising=False)
        monkeypatch.delenv("TL_INSPECT_RTOL", raising=False)
        module = self._load(monkeypatch)
        assert module.ATOL == 1e-3 and module.RTOL == 1e-3

    def test_exit_code_1_on_fail(self, monkeypatch, capsys):
        module = self._load(monkeypatch, {"TL_PARITY_MODELS": "m1,m2"})
        rows = {
            "m1": {"model": "m1", "arch": "A", "status": "PASS", "detail": ""},
            "m2": {"model": "m2", "arch": "B", "status": "FAIL", "detail": "diff"},
        }
        monkeypatch.setattr(module, "verify", lambda m: rows[m])
        with pytest.raises(SystemExit) as excinfo:
            module.main()
        assert excinfo.value.code == 1

    def test_exit_code_0_on_pass_and_skip(self, monkeypatch, capsys):
        module = self._load(monkeypatch, {"TL_PARITY_MODELS": "m1,m2"})
        rows = {
            "m1": {"model": "m1", "arch": "A", "status": "PASS", "detail": ""},
            "m2": {"model": "m2", "arch": "B", "status": "SKIP", "detail": "gated"},
        }
        monkeypatch.setattr(module, "verify", lambda m: rows[m])
        with pytest.raises(SystemExit) as excinfo:
            module.main()
        assert excinfo.value.code == 0


def test_driver_imports_no_torch():
    """The acceptance gate: the driver file must not import torch (data-only boundary)."""
    import transformer_lens.model_bridge.sources.inspect.driver as drv

    tree = ast.parse(pathlib.Path(drv.__file__).read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] == "torch"]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "torch":
                offenders.append(node.module or "")
    assert not offenders, f"driver.py must be torch-free; found torch imports: {offenders}"
