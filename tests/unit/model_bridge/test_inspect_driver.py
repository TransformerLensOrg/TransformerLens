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


class TestHookSets:
    def test_full_residual_attn_mlp_set(self):
        supported = _driver().supported_hook_points
        assert len(supported) == 7 * N_LAYERS
        for name in [
            "blocks.0.hook_resid_pre",
            "blocks.0.hook_resid_mid",
            "blocks.0.hook_resid_post",
            "blocks.0.hook_attn_out",
            "blocks.0.attn.hook_out",
            "blocks.0.hook_mlp_out",
            "blocks.0.mlp.hook_out",
        ]:
            assert name in supported

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
            f"blocks.{i}.hook_resid_post" for i in range(N_LAYERS)
        )
        assert driver.provides_sequence_logits is False  # no full logits via that path
        # attn/mlp hooks our provider could do are non-fireable for the residual-only peer.
        assert "blocks.0.attn.hook_out" in driver.non_fireable_hook_points


class TestNormalizeInputIds:
    def test_1d_tensor(self):
        assert InspectDriver._normalize_input_ids(torch.tensor([1, 2, 3])) == [1, 2, 3]

    def test_2d_single_row(self):
        assert InspectDriver._normalize_input_ids(torch.tensor([[1, 2, 3]])) == [1, 2, 3]

    def test_batch_gt_one_raises(self):
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            InspectDriver._normalize_input_ids(torch.tensor([[1, 2], [3, 4]]))


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
            capture=("blocks.0.hook_resid_post", "blocks.0.attn.hook_out"),
        )
        assert isinstance(result, ForwardResult)
        assert tuple(result.captured["blocks.0.hook_resid_post"].shape) == (1, 3, D_MODEL)
        # alias attn.hook_out resolves to the same wire key 0:attn_out
        assert tuple(result.captured["blocks.0.attn.hook_out"].shape) == (1, 3, D_MODEL)
        assert tuple(result.logits.shape) == (1, 3, D_VOCAB)  # full sequence
        assert int(result.logits[0, -1].argmax()) == 7

    def test_resid_mid_consumed_under_its_name(self):
        # The provider derives resid_mid (= resid_pre + attn_out) and sends it under
        # its wire key; the driver surfaces it as blocks.{i}.hook_resid_mid.
        caps = {"0:resid_mid": np.full((2, D_MODEL), 4.0, np.float32)}
        driver = _driver(_fake_model(captures=caps, logits=np.zeros((2, D_VOCAB), np.float32)))
        result = driver.forward(torch.tensor([[1, 2]]), capture=("blocks.0.hook_resid_mid",))
        assert np.allclose(result.captured["blocks.0.hook_resid_mid"][0], 4.0)

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
    def test_resolve_aliases(self):
        assert hooks.resolve("blocks.2.attn.hook_out") == (2, "attn_out")
        assert hooks.resolve("blocks.2.hook_attn_out") == (2, "attn_out")
        assert hooks.resolve("blocks.0.hook_resid_mid") == (0, "resid_mid")
        assert hooks.resolve("embed.hook_out") is None


class TestInterventionTranslation:
    def _supported(self):
        return hooks.supported_hook_points(N_LAYERS)

    def test_op_translates_to_wire_key(self):
        out = intervention.build_interventions(
            {"blocks.1.hook_resid_post": {"op": "suppress"}}, self._supported()
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
                {"blocks.0.hook_resid_mid": {"op": "suppress"}}, self._supported()
            )

    def test_callable_rejected(self):
        with pytest.raises(NotImplementedError, match="specs"):
            intervention.build_interventions(
                {"blocks.0.hook_resid_post": lambda a: a}, self._supported()
            )

    def test_unknown_hook_rejected(self):
        with pytest.raises(ValueError, match="not in supported_hook_points"):
            intervention.build_interventions(
                {"blocks.9.hook_resid_post": {"op": "suppress"}}, self._supported()
            )

    def test_bad_op_rejected(self):
        with pytest.raises(ValueError, match="Unsupported intervention op"):
            intervention.build_interventions(
                {"blocks.0.hook_resid_post": {"op": "clamp"}}, self._supported()
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
                {"blocks.0.hook_resid_post": {"op": "suppress"}},
                frozenset({"blocks.0.hook_resid_post"}),
            )

    def test_no_interventions_needs_no_vllm_lens(self):
        assert profiles.VLLMLensProfile().translate_interventions({}, frozenset()) == []


class TestThroughBridge:
    def test_replays_as_torch(self):
        caps = {"0:resid_post": np.ones((3, D_MODEL), np.float32)}
        driver = _driver(_fake_model(captures=caps, logits=np.zeros((3, D_VOCAB), np.float32)))
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)

        fired: list = []
        bridge.add_hook("blocks.0.hook_resid_post", lambda act, hook: fired.append(act))
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
