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
from transformer_lens.model_bridge.sources.inspect import intervention, wire
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


def _fake_model(*, residual=None, last_logits=None, raises=None):
    """A stand-in inspect_ai Model whose async generate returns a wire-encoded output."""
    metadata: dict = {}
    if residual is not None:
        metadata["activations"] = wire.encode_activations(residual)
    if last_logits is not None:
        metadata["tl_last_logits"] = wire.encode_array(last_logits)
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

    def test_provides_sequence_logits_false(self):
        assert _driver().provides_sequence_logits is False


class TestHookSets:
    def test_supported_are_residual_post(self):
        assert _driver().supported_hook_points == frozenset(
            {"blocks.0.hook_resid_post", "blocks.1.hook_resid_post"}
        )

    def test_nonfireable_disjoint_and_includes_attn(self):
        driver = _driver()
        assert driver.supported_hook_points.isdisjoint(driver.non_fireable_hook_points)
        assert "blocks.0.attn.hook_pattern" in driver.non_fireable_hook_points
        assert "unembed.hook_out" in driver.non_fireable_hook_points


class TestNormalizeInputIds:
    def test_1d_tensor(self):
        assert InspectDriver._normalize_input_ids(torch.tensor([1, 2, 3])) == [1, 2, 3]

    def test_2d_single_row(self):
        assert InspectDriver._normalize_input_ids(torch.tensor([[1, 2, 3]])) == [1, 2, 3]

    def test_list(self):
        assert InspectDriver._normalize_input_ids([4, 5]) == [4, 5]

    def test_batch_gt_one_raises(self):
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            InspectDriver._normalize_input_ids(torch.tensor([[1, 2], [3, 4]]))


class TestForward:
    def test_assembles_captures_and_logits(self):
        residual = {0: np.ones((3, D_MODEL), np.float32), 1: np.full((3, D_MODEL), 2.0, np.float32)}
        last_logits = np.full((D_VOCAB,), -np.inf, np.float32)
        last_logits[7] = 5.0
        driver = _driver(_fake_model(residual=residual, last_logits=last_logits))

        result = driver.forward(torch.tensor([[1, 2, 3]]))
        assert isinstance(result, ForwardResult)
        emb = result.captured["blocks.0.hook_resid_post"]
        assert tuple(emb.shape) == (1, 3, D_MODEL)  # batch dim added
        assert result.logits is not None and tuple(result.logits.shape) == (1, 3, D_VOCAB)
        assert int(result.logits[0, -1].argmax()) == 7

    def test_capture_filters_layers(self):
        residual = {0: np.ones((2, D_MODEL), np.float32)}
        driver = _driver(_fake_model(residual=residual, last_logits=np.zeros(D_VOCAB, np.float32)))
        result = driver.forward(torch.tensor([[1, 2]]), capture=("blocks.0.hook_resid_post",))
        # Only layer 0 requested; provider echoes only what we asked for.
        assert set(result.captured) == {"blocks.0.hook_resid_post"}

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
        residual = {0: np.ones((1, D_MODEL), np.float32)}
        driver = _driver(_fake_model(residual=residual, last_logits=np.zeros(D_VOCAB, np.float32)))
        driver.forward(torch.tensor([[1]]))
        first = driver._loop_thread
        driver.forward(torch.tensor([[1]]))
        assert driver._loop_thread is first and first.is_alive()

    def test_works_inside_running_loop(self):
        """The Jupyter case: a loop already running on the calling thread."""
        import asyncio

        residual = {0: np.ones((1, D_MODEL), np.float32)}
        driver = _driver(_fake_model(residual=residual, last_logits=np.zeros(D_VOCAB, np.float32)))

        async def run():
            return driver.forward(torch.tensor([[1]]))

        result = asyncio.run(run())
        assert isinstance(result, ForwardResult)


class TestWire:
    def test_array_round_trip(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert np.array_equal(wire.decode_array(wire.encode_array(arr)), arr)

    def test_activations_round_trip_selects_layers(self):
        residual = {0: np.ones((2, 4), np.float32), 2: np.full((2, 4), 3.0, np.float32)}
        meta = {"activations": wire.encode_activations(residual)}
        out = wire.decode_activations(meta, [0, 2])
        assert set(out) == {0, 2} and np.array_equal(out[2], residual[2])

    def test_decode_missing_layer_skipped(self):
        meta = {"activations": wire.encode_activations({0: np.ones((1, 4), np.float32)})}
        assert set(wire.decode_activations(meta, [0, 1])) == {0}


class TestInterventionTranslation:
    def _h2l(self):
        return {"blocks.0.hook_resid_post": 0, "blocks.1.hook_resid_post": 1}

    def _supported(self):
        return frozenset(self._h2l())

    def test_ops_translate_to_layer_keyed_specs(self):
        out = intervention.build_extra_args(
            {"blocks.1.hook_resid_post": {"op": "suppress"}}, self._supported(), self._h2l()
        )
        assert out == {"1": {"op": "suppress"}}

    def test_callable_rejected(self):
        with pytest.raises(NotImplementedError, match="specs"):
            intervention.build_extra_args(
                {"blocks.0.hook_resid_post": lambda a: a}, self._supported(), self._h2l()
            )

    def test_unknown_hook_rejected(self):
        with pytest.raises(ValueError, match="not in supported_hook_points"):
            intervention.build_extra_args(
                {"blocks.9.hook_resid_post": {"op": "suppress"}}, self._supported(), self._h2l()
            )

    def test_bad_op_rejected(self):
        with pytest.raises(ValueError, match="Unsupported intervention op"):
            intervention.build_extra_args(
                {"blocks.0.hook_resid_post": {"op": "clamp"}}, self._supported(), self._h2l()
            )

    def test_scale_requires_factor(self):
        with pytest.raises(ValueError, match="requires 'factor'"):
            intervention.build_extra_args(
                {"blocks.0.hook_resid_post": {"op": "scale"}}, self._supported(), self._h2l()
            )


class TestThroughBridge:
    def test_replays_as_torch_and_rejects_loss(self):
        residual = {0: np.ones((3, D_MODEL), np.float32), 1: np.ones((3, D_MODEL), np.float32)}
        driver = _driver(_fake_model(residual=residual, last_logits=np.zeros(D_VOCAB, np.float32)))
        bridge = RemoteBridge(adapter=_adapter(), tokenizer=None, driver=driver)

        fired: list = []
        bridge.add_hook("blocks.0.hook_resid_post", lambda act, hook: fired.append(act))
        bridge.forward(torch.tensor([[1, 2, 3]]))
        assert len(fired) == 1 and isinstance(fired[0], torch.Tensor)  # numpy→torch at the bridge
        assert tuple(fired[0].shape) == (1, 3, D_MODEL)

        for rt in ("loss", "both"):
            with pytest.raises(NotImplementedError, match="return_type"):
                bridge.forward(torch.tensor([[1, 2, 3]]), return_type=rt)

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
