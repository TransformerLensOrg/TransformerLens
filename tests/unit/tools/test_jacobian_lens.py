"""Unit tests for the TransformerBridge-only Jacobian lens implementation."""

from contextlib import contextmanager
from enum import IntEnum
from inspect import Parameter, signature
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

import transformer_lens.tools.analysis.jacobian_lens as jacobian_lens_module
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import AltUpBlockBridge
from transformer_lens.model_bridge.supported_architectures.deepseek_v4 import (
    DeepseekV4BlockBridge,
)
from transformer_lens.tools.analysis import JacobianLens
from transformer_lens.utilities.activation_functions import apply_softcap

D_MODEL = 6
N_LAYERS = 4
D_VOCAB = 11
SEQ_LEN = 9
SKIP_FIRST = 2
CORPUS = "unit-test-corpus"


class _UnsafeMetadataEnum(IntEnum):
    VALUE = 7


class _UnsafeMetadataKey(str):
    pass


class _ToyBlock(nn.Module):
    def __init__(self, d_model: int, layer: int, dtype: torch.dtype):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        nn.init.normal_(self.linear.weight, std=0.2)
        self.hook_out = HookPoint()
        self.hook_out.name = f"blocks.{layer}.hook_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hook_out(residual + self.linear(residual))


class _CausalSumBlock(nn.Module):
    """Causal cross-position mixing with an exact triangular Jacobian."""

    def __init__(self, layer: int):
        super().__init__()
        self.hook_out = HookPoint()
        self.hook_out.name = f"blocks.{layer}.hook_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hook_out(residual.cumsum(dim=1))


class _ToyTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return f"token-{token_ids[0]}"


class _ToyBridge(TransformerBridge):
    """Small real ``TransformerBridge`` subclass with Bridge-native hooks.

    The production constructor needs a Hugging Face model and architecture
    adapter. Unit tests only need its public analysis surface, so this subclass
    initializes ``nn.Module`` directly while retaining the concrete
    ``TransformerBridge`` isinstance contract.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        causal_final_block: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            d_vocab=D_VOCAB,
            d_vocab_out=D_VOCAB,
            normalization_type="LN",
            output_logits_soft_cap=None,
            model_name="toy-bridge",
            dtype=dtype,
            device="cpu",
        )
        self.adapter = SimpleNamespace(
            supports_generation=True,
            get_component_mapping=lambda: {
                "blocks": SimpleNamespace(hook_out_is_single_residual_stream=True),
                "ln_final": object(),
                "unembed": object(),
            },
            validate_output_logits_transform=lambda: None,
            apply_output_logits_transform=lambda logits: apply_softcap(
                logits, self.cfg.output_logits_soft_cap
            ),
        )
        self.compatibility_mode = False
        self._weights_processed = False
        self.tokenizer = _ToyTokenizer()
        self.embed = nn.Embedding(D_VOCAB, D_MODEL, dtype=dtype)
        blocks: list[nn.Module] = [_ToyBlock(D_MODEL, layer, dtype) for layer in range(N_LAYERS)]
        if causal_final_block:
            blocks[-1] = _CausalSumBlock(N_LAYERS - 1)
        self.blocks = nn.ModuleList(blocks)
        self.ln_final = nn.Identity()
        self.unembed = nn.Linear(D_MODEL, D_VOCAB, bias=False, dtype=dtype)

    @property
    def W_U(self) -> torch.Tensor:
        return self.unembed.weight.T

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        return {
            f"blocks.{layer}.hook_out": block.hook_out for layer, block in enumerate(self.blocks)
        }

    def parameters(self, recurse: bool = True):
        # A production bridge delegates this to its wrapped HF model. This toy
        # owns its small modules directly, so enumerate the nn.Module tree.
        return nn.Module.parameters(self, recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        return nn.Module.named_parameters(
            self,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def to_tokens(self, prompt: str) -> torch.Tensor:
        ids = [(3 * index + len(prompt)) % D_VOCAB for index in range(SEQ_LEN)]
        return torch.tensor([ids], dtype=torch.long)

    def to_single_token(self, string: str) -> int:
        return len(string) % D_VOCAB

    def forward(
        self, tokens: torch.Tensor, return_type: str | None = "logits"
    ) -> torch.Tensor | None:
        residual = self.embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        if return_type is None:
            return None
        return self.unembed(self.ln_final(residual))

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[str, Any]] = [],
        bwd_hooks: list[tuple[str, Any]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        del clear_contexts
        added: list[tuple[HookPoint, str, Any]] = []
        for direction, hook_specs in (("fwd", fwd_hooks), ("bwd", bwd_hooks)):
            for name, hook_fn in hook_specs:
                hook_point = self.hook_dict[name]
                hook_point.add_hook(hook_fn, dir=direction)
                handles = hook_point.fwd_hooks if direction == "fwd" else hook_point.bwd_hooks
                added.append((hook_point, direction, handles[-1]))
        try:
            yield self
        finally:
            if reset_hooks_end:
                for hook_point, direction, handle in added:
                    handle.hook.remove()
                    handles = hook_point.fwd_hooks if direction == "fwd" else hook_point.bwd_hooks
                    if handle in handles:
                        handles.remove(handle)

    def run_with_cache(
        self,
        input: torch.Tensor,
        return_cache_object: bool = False,
        remove_batch_dim: bool = False,
        names_filter: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del return_cache_object, remove_batch_dim, kwargs

        def wanted(name: str) -> bool:
            if names_filter is None:
                return True
            if isinstance(names_filter, str):
                return name == names_filter
            if callable(names_filter):
                return bool(names_filter(name))
            return name in names_filter

        cache: dict[str, torch.Tensor] = {}

        def cache_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            assert hook.name is not None
            cache[hook.name] = activation.detach()
            return activation

        cache_hooks = [(name, cache_hook) for name in self.hook_dict if wanted(name)]
        with self.hooks(fwd_hooks=cache_hooks):
            logits = self(input)
        assert logits is not None
        return logits, cache


class _NotABridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL)


def _closed_form_jacobian(model: _ToyBridge, layer: int) -> torch.Tensor:
    """Exact d h_final / d h_layer for the position-wise linear toy."""
    jacobian = torch.eye(D_MODEL)
    for index in range(layer + 1, N_LAYERS):
        block = model.blocks[index]
        assert isinstance(block, _ToyBlock)
        jacobian = (torch.eye(D_MODEL) + block.linear.weight) @ jacobian
    return jacobian


def _lens(
    *,
    n_prompts: int = 1,
    metadata: dict[str, Any] | None = None,
) -> JacobianLens:
    return JacobianLens(
        {0: torch.eye(D_MODEL)},
        n_prompts=n_prompts,
        d_model=D_MODEL,
        metadata=metadata,
    )


@pytest.fixture(scope="module")
def toy_model() -> _ToyBridge:
    return _ToyBridge()


@pytest.fixture(scope="module")
def fitted_lens(toy_model: _ToyBridge) -> JacobianLens:
    return JacobianLens.fit(
        toy_model,
        ["a toy prompt", "another toy prompt"],
        corpus=CORPUS,
        dim_batch=4,
        skip_first_positions=SKIP_FIRST,
        show_progress=False,
        metadata={"run": {"seed": 0, "tags": ["unit", "bridge"]}},
    )


def test_toy_model_satisfies_raw_bridge_contract(toy_model: _ToyBridge) -> None:
    assert isinstance(toy_model, TransformerBridge)
    assert toy_model.compatibility_mode is False
    assert toy_model._weights_processed is False
    assert set(toy_model.hook_dict) == {f"blocks.{layer}.hook_out" for layer in range(N_LAYERS)}


def test_fit_recovers_closed_form_jacobians(
    toy_model: _ToyBridge, fitted_lens: JacobianLens
) -> None:
    assert fitted_lens.source_layers == [0, 1, 2]
    assert fitted_lens.n_prompts == 2
    for layer in fitted_lens.source_layers:
        torch.testing.assert_close(
            fitted_lens.jacobians[layer],
            _closed_form_jacobian(toy_model, layer),
            atol=1e-5,
            rtol=1e-4,
        )


def test_fit_sums_only_causal_target_positions() -> None:
    model = _ToyBridge(causal_final_block=True)
    lens = JacobianLens.fit(
        model,
        ["causal mixing"],
        corpus=CORPUS,
        source_layers=[N_LAYERS - 2],
        dim_batch=4,
        skip_first_positions=SKIP_FIRST,
        show_progress=False,
    )
    n_valid_positions = SEQ_LEN - SKIP_FIRST - 1
    expected_scale = (n_valid_positions + 1) / 2
    torch.testing.assert_close(
        lens.jacobians[N_LAYERS - 2],
        expected_scale * torch.eye(D_MODEL),
        atol=1e-6,
        rtol=1e-6,
    )


def test_transport_orientation(toy_model: _ToyBridge, fitted_lens: JacobianLens) -> None:
    residual = torch.randn(3, D_MODEL)
    expected = residual @ _closed_form_jacobian(toy_model, 2).T
    torch.testing.assert_close(fitted_lens.transport(residual, 2), expected)


def test_device_jacobian_cache_is_reused_by_public_operations(
    toy_model: _ToyBridge,
) -> None:
    lens = _lens()
    assert lens._device_jacobians == {}
    lens.transport(torch.randn(2, D_MODEL), 0)
    cached = lens._device_jacobians[(0, torch.device("cpu"))]
    lens.lens_vectors(toy_model, 3, 0)
    assert lens._device_jacobians[(0, torch.device("cpu"))] is cached
    assert len(lens._device_jacobians) == 1
    lens.clear_device_cache()
    assert lens._device_jacobians == {}


def test_readout_defaults_to_topk_only(toy_model: _ToyBridge, fitted_lens: JacobianLens) -> None:
    result = fitted_lens.readout(toy_model, "a toy prompt", layers=[0, 3], top_k=3)
    assert result.lens_logits is None
    assert result.model_logits is None
    assert set(result.lens_topk_values) == {0, 3}
    assert result.lens_topk_values[0].shape == (SEQ_LEN, 3)
    assert result.lens_topk_indices[0].shape == (SEQ_LEN, 3)
    assert result.model_topk_values.shape == (SEQ_LEN, 3)
    assert result.model_topk_indices.shape == (SEQ_LEN, 3)


def test_readout_full_logits_are_opt_in_and_back_topk(
    toy_model: _ToyBridge, fitted_lens: JacobianLens
) -> None:
    result = fitted_lens.readout(
        toy_model,
        "a toy prompt",
        layers=[0, 2, N_LAYERS - 1],
        top_k=4,
        return_full_logits=True,
    )
    assert result.lens_logits is not None
    assert result.model_logits is not None
    for layer, logits in result.lens_logits.items():
        expected = logits.topk(4, dim=-1)
        torch.testing.assert_close(result.lens_topk_values[layer], expected.values)
        assert torch.equal(result.lens_topk_indices[layer], expected.indices)
    expected_model = result.model_logits.topk(4, dim=-1)
    torch.testing.assert_close(result.model_topk_values, expected_model.values)
    assert torch.equal(result.model_topk_indices, expected_model.indices)


def test_readout_final_layer_reuses_model_outputs(
    toy_model: _ToyBridge, fitted_lens: JacobianLens
) -> None:
    final_layer = N_LAYERS - 1
    result = fitted_lens.readout(
        toy_model,
        "a toy prompt",
        layers=[final_layer],
        return_full_logits=True,
    )
    assert result.lens_logits is not None
    assert result.model_logits is not None
    torch.testing.assert_close(result.lens_logits[final_layer], result.model_logits)
    torch.testing.assert_close(result.lens_topk_values[final_layer], result.model_topk_values)
    assert torch.equal(result.lens_topk_indices[final_layer], result.model_topk_indices)


def test_unembed_rows_applies_softcap_before_fp32_conversion() -> None:
    model = _ToyBridge(dtype=torch.bfloat16)
    model.cfg.output_logits_soft_cap = 0.75
    residual = 4 * torch.randn(3, D_MODEL)

    actual = jacobian_lens_module._unembed(model, residual)
    raw_logits = model.unembed(model.ln_final(residual.to(torch.bfloat16).unsqueeze(0))).squeeze(0)
    expected = (
        model.cfg.output_logits_soft_cap * torch.tanh(raw_logits / model.cfg.output_logits_soft_cap)
    ).float()
    fp32_first = model.cfg.output_logits_soft_cap * torch.tanh(
        raw_logits.float() / model.cfg.output_logits_soft_cap
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, fp32_first)


def test_unembed_rows_applies_architecture_logit_scale_before_softcap() -> None:
    model = _ToyBridge()
    model.cfg.logit_scale = 0.125
    model.cfg.output_logits_soft_cap = 0.75
    model.adapter.apply_output_logits_transform = lambda logits: apply_softcap(
        logits * model.cfg.logit_scale, model.cfg.output_logits_soft_cap
    )
    residual = 4 * torch.randn(3, D_MODEL)

    actual = jacobian_lens_module._unembed(model, residual)
    raw_logits = model.unembed(model.ln_final(residual.unsqueeze(0))).squeeze(0)
    scaled_logits = raw_logits * model.cfg.logit_scale
    expected = model.cfg.output_logits_soft_cap * torch.tanh(
        scaled_logits / model.cfg.output_logits_soft_cap
    )
    wrong_order = (
        model.cfg.logit_scale
        * model.cfg.output_logits_soft_cap
        * torch.tanh(raw_logits / model.cfg.output_logits_soft_cap)
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong_order)


def test_unembed_rows_ignores_unowned_logit_scale_config() -> None:
    model = _ToyBridge()
    model.cfg.logit_scale = "inv_sqrt_d_model"
    residual = torch.randn(2, D_MODEL)

    actual = jacobian_lens_module._unembed(model, residual)
    expected = model.unembed(model.ln_final(residual.unsqueeze(0))).squeeze(0).float()

    torch.testing.assert_close(actual, expected)


def test_transported_readout_matches_linear_model(
    toy_model: _ToyBridge, fitted_lens: JacobianLens
) -> None:
    result = fitted_lens.readout(
        toy_model,
        "a toy prompt",
        layers=[0, 2],
        return_full_logits=True,
    )
    assert result.lens_logits is not None
    assert result.model_logits is not None
    for layer in (0, 2):
        torch.testing.assert_close(
            result.lens_logits[layer], result.model_logits, atol=1e-3, rtol=1e-3
        )


def test_save_load_roundtrip_preserves_safe_provenance(
    tmp_path: Any, fitted_lens: JacobianLens
) -> None:
    path = str(tmp_path / "lens.pt")
    fitted_lens.save(path)
    loaded = JacobianLens.load(path)
    assert loaded.source_layers == fitted_lens.source_layers
    assert loaded.n_prompts == fitted_lens.n_prompts
    assert loaded.d_model == fitted_lens.d_model
    assert loaded.metadata == fitted_lens.metadata
    for layer in loaded.source_layers:
        torch.testing.assert_close(
            loaded.jacobians[layer], fitted_lens.jacobians[layer], atol=2e-3, rtol=2e-3
        )
        assert loaded.jacobians[layer].dtype == torch.float32


@pytest.mark.parametrize(
    "metadata",
    [
        {"numpy_scalar": np.int64(7)},
        {"enum_scalar": _UnsafeMetadataEnum.VALUE},
        {_UnsafeMetadataKey("custom_key"): 7},
        {"nested": {"values": ["safe", object()]}},
    ],
)
def test_save_rejects_metadata_that_weights_only_cannot_load(
    tmp_path: Any, metadata: dict[str, Any]
) -> None:
    lens = _lens(metadata=metadata)
    with pytest.raises((TypeError, ValueError), match="metadata"):
        lens.save(str(tmp_path / "unsafe.pt"))


def test_load_official_format_without_metadata(tmp_path: Any) -> None:
    path = str(tmp_path / "official.pt")
    torch.save(
        {
            "J": {0: torch.eye(D_MODEL, dtype=torch.float16)},
            "n_prompts": 7,
            "source_layers": [0],
            "d_model": D_MODEL,
        },
        path,
    )
    lens = JacobianLens.load(path)
    assert lens.n_prompts == 7
    assert lens.metadata == {}
    assert lens.jacobians[0].dtype == torch.float32


def test_load_rejects_fit_checkpoints(tmp_path: Any) -> None:
    path = str(tmp_path / "ckpt.pt")
    torch.save({"jacobian_sum": {}, "n_done": 3}, path)
    with pytest.raises(ValueError, match="no 'J' key"):
        JacobianLens.load(path)


def test_from_pretrained_uses_retry_and_forwards_hub_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import huggingface_hub

    path = tmp_path / "lens.pt"
    _lens().save(str(path))
    calls: dict[str, Any] = {}

    def fake_download(**kwargs: Any) -> str:
        raise AssertionError(f"download must be called through retry: {kwargs}")

    def fake_retry(function: Any, **kwargs: Any) -> str:
        calls["function"] = function
        calls["kwargs"] = kwargs
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(jacobian_lens_module, "call_hf_with_retry", fake_retry)

    loaded = JacobianLens.from_pretrained(
        "example/lenses",
        filename="model/lens.pt",
        revision="0123456789abcdef",
    )

    assert calls == {
        "function": fake_download,
        "kwargs": {
            "repo_id": "example/lenses",
            "filename": "model/lens.pt",
            "revision": "0123456789abcdef",
        },
    }
    assert loaded.source_layers == _lens().source_layers


def test_merge_is_n_prompts_weighted() -> None:
    lens_a = JacobianLens({0: torch.ones(D_MODEL, D_MODEL)}, n_prompts=2, d_model=D_MODEL)
    lens_b = JacobianLens({0: 4 * torch.ones(D_MODEL, D_MODEL)}, n_prompts=6, d_model=D_MODEL)
    merged = JacobianLens.merge([lens_a, lens_b])
    assert merged.n_prompts == 8
    torch.testing.assert_close(merged.jacobians[0], torch.full((D_MODEL, D_MODEL), 3.25))


@pytest.mark.parametrize("invalid_n_prompts", [0, -1])
def test_merge_rejects_each_nonpositive_shard(invalid_n_prompts: int) -> None:
    valid = _lens(n_prompts=2)
    invalid = _lens(n_prompts=invalid_n_prompts)
    with pytest.raises(ValueError, match="positive n_prompts"):
        JacobianLens.merge([valid, invalid])


def test_merge_rejects_mismatched_or_empty_lenses() -> None:
    lens_a = _lens()
    lens_b = JacobianLens({1: torch.ones(D_MODEL, D_MODEL)}, n_prompts=1, d_model=D_MODEL)
    with pytest.raises(ValueError, match="must share"):
        JacobianLens.merge([lens_a, lens_b])
    with pytest.raises(ValueError, match="empty sequence"):
        JacobianLens.merge([])


def test_merge_rejects_mismatched_provenance() -> None:
    lens_a = _lens(metadata={"model_name": "model-a", "n_prompts": 1})
    lens_b = _lens(metadata={"model_name": "model-b", "n_prompts": 2})
    with pytest.raises(ValueError, match="provenance metadata"):
        JacobianLens.merge([lens_a, lens_b])

    matching = _lens(metadata={"model_name": "model-a", "n_prompts": 3})
    merged = JacobianLens.merge([lens_a, matching])
    assert merged.metadata == {"model_name": "model-a", "n_prompts": 2}


def test_validate_model_requires_raw_bridge(toy_model: _ToyBridge) -> None:
    assert _lens().validate_model(toy_model) is not None
    with pytest.raises(TypeError, match="TransformerBridge"):
        _lens().validate_model(_NotABridge())

    compatibility_model = _ToyBridge()
    compatibility_model.compatibility_mode = True
    with pytest.raises(ValueError, match="compatibility mode"):
        _lens().validate_model(compatibility_model)

    processed_model = _ToyBridge()
    processed_model._weights_processed = True
    with pytest.raises(ValueError, match="process_weights"):
        _lens().validate_model(processed_model)


def test_validate_model_rejects_noncausal_attention() -> None:
    model = _ToyBridge()
    model.cfg.attention_dir = "bidirectional"

    with pytest.raises(ValueError, match="requires causal attention"):
        _lens().validate_model(model)
    with pytest.raises(ValueError, match="requires causal attention"):
        JacobianLens.fit(model, ["long enough"], corpus="toy", show_progress=False)


def test_validate_model_rejects_looped_depth_hooks() -> None:
    model = _ToyBridge()
    model.cfg.total_ut_steps = 4

    with pytest.raises(ValueError, match="total_ut_steps=4"):
        _lens().validate_model(model)


def test_validate_model_rejects_non_generation_adapter() -> None:
    model = _ToyBridge()
    model.adapter.supports_generation = False

    with pytest.raises(ValueError, match="causal decoder-only"):
        _lens().validate_model(model)


def test_validate_model_rejects_unsupported_final_output_paths() -> None:
    missing_norm = _ToyBridge()
    del missing_norm.ln_final
    with pytest.raises(ValueError, match="missing.*ln_final"):
        _lens().validate_model(missing_norm)

    projected_unembed = _ToyBridge()
    projected_unembed.unembed = nn.Linear(D_MODEL - 1, D_VOCAB, bias=False)
    with pytest.raises(ValueError, match="final output projection"):
        _lens().validate_model(projected_unembed)

    projected_output = _ToyBridge()
    projected_output.adapter.get_component_mapping = lambda: {
        "blocks": SimpleNamespace(hook_out_is_single_residual_stream=True),
        "ln_final": object(),
        "project_out": object(),
        "unembed": object(),
    }
    with pytest.raises(ValueError, match="final output projection"):
        _lens().validate_model(projected_output)

    altup_model = _ToyBridge()
    altup_model.adapter.get_component_mapping = lambda: {
        "blocks": AltUpBlockBridge(name="blocks"),
        "ln_final": object(),
        "unembed": object(),
    }
    with pytest.raises(ValueError, match="single-stream.*AltUpBlockBridge"):
        _lens().validate_model(altup_model)

    mhc_model = _ToyBridge()
    mhc_model.adapter.get_component_mapping = lambda: {
        "blocks": DeepseekV4BlockBridge(name="blocks"),
        "ln_final": object(),
        "unembed": object(),
    }
    with pytest.raises(ValueError, match="single-stream.*DeepseekV4BlockBridge"):
        _lens().validate_model(mhc_model)


def test_residual_runtime_shape_guard() -> None:
    with pytest.raises(ValueError, match=r"blocks.0.hook_out.*\(2, 1, 3, 6\)"):
        jacobian_lens_module._validate_residual_activation(
            torch.zeros(2, 1, 3, D_MODEL),
            d_model=D_MODEL,
            hook_name="blocks.0.hook_out",
        )
    with pytest.raises(ValueError, match=r"blocks.0.hook_out.*\(1, 3, 5\)"):
        jacobian_lens_module._validate_residual_activation(
            torch.zeros(1, 3, D_MODEL - 1),
            d_model=D_MODEL,
            hook_name="blocks.0.hook_out",
        )


def test_validate_model_rejects_wrong_shape_and_layers(toy_model: _ToyBridge) -> None:
    wrong_width = JacobianLens({0: torch.ones(3, 3)}, n_prompts=1, d_model=3)
    with pytest.raises(ValueError, match="d_model"):
        wrong_width.validate_model(toy_model)

    wrong_layer = JacobianLens(
        {N_LAYERS + 5: torch.ones(D_MODEL, D_MODEL)},
        n_prompts=1,
        d_model=D_MODEL,
    )
    with pytest.raises(ValueError, match="different model"):
        wrong_layer.validate_model(toy_model)


def test_validate_model_enforces_recorded_model_provenance() -> None:
    model = _ToyBridge()
    original_model = nn.Module()
    setattr(original_model, "config", SimpleNamespace(_commit_hash="revision-b"))
    model.__dict__["original_model"] = original_model

    with pytest.raises(ValueError, match="model 'different-model'"):
        _lens(metadata={"model_name": "different-model"}).validate_model(model)
    with pytest.raises(ValueError, match="model revision 'revision-a'"):
        _lens(metadata={"model_name": "toy-bridge", "model_revision": "revision-a"}).validate_model(
            model
        )

    matching = _lens(metadata={"model_name": "toy-bridge", "model_revision": "revision-b"})
    assert matching.validate_model(model) is matching


def test_validate_model_rejects_nonfinal_target_metadata(toy_model: _ToyBridge) -> None:
    lens = _lens(metadata={"target_layer": N_LAYERS - 2})
    with pytest.raises(ValueError, match=r"final.?layer"):
        lens.validate_model(toy_model)


def test_readout_bounds_checks(toy_model: _ToyBridge, fitted_lens: JacobianLens) -> None:
    with pytest.raises(ValueError, match="out of range"):
        fitted_lens.readout(toy_model, "a toy prompt", layers=[99], use_jacobian=False)
    with pytest.raises(ValueError, match="out of range"):
        fitted_lens.readout(toy_model, "a toy prompt", positions=[-2 * SEQ_LEN])


def test_fit_skips_short_prompts(toy_model: _ToyBridge) -> None:
    with pytest.raises(ValueError, match="too short to contribute"):
        with pytest.warns(UserWarning, match="skipping prompt"):
            JacobianLens.fit(
                toy_model,
                ["a toy prompt"],
                corpus=CORPUS,
                dim_batch=6,
                skip_first_positions=SEQ_LEN,
                show_progress=False,
            )


def test_fit_rejects_negative_skip_first_positions(toy_model: _ToyBridge) -> None:
    with pytest.raises(ValueError, match="skip_first_positions must be >= 0"):
        JacobianLens.fit(
            toy_model,
            ["a toy prompt"],
            corpus=CORPUS,
            skip_first_positions=-1,
            show_progress=False,
        )


def test_fit_signature_requires_corpus_and_has_no_target_layer() -> None:
    parameters = signature(JacobianLens.fit).parameters
    assert parameters["corpus"].default is Parameter.empty
    assert "target_layer" not in parameters


def test_fit_records_complete_provenance(fitted_lens: JacobianLens) -> None:
    metadata = fitted_lens.metadata
    assert metadata["model_name"] == "toy-bridge"
    assert metadata["model_revision"] is None
    assert metadata["corpus"] == CORPUS
    assert metadata["n_prompts"] == fitted_lens.n_prompts
    assert metadata["hook_convention"] == "blocks.{layer}.hook_out"
    assert metadata["processing"] == {
        "compatibility_mode": False,
        "weight_basis": "raw_huggingface",
    }
    assert metadata["fit_dtype"].endswith("float32")
    assert isinstance(metadata["transformer_lens_version"], str)
    assert metadata["transformer_lens_version"]


def test_fit_warns_for_low_precision_model() -> None:
    model = _ToyBridge(dtype=torch.bfloat16)
    with pytest.warns(UserWarning, match="float32"):
        lens = JacobianLens.fit(
            model,
            ["low precision"],
            corpus=CORPUS,
            source_layers=[N_LAYERS - 2],
            dim_batch=3,
            skip_first_positions=SKIP_FIRST,
            show_progress=False,
        )
    assert lens.metadata["fit_dtype"].endswith("bfloat16")


def test_lens_vectors_validates_model() -> None:
    compatibility_model = _ToyBridge()
    compatibility_model.compatibility_mode = True
    with pytest.raises(ValueError, match="compatibility mode"):
        _lens().lens_vectors(compatibility_model, 3, 0)


@pytest.mark.parametrize("tokens", [[], [-1], [D_VOCAB]])
def test_lens_vectors_rejects_empty_or_out_of_range_tokens(tokens: list[int]) -> None:
    model = _ToyBridge()
    match = "at least one token" if not tokens else "out of range"
    with pytest.raises(ValueError, match=match):
        _lens().lens_vectors(model, tokens, 0)


def _intervention_hooks(
    name: str,
    lens: JacobianLens,
    model: _ToyBridge,
    positions: list[int] | None,
) -> list[tuple[str, Any]]:
    if name == "steering":
        return lens.steering_hooks(model, 3, layers=[0], positions=positions)
    if name == "ablation":
        return lens.ablation_hooks(model, 3, layers=[0], positions=positions)
    assert name == "swap"
    return lens.swap_hooks(model, 3, 5, layers=[0], positions=positions)


def test_intervention_hooks_use_bridge_names_and_run(
    toy_model: _ToyBridge, fitted_lens: JacobianLens
) -> None:
    swap_hooks = fitted_lens.swap_hooks(toy_model, 3, 5, layers=[1, 2])
    steer_hooks = fitted_lens.steering_hooks(toy_model, 3, layers=[1], alpha=2.0)
    ablate_hooks = fitted_lens.ablation_hooks(toy_model, [3, 5], layers=[2])
    assert [name for name, _ in swap_hooks] == [
        "blocks.1.hook_out",
        "blocks.2.hook_out",
    ]
    tokens = toy_model.to_tokens("a toy prompt")
    baseline = toy_model(tokens)
    assert baseline is not None
    for hooks in (swap_hooks, steer_hooks, ablate_hooks):
        with toy_model.hooks(fwd_hooks=hooks):
            intervened = toy_model(tokens)
        assert intervened is not None
        assert intervened.shape == baseline.shape
        assert torch.isfinite(intervened).all()
        assert not torch.allclose(intervened, baseline)


@pytest.mark.parametrize("intervention", ["steering", "ablation", "swap"])
def test_intervention_negative_position_is_normalized(
    intervention: str, toy_model: _ToyBridge
) -> None:
    lens = _lens()
    tokens = toy_model.to_tokens("a toy prompt")
    _, baseline = toy_model.run_with_cache(tokens)
    hooks = _intervention_hooks(intervention, lens, toy_model, [-1])
    with toy_model.hooks(fwd_hooks=hooks):
        _, intervened = toy_model.run_with_cache(tokens)
    delta = intervened["blocks.0.hook_out"] - baseline["blocks.0.hook_out"]
    torch.testing.assert_close(delta[:, :-1], torch.zeros_like(delta[:, :-1]))
    assert delta[:, -1].abs().max() > 0


@pytest.mark.parametrize("intervention", ["steering", "ablation", "swap"])
def test_intervention_positions_are_checked_on_each_hook_call(
    intervention: str, toy_model: _ToyBridge
) -> None:
    lens = _lens()
    hooks = _intervention_hooks(intervention, lens, toy_model, [SEQ_LEN - 1])
    full_tokens = toy_model.to_tokens("a toy prompt")
    with toy_model.hooks(fwd_hooks=hooks):
        assert toy_model(full_tokens) is not None
    with pytest.raises(ValueError, match="out of range"):
        with toy_model.hooks(fwd_hooks=hooks):
            toy_model(full_tokens[:, :-1])


@pytest.mark.parametrize(
    ("intervention", "expected_transfers"),
    [("steering", 1), ("ablation", 1), ("swap", 2)],
)
def test_intervention_tensors_follow_each_activation_device(
    intervention: str,
    expected_transfers: int,
    toy_model: _ToyBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[torch.device] = []
    original = jacobian_lens_module._cached_on_device

    def record_device(
        tensor: torch.Tensor,
        cache: dict[torch.device, torch.Tensor],
        device: str | torch.device,
    ) -> torch.Tensor:
        calls.append(torch.device(device))
        return original(tensor, cache, device)

    monkeypatch.setattr(jacobian_lens_module, "_cached_on_device", record_device)
    hook = _intervention_hooks(intervention, _lens(), toy_model, [-1])[0][1]
    activation = torch.randn(1, SEQ_LEN, D_MODEL)

    output = hook(activation, None)

    assert output.device == activation.device
    assert calls == [activation.device] * expected_transfers


def test_swap_rejects_identical_tokens(toy_model: _ToyBridge) -> None:
    with pytest.raises(ValueError, match="same token|identical|distinct"):
        _lens().swap_hooks(toy_model, 3, 3, layers=[0])


def test_swap_warns_for_near_parallel_vectors() -> None:
    model = _ToyBridge()
    with torch.no_grad():
        source = model.unembed.weight[3]
        noise = torch.randn_like(source)
        noise -= noise.dot(source) / source.square().sum() * source
        # cosine ~= 0.99875: near enough to warn, below the hard rejection
        # threshold reserved for effectively rank-one bases.
        noise *= 0.05 * source.norm() / noise.norm()
        model.unembed.weight[5].copy_(source + noise)
    with pytest.warns(UserWarning, match="parallel|ill-conditioned|poorly conditioned"):
        hooks = _lens().swap_hooks(model, 3, 5, layers=[0])
    assert hooks[0][0] == "blocks.0.hook_out"


def test_swap_rejects_effectively_rank_one_vectors() -> None:
    model = _ToyBridge()
    with torch.no_grad():
        model.unembed.weight[5].copy_(2 * model.unembed.weight[3])

    with pytest.raises(ValueError, match="near-parallel"):
        _lens().swap_hooks(model, 3, 5, layers=[0])


def test_swap_leaves_orthogonal_complement_unchanged(
    toy_model: _ToyBridge, fitted_lens: JacobianLens
) -> None:
    layer = 2
    vectors = fitted_lens.lens_vectors(toy_model, [3, 5], layer)
    hooks = fitted_lens.swap_hooks(toy_model, 3, 5, layers=[layer])
    tokens = toy_model.to_tokens("a toy prompt")
    _, base_cache = toy_model.run_with_cache(tokens)
    with toy_model.hooks(fwd_hooks=hooks):
        _, swap_cache = toy_model.run_with_cache(tokens)
    name = f"blocks.{layer}.hook_out"
    delta = (swap_cache[name] - base_cache[name]).reshape(-1, D_MODEL)
    basis, _ = torch.linalg.qr(vectors.T)
    orthogonal_part = delta - (delta @ basis) @ basis.T
    assert orthogonal_part.abs().max().item() < 1e-4


def test_exports() -> None:
    from transformer_lens.tools import analysis

    assert analysis.JacobianLens is JacobianLens
    assert hasattr(analysis, "JacobianLensReadout")


# ---------------------------------------------------------------------------
# Registry / from_pretrained short-name resolution tests
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the bundled jacobian_lens_registry.json and its resolution helpers."""

    def test_registry_is_valid_json_and_nonempty(self) -> None:
        import json
        import pathlib

        path = (
            pathlib.Path(__file__).parents[3]
            / "transformer_lens"
            / "tools"
            / "analysis"
            / "jacobian_lens_registry.json"
        )
        assert path.is_file(), f"Registry not found at {path}"
        with path.open() as fh:
            registry = json.load(fh)
        model_entries = {k: v for k, v in registry.items() if not k.startswith("_")}
        assert len(model_entries) >= 30, "Expected at least 30 model entries"

    def test_registry_entries_have_required_fields(self) -> None:
        import json
        import pathlib

        path = (
            pathlib.Path(__file__).parents[3]
            / "transformer_lens"
            / "tools"
            / "analysis"
            / "jacobian_lens_registry.json"
        )
        with path.open() as fh:
            registry = json.load(fh)
        for key, entry in registry.items():
            if key.startswith("_"):
                continue
            assert "repo_id" in entry, f"Missing repo_id in entry '{key}'"
            assert "filename" in entry, f"Missing filename in entry '{key}'"
            assert entry["filename"].endswith(
                ".pt"
            ), f"filename in '{key}' does not end with .pt: {entry['filename']}"
            assert "aliases" in entry, f"Missing aliases list in entry '{key}'"

    def test_resolve_registry_entry_by_short_name(self) -> None:
        from transformer_lens.tools.analysis.jacobian_lens import (
            _resolve_registry_entry,
        )

        result = _resolve_registry_entry("gemma-2-2b")
        assert result is not None
        repo_id, filename = result
        assert repo_id == "neuronpedia/jacobian-lens"
        assert "gemma-2-2b" in filename
        assert filename.endswith(".pt")

    def test_resolve_registry_entry_by_hf_model_id(self) -> None:
        from transformer_lens.tools.analysis.jacobian_lens import (
            _resolve_registry_entry,
        )

        result = _resolve_registry_entry("google/gemma-2-2b")
        assert result is not None
        repo_id, filename = result
        assert repo_id == "neuronpedia/jacobian-lens"
        assert "gemma-2-2b" in filename

    def test_resolve_registry_entry_unknown_returns_none(self) -> None:
        from transformer_lens.tools.analysis.jacobian_lens import (
            _resolve_registry_entry,
        )

        assert _resolve_registry_entry("not-a-real-model") is None
        assert _resolve_registry_entry("some/unknown-repo") is None

    def test_from_pretrained_resolves_short_name_via_registry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Short model name resolves to the registry's repo_id and filename."""
        import huggingface_hub

        path = tmp_path / "lens.pt"
        _lens().save(str(path))
        calls: dict[str, Any] = {}

        def fake_download(**kwargs: Any) -> str:
            raise AssertionError("Must use retry wrapper")

        def fake_retry(function: Any, **kwargs: Any) -> str:
            calls["kwargs"] = kwargs
            return str(path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        monkeypatch.setattr(jacobian_lens_module, "call_hf_with_retry", fake_retry)

        JacobianLens.from_pretrained("gemma-2-2b")

        assert calls["kwargs"]["repo_id"] == "neuronpedia/jacobian-lens"
        assert "gemma-2-2b" in calls["kwargs"]["filename"]
        assert calls["kwargs"]["filename"].endswith(".pt")

    def test_from_pretrained_resolves_hf_model_id_via_registry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """HF model ID alias resolves to the same artifact as the short name."""
        import huggingface_hub

        path = tmp_path / "lens.pt"
        _lens().save(str(path))

        short_calls: dict[str, Any] = {}
        alias_calls: dict[str, Any] = {}

        def make_fake_retry(store: dict[str, Any]):
            def fake_retry(function: Any, **kwargs: Any) -> str:
                store["kwargs"] = kwargs
                return str(path)

            return fake_retry

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: None)

        monkeypatch.setattr(
            jacobian_lens_module, "call_hf_with_retry", make_fake_retry(short_calls)
        )
        JacobianLens.from_pretrained("gemma-2-2b")

        monkeypatch.setattr(
            jacobian_lens_module, "call_hf_with_retry", make_fake_retry(alias_calls)
        )
        JacobianLens.from_pretrained("google/gemma-2-2b")

        assert short_calls["kwargs"]["filename"] == alias_calls["kwargs"]["filename"]
        assert short_calls["kwargs"]["repo_id"] == alias_calls["kwargs"]["repo_id"]

    def test_from_pretrained_unknown_name_falls_through_to_hub(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """An unrecognised name is forwarded to Hub as the repo_id (backward compat)."""
        import huggingface_hub

        path = tmp_path / "lens.pt"
        _lens().save(str(path))
        calls: dict[str, Any] = {}

        def fake_retry(function: Any, **kwargs: Any) -> str:
            calls["kwargs"] = kwargs
            return str(path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: None)
        monkeypatch.setattr(jacobian_lens_module, "call_hf_with_retry", fake_retry)

        custom_filename = "custom/path/lens.pt"
        JacobianLens.from_pretrained("my-org/my-repo", filename=custom_filename)

        assert calls["kwargs"]["repo_id"] == "my-org/my-repo"
        assert calls["kwargs"]["filename"] == custom_filename

    def test_from_pretrained_registry_ignores_explicit_filename_kwarg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """When the registry resolves a name, the registry filename wins over the kwarg."""
        import huggingface_hub

        path = tmp_path / "lens.pt"
        _lens().save(str(path))
        calls: dict[str, Any] = {}

        def fake_retry(function: Any, **kwargs: Any) -> str:
            calls["kwargs"] = kwargs
            return str(path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: None)
        monkeypatch.setattr(jacobian_lens_module, "call_hf_with_retry", fake_retry)

        JacobianLens.from_pretrained("gemma-2-2b", filename="should-be-ignored.pt")

        assert calls["kwargs"]["filename"] != "should-be-ignored.pt"
        assert "gemma-2-2b" in calls["kwargs"]["filename"]

    def test_gpt2_small_alias_resolves(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Both 'gpt2' and 'openai-community/gpt2' are aliases for gpt2-small."""
        import huggingface_hub

        path = tmp_path / "lens.pt"
        _lens().save(str(path))

        filenames: list[str] = []

        def fake_retry(function: Any, **kwargs: Any) -> str:
            filenames.append(kwargs["filename"])
            return str(path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: None)
        monkeypatch.setattr(jacobian_lens_module, "call_hf_with_retry", fake_retry)

        for name in ("gpt2-small", "gpt2", "openai-community/gpt2"):
            JacobianLens.from_pretrained(name)

        assert len(set(filenames)) == 1, "All three aliases should resolve to the same filename"
        assert filenames[0].endswith("gpt2_jacobian_lens.pt")
