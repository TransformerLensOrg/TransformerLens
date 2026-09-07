"""Tests for multi-GPU utilities."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from transformer_lens.utilities import (
    calculate_available_device_cuda_memory,
    determine_available_memory_for_available_devices,
    sort_devices_based_on_available_memory,
)
from transformer_lens.utilities.multi_gpu import (
    cast_floating_params_to_dtype,
    get_device_for_block_index,
)


def mock_available_devices(memory_stats: list[tuple[int, int]]):
    torch.cuda.device_count = Mock(return_value=len(memory_stats))

    def device_props_return(*args, **kwargs):
        total_memory = memory_stats[args[0]][0]
        device_props = Mock()
        device_props.total_memory = total_memory
        return device_props

    def memory_allocated_return(*args, **kwargs):
        return memory_stats[args[0]][1]

    torch.cuda.get_device_properties = Mock(side_effect=device_props_return)
    torch.cuda.memory_allocated = Mock(side_effect=memory_allocated_return)


def test_calculate_available_device_cuda_memory():
    mock_available_devices([(80, 40)])

    result = calculate_available_device_cuda_memory(0)
    assert result == 40


def test_determine_available_memory_for_available_devices():
    mock_available_devices(
        [
            (80, 60),
            (80, 15),
            (80, 40),
        ]
    )

    result = determine_available_memory_for_available_devices(3)

    assert result == [
        (0, 20),
        (1, 65),
        (2, 40),
    ]


def test_sort_devices_based_on_available_memory():
    devices = [
        (0, 20),
        (1, 65),
        (2, 40),
    ]

    result = sort_devices_based_on_available_memory(devices)

    assert result == [
        (1, 65),
        (2, 40),
        (0, 20),
    ]


def _cuda_cfg(n_layers: int, n_devices: int) -> SimpleNamespace:
    return SimpleNamespace(n_layers=n_layers, n_devices=n_devices, device="cuda")


class TestGetDeviceForBlockIndex:
    """Regression tests for the layer-to-device index math.

    Issue #1356: the previous formula ``index // (n_layers // n_devices)``
    overshot ``n_devices - 1`` whenever ``n_layers`` was not a multiple of
    ``n_devices``, and divided by zero when ``n_layers < n_devices``.
    """

    @pytest.mark.parametrize(
        "n_layers,n_devices",
        [(62, 8), (12, 8), (32, 4), (24, 8), (8, 8), (1, 8), (7, 8)],
    )
    def test_device_index_stays_in_bounds(self, n_layers: int, n_devices: int):
        cfg = _cuda_cfg(n_layers, n_devices)
        for index in range(n_layers):
            result = get_device_for_block_index(index, cfg)
            assert 0 <= result.index < n_devices, (
                f"index={index} mapped to device {result.index} which is outside "
                f"[0, {n_devices - 1}] for n_layers={n_layers}, n_devices={n_devices}"
            )

    @pytest.mark.parametrize("n_layers,n_devices", [(62, 8), (32, 4), (24, 8), (8, 8)])
    def test_layer_distribution_is_balanced(self, n_layers: int, n_devices: int):
        """Every device sees ``floor(n_layers / n_devices)`` or that plus 1 layers
        — never more than 1 layer off, and the counts sum to ``n_layers``."""
        cfg = _cuda_cfg(n_layers, n_devices)
        counts = [0] * n_devices
        for index in range(n_layers):
            counts[get_device_for_block_index(index, cfg).index] += 1
        assert sum(counts) == n_layers
        assert max(counts) - min(counts) <= 1, f"unbalanced distribution: {counts}"

    def test_first_index_lands_on_first_device(self):
        cfg = _cuda_cfg(n_layers=62, n_devices=8)
        result = get_device_for_block_index(0, cfg)
        assert result.index == 0

    def test_last_index_lands_on_last_device(self):
        cfg = _cuda_cfg(n_layers=62, n_devices=8)
        result = get_device_for_block_index(61, cfg)
        assert result.index == 7

    def test_starting_device_offset_is_honored(self):
        """When ``device`` carries an explicit index, layer offsets are added on top."""
        cfg = _cuda_cfg(n_layers=32, n_devices=4)
        result = get_device_for_block_index(0, cfg, device=torch.device("cuda", 2))
        assert result.index == 2
        result = get_device_for_block_index(31, cfg, device=torch.device("cuda", 2))
        assert result.index == 5  # 2 (starting offset) + 3 (last layer on 4 devices)

    def test_cpu_device_is_returned_unchanged(self):
        cfg = _cuda_cfg(n_layers=62, n_devices=8)
        result = get_device_for_block_index(30, cfg, device="cpu")
        assert result.type == "cpu"


def _fp8_linear(scale_fmt: str) -> nn.Module:
    """A real transformers finegrained-FP8 ``Linear``, or skip if the integration moved."""
    integration = pytest.importorskip(
        "transformers.integrations.finegrained_fp8",
        reason="requires transformers' finegrained-FP8 integration",
    )
    fp8_linear = getattr(integration, "FP8Linear", None)
    if fp8_linear is None:
        pytest.skip("transformers.integrations.finegrained_fp8.FP8Linear is unavailable")
    return fp8_linear(
        in_features=128,
        out_features=128,
        block_size=(128, 128),
        activation_scheme="static",
        scale_fmt=scale_fmt,
        has_bias=True,
    )


# Quantizer-owned storage on FP8Linear: the packed weight and its scales. ``bias`` is
# excluded on purpose -- HF keeps biases in the model's compute dtype, so a bias is an
# ordinary parameter that the quantizer does not own.
_FP8_OWNED_PARAMS = ("weight", "weight_scale_inv", "activation_scale")


class TestCastFloatingParamsToDtype:
    """Regression tests for cast_floating_params_to_dtype.

    See: https://github.com/TransformerLensOrg/TransformerLens/issues/1713
    The function was casting quantizer-owned FP8 scale tensors (float8_e8m0fnu)
    to bfloat16, which corrupts the weight/scale pair relationship and breaks
    MXFP4 checkpoints.
    """

    def test_casts_standard_floats_to_target_dtype(self):
        """Positive control: standard float dtypes should be cast."""
        model = nn.Linear(4, 4)
        model.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
        cast_floating_params_to_dtype(model, torch.bfloat16)
        assert model.weight.dtype == torch.bfloat16

    def test_skips_params_already_at_target_dtype(self):
        """Params already at target dtype are left untouched."""
        model = nn.Linear(4, 4)
        original = torch.zeros(4, 4, dtype=torch.bfloat16)
        model.weight = nn.Parameter(original)
        cast_floating_params_to_dtype(model, torch.bfloat16)
        assert model.weight.dtype == torch.bfloat16
        assert model.weight.data_ptr() == original.data_ptr()

    def test_skips_non_floating_point_params(self):
        """Integer params (packed quantized weights) are left untouched."""
        model = nn.Linear(4, 4, bias=False)
        model.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.int8), requires_grad=False)
        cast_floating_params_to_dtype(model, torch.bfloat16)
        assert model.weight.dtype == torch.int8

    @pytest.mark.parametrize(
        "fp8_dtype",
        [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            pytest.param(
                getattr(torch, "float8_e8m0fnu", None),
                marks=pytest.mark.skipif(
                    not hasattr(torch, "float8_e8m0fnu"), reason="torch < 2.7"
                ),
            ),
        ],
    )
    def test_skips_one_byte_floats_fp8_scales(self, fp8_dtype):
        """FP8 scale tensors must NOT be cast — they are quantizer-owned.

        This is the key regression test for the MXFP4 bug: casting float8_e8m0fnu
        scales to bfloat16 breaks the weight/scale pair relationship.
        """
        model = nn.Linear(4, 4, bias=False)
        model.weight = nn.Parameter(torch.zeros(4, 4, dtype=fp8_dtype), requires_grad=False)
        cast_floating_params_to_dtype(model, torch.bfloat16)
        assert model.weight.dtype == fp8_dtype

    def test_mixed_module_casts_selectively(self):
        """A module with both standard and FP8 params: only standard params cast."""

        class MixedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.standard_weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
                self.fp8_scale = nn.Parameter(
                    torch.zeros(4, 4, dtype=torch.float8_e4m3fn), requires_grad=False
                )
                self.packed_weight = nn.Parameter(
                    torch.zeros(4, 4, dtype=torch.int8), requires_grad=False
                )

        model = MixedModule()
        cast_floating_params_to_dtype(model, torch.bfloat16)

        assert model.standard_weight.dtype == torch.bfloat16
        assert model.fp8_scale.dtype == torch.float8_e4m3fn
        assert model.packed_weight.dtype == torch.int8

    @pytest.mark.parametrize("scale_fmt", ["ue8m0", "float"])
    def test_itemsize_guard_does_not_identify_quantizer_owned_scales(self, scale_fmt):
        """``itemsize < 2`` is a narrow-float guard, not an ownership test.

        Transformers picks ``weight_scale_inv``'s storage dtype from the checkpoint's
        ``scale_fmt``: a one-byte UE8M0 float for "ue8m0", float32 for "float" (the
        default). Both spell the same quantizer-owned scale, and ``activation_scale``
        is float32 under either format. So a one-byte test protects some quantizer-owned
        scales and silently rewrites others, which is why the cast has to be gated on
        the model having no active quantizer rather than on per-parameter dtype.

        See: https://github.com/TransformerLensOrg/TransformerLens/issues/1743
        """
        target = torch.bfloat16
        module = _fp8_linear(scale_fmt)
        before = {name: param.dtype for name, param in module.named_parameters()}
        cast_floating_params_to_dtype(module, target)
        after = {name: param.dtype for name, param in module.named_parameters()}

        owned = {name: before[name] for name in _FP8_OWNED_PARAMS if name in before}
        assert owned, "FP8Linear exposed none of its quantizer-owned parameters"
        # Mirror the cast's whole predicate, not dtype width alone: it rewrites a
        # parameter only when that parameter is floating, wider than one byte and not
        # already at the target. (Its fourth condition, meta, cannot apply to this
        # materialized fixture.) Deriving the set this way stays correct if
        # transformers moves `weight` to a wide packed-integer storage like GPTQ's
        # int32, which the cast is right to skip.
        eligible = {
            name
            for name, dtype in owned.items()
            if dtype.is_floating_point and dtype.itemsize >= 2 and dtype != target
        }
        assert eligible, "expected a quantizer-owned float wider than one byte"
        rewritten = {name for name, dtype in owned.items() if after[name] != dtype}
        assert rewritten == eligible, (
            "the cast should rewrite exactly those quantizer-owned parameters eligible "
            f"under its dtype-only predicate: rewritten={sorted(rewritten)} "
            f"eligible={sorted(eligible)}"
        )


class TestMaybeCastFloatingParams:
    """Tests for maybe_cast_floating_params helper.

    See: https://github.com/TransformerLensOrg/TransformerLens/issues/1713
    The helper wraps cast_floating_params_to_dtype with a quantization check,
    skipping the cast entirely when the model has an active quantization_config.
    """

    def test_casts_unquantized_model(self):
        """Unquantized models should have their params cast."""
        from types import SimpleNamespace

        from transformer_lens.utilities.multi_gpu import maybe_cast_floating_params

        model = nn.Linear(4, 4)
        model.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
        model.config = SimpleNamespace(quantization_config=None)

        maybe_cast_floating_params(model, torch.bfloat16)
        assert model.weight.dtype == torch.bfloat16

    def test_skips_quantized_model(self):
        """An active quantizer means HF owns the storage dtypes, so nothing is cast.

        The skip is whole-model deliberately. ``from_pretrained`` is responsible for
        applying the requested dtype to ordinary floating parameters before this helper
        runs, and TransformerLens must not second-guess quantizer-owned storage
        afterward. Any parameter still off the requested dtype here may be
        quantizer-owned, and dtype alone cannot distinguish ownership safely (see
        ``test_itemsize_guard_does_not_identify_quantizer_owned_scales``). transformers
        draws the same line itself: ``.to(dtype=...)`` raises for bitsandbytes and
        GPTQ models, ``.half()`` / ``.float()`` for any quantized model.

        See: https://github.com/TransformerLensOrg/TransformerLens/issues/1713
        See: https://github.com/TransformerLensOrg/TransformerLens/issues/1743
        """
        from transformer_lens.utilities.multi_gpu import maybe_cast_floating_params

        model = nn.Linear(4, 4)
        model.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
        model.config = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4"))

        maybe_cast_floating_params(model, torch.bfloat16)
        assert model.weight.dtype == torch.float32  # NOT cast

    @pytest.mark.parametrize("scale_fmt", ["ue8m0", "float"])
    def test_preserves_quantizer_owned_scales_of_any_width(self, scale_fmt):
        """Both widths of ``weight_scale_inv`` survive, which the dtype guard cannot do.

        The ordinary parameter is set up as the loader would have delivered it, at the
        requested dtype, so the correct outcome for the whole model is that nothing
        moves. Under ``scale_fmt="float"`` the scale is float32, so this fails outright
        if the cast is ever re-enabled behind only the one-byte-float guard.
        """
        from transformer_lens.utilities.multi_gpu import maybe_cast_floating_params

        model = nn.Module()
        model.quantized = _fp8_linear(scale_fmt)
        model.ln_weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        model.config = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="fp8"))

        before = {name: param.dtype for name, param in model.named_parameters()}
        maybe_cast_floating_params(model, torch.bfloat16)

        assert {name: param.dtype for name, param in model.named_parameters()} == before
        assert model.quantized.weight_scale_inv.dtype == before["quantized.weight_scale_inv"]

    def test_casts_once_hf_releases_quantizer_ownership(self):
        """The guard is a hand-off, not a permanent opt-out.

        ``HfQuantizer.postprocess_model`` calls ``remove_quantization_config`` when a
        checkpoint is loaded with ``dequantize=True``, which deletes
        ``config.quantization_config``. ``quantization_method`` then returns None and
        normalization resumes. That is what stops the whole-model guard from stranding a
        genuinely dequantized checkpoint in its load dtype.
        """
        from transformer_lens.utilities.multi_gpu import maybe_cast_floating_params

        model = nn.Linear(4, 4)
        model.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
        model.config = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4"))

        maybe_cast_floating_params(model, torch.bfloat16)
        assert model.weight.dtype == torch.float32

        del model.config.quantization_config  # what remove_quantization_config does
        maybe_cast_floating_params(model, torch.bfloat16)
        assert model.weight.dtype == torch.bfloat16

    def test_hf_still_clears_quantization_config_when_dequantizing(self):
        """Pins the upstream behaviour the whole-model guard is calibrated against.

        If transformers ever stops deleting ``quantization_config`` on a dequantized
        load, ``quantization_method`` would keep reporting a method for a model whose
        storage the quantizer no longer owns, and the guard really would be too broad.
        Fail here rather than silently stranding those checkpoints.
        """
        base = pytest.importorskip("transformers.quantizers.base")

        model = nn.Linear(4, 4)
        model.config = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4"))
        model.is_quantized = True
        # Called unbound: remove_quantization_config only touches `model`, and building
        # a real quantizer would need a live quantization config per method.
        base.HfQuantizer.remove_quantization_config(None, model)

        assert not hasattr(model.config, "quantization_config")
        assert model.is_quantized is False

    def test_skips_model_without_config(self):
        """Models without a config attribute should be cast (no quantization)."""
        from transformer_lens.utilities.multi_gpu import maybe_cast_floating_params

        model = nn.Linear(4, 4)
        model.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
        # No model.config attribute

        maybe_cast_floating_params(model, torch.bfloat16)
        assert model.weight.dtype == torch.bfloat16
