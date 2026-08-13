"""Guards against reading quantized weights as if they were plain matrices.

TransformerLens supports quantized *forward* passes — the wrapped HF module
dequantizes internally. What it cannot do is arithmetic on the stored weight:
reshaping it per head, slicing a fused projection, folding LayerNorm into it.
Those reads must fail loudly rather than return packed bytes or scale-less
narrow floats, which look like ordinary tensors and produce wrong numbers.

Everything here is built in memory: bitsandbytes is an optional extra and is not
installed in CI, so the quantized shapes are simulated.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from transformer_lens.utilities.quantization import (
    describe_quantization,
    require_readable_weight,
    unreadable_weight_reason,
)

# The shapes a quantized weight actually takes in this stack, with the reason
# fragment each should produce. float8 is the important one: it reports
# is_floating_point=True, so a plain float check admits it — and it is the only
# family that survives both slicing and nn.Parameter() without complaint.
UNREADABLE = [
    pytest.param(torch.zeros(4, 4, dtype=torch.int8), "packed integer storage", id="bnb-int8"),
    pytest.param(torch.zeros(8, 1, dtype=torch.uint8), "packed integer storage", id="packed-uint8"),
    pytest.param(torch.zeros(4, 4, dtype=torch.int32), "packed integer storage", id="gptq-int32"),
    pytest.param(torch.zeros(4, 4, dtype=torch.float8_e4m3fn), "narrow float", id="fp8-e4m3fn"),
    pytest.param(torch.zeros(4, 4, dtype=torch.float8_e5m2), "narrow float", id="fp8-e5m2"),
]

# Legitimate weights, including the dtypes a quantized *forward* pass uses for
# its compute. A guard that fires on any of these is a regression.
READABLE = [
    pytest.param(torch.zeros(4, 4, dtype=torch.float32), id="fp32"),
    pytest.param(torch.zeros(4, 4, dtype=torch.bfloat16), id="bf16"),
    pytest.param(torch.zeros(4, 4, dtype=torch.float16), id="fp16"),
    pytest.param(torch.zeros(4, 4, dtype=torch.float64), id="fp64"),
]


class _PackedParams(torch.Tensor):
    """Stands in for bitsandbytes Params4bit: a Tensor SUBCLASS holding packed
    bytes, so isinstance(w, Tensor) is True and only the dtype gives it away."""


class _TritonWrapper:
    """Stands in for the MXFP4 triton-kernels object — named Tensor upstream,
    which is what makes the resulting TypeError look like a torch bug."""

    dtype = "packed"


@pytest.mark.parametrize("weight,reason", UNREADABLE)
def test_quantized_storage_is_rejected(weight, reason) -> None:
    assert reason in (unreadable_weight_reason(weight) or "")
    with pytest.raises(NotImplementedError, match=reason):
        require_readable_weight(weight, operation="read W_Q")


@pytest.mark.parametrize("weight", READABLE)
def test_real_float_weights_pass(weight) -> None:
    """The positive control: these are what a working model (including a
    quantized forward pass's compute dtype) presents."""
    assert unreadable_weight_reason(weight) is None
    assert require_readable_weight(weight, operation="read W_Q") is weight


def test_tensor_subclass_with_packed_storage_is_rejected() -> None:
    """bitsandbytes params pass isinstance(w, torch.Tensor), so the dtype branch
    is the only thing standing between them and a silent reshape."""
    packed = _PackedParams(torch.zeros(8, 1, dtype=torch.uint8))
    assert isinstance(packed, torch.Tensor)
    with pytest.raises(NotImplementedError, match="packed integer storage"):
        require_readable_weight(packed, operation="read W_Q")


def test_non_tensor_wrapper_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="not a torch.Tensor"):
        require_readable_weight(_TritonWrapper(), operation="read W_Q")


def test_meta_device_is_reported_as_a_load_problem_not_quantization() -> None:
    """A meta tensor is float and full-width, so it passes the dtype checks; it
    must be named as an unmaterialized load rather than sent after a
    quantization that is not there."""
    with pytest.raises(NotImplementedError) as excinfo:
        require_readable_weight(torch.zeros(4, 4, device="meta"), operation="read W_Q")
    assert "meta device" in str(excinfo.value)
    assert "quantization:" not in str(excinfo.value)


def test_error_names_the_quantization_method() -> None:
    class _Owner:
        class config:
            class quantization_config:
                quant_method = "bitsandbytes"

    with pytest.raises(NotImplementedError, match="bitsandbytes"):
        require_readable_weight(
            torch.zeros(4, 4, dtype=torch.int8), operation="read W_Q", owner=_Owner()
        )
    assert describe_quantization(_Owner()) == "bitsandbytes"


def test_quantization_method_falls_back_when_unknowable() -> None:
    assert describe_quantization(nn.Linear(2, 2)) == "an unknown quantization"


class TestBridgeAccessorsAreGuarded:
    """The accessors are the most dangerous site: they returned packed bytes
    with no error, so downstream analyses decomposed garbage."""

    @staticmethod
    def _bridge():
        from tests.unit.model_bridge.supported_architectures.helpers import (
            make_bridge_cfg,
        )
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )
        from transformer_lens.model_bridge.component_setup import setup_submodules

        cfg = make_bridge_cfg("LlamaForCausalLM", d_model=8, n_heads=2, d_head=4)
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        return adapter, setup_submodules

    @pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.float8_e4m3fn])
    def test_attention_accessor_refuses_quantized_weight(self, dtype) -> None:
        import copy

        adapter, setup = self._bridge()
        attn_template = adapter.component_mapping["blocks"].submodules["attn"]

        class _QuantAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    linear = nn.Linear(8, 8, bias=False)
                    linear.weight = nn.Parameter(
                        torch.zeros(8, 8, dtype=dtype), requires_grad=False
                    )
                    setattr(self, name, linear)

        module = _QuantAttn()
        bridge = copy.deepcopy(attn_template)
        bridge.set_original_component(module)
        setup(bridge, adapter, module)

        with pytest.raises(NotImplementedError, match="W_Q"):
            _ = bridge.W_Q
