"""Unit tests for the BitNetArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import GatedMLPBridge
from transformer_lens.model_bridge.supported_architectures.bitnet import (
    BitNetArchitectureAdapter,
    _BitNetAttentionBridge,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "BitNetForCausalLM",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> BitNetArchitectureAdapter:
    return BitNetArchitectureAdapter(_make_cfg())


class TestBitNetComponentMapping:
    def test_attention_uses_sub_norm_bridge(self, adapter):
        """BitNet's distinguishing feature: RMSNorms before both output
        projections, applied via the adapter-local attention bridge and the
        delegated HF MLP."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert isinstance(submodules["attn"], _BitNetAttentionBridge)
        assert isinstance(submodules["mlp"], GatedMLPBridge)

    def test_sub_norms_disable_folding(self, adapter):
        assert adapter.supports_fold_ln is False

    def test_attention_reconstruction_applies_sub_norm(self, adapter):
        """The generic reconstruction skips attn_sub_norm; the subclass's
        pre-output-projection seam must apply it."""
        import torch

        attn = adapter.component_mapping["blocks"].submodules["attn"]

        class _FakeNorm(torch.nn.Module):
            def forward(self, x):
                return x * 2.0

        class _FakeAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_sub_norm = _FakeNorm()

        attn._modules["_original_component"] = _FakeAttn()
        x = torch.ones(1, 2, 4)
        out = attn._pre_output_projection(x)
        assert torch.equal(out, x * 2.0)


class TestBitNetPackedCheckpointGuard:
    """`prepare_model` must refuse packed 1.58-bit checkpoints.

    The flagship microsoft/bitnet-b1.58-2B-4T stores `weight` as packed uint8
    with a collapsed first dim plus a separate weight_scale, so every
    weight-space read reshapes it into a wrong-but-plausible matrix instead of
    failing — the registry records that checkpoint at 0% on the forward phase.
    """

    @staticmethod
    def _model(dtype, packed_shape=(8, 1)):
        class _Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4, bias=False)
                self.q_proj.weight = torch.nn.Parameter(
                    torch.zeros(*packed_shape, dtype=dtype), requires_grad=False
                )

        model = _Tiny()
        # prepare_model's base implementation reads cfg.attn_implementation.
        model.config = SimpleNamespace(attn_implementation="eager")
        return model

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
    def test_packed_weights_are_refused(self, adapter, dtype):
        with pytest.raises(NotImplementedError, match="packed weights"):
            adapter.prepare_model(self._model(dtype))

    def test_error_names_the_dequantized_sibling(self, adapter):
        with pytest.raises(NotImplementedError, match="bitnet-b1.58-2B-4T-bf16"):
            adapter.prepare_model(self._model(torch.uint8))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_dequantized_checkpoints_still_load(self, adapter, dtype):
        """The positive control: the bf16 sibling this error points users at
        must pass, or the guard would make BitNet unusable entirely."""
        adapter.prepare_model(self._model(dtype, packed_shape=(4, 4)))


class TestBitNetRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["BitNetForCausalLM"] is BitNetArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="bitnet", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "BitNetForCausalLM"
