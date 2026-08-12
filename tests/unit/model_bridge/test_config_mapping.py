"""Unit tests for map_default_transformer_lens_config field precedence.

Style mirrors tests/unit/model_bridge/test_output_logits_contract.py: plain
SimpleNamespace HF-config stand-ins, no network access.
"""

from types import SimpleNamespace

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.sources.transformers import (
    map_default_transformer_lens_config,
)
from transformer_lens.utilities.heterogeneous_config import majority_value


def _hf_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "intermediate_size": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_explicit_head_dim_wins_over_derived() -> None:
    """head_dim need not equal hidden_size // num_attention_heads (16 // 4 = 4)."""
    mapped = map_default_transformer_lens_config(_hf_config(head_dim=8))
    assert mapped.d_head == 8


def test_d_head_falls_back_to_derived_when_head_dim_absent() -> None:
    mapped = map_default_transformer_lens_config(_hf_config())
    assert mapped.d_head == 16 // 4


def test_d_head_falls_back_to_derived_when_head_dim_is_none() -> None:
    """HF configs may carry head_dim=None; treat it the same as absent."""
    mapped = map_default_transformer_lens_config(_hf_config(head_dim=None))
    assert mapped.d_head == 16 // 4


class _AmbiguousAccessError(Exception):
    """Stand-in for transformers>=5.15 AmbiguousGlobalPerLayerAttributeError.

    Deliberately not an AttributeError: neither hasattr() nor getattr(..., default)
    suppresses the real exception, which is the crash mode of issue #1647.
    """


class _HeterogeneousConfig:
    """Minimal transformers>=5.15 heterogeneous-config contract.

    Exposes ``is_heterogeneous`` / ``per_layer_attributes`` / ``per_layer_config``
    and raises a non-AttributeError on any global read of a registered per-layer
    attribute, proving the mapping never performs the forbidden access.
    """

    def __init__(self, per_layer: dict, **global_fields: object) -> None:
        object.__setattr__(self, "_per_layer", per_layer)
        for key, value in global_fields.items():
            object.__setattr__(self, key, value)

    def __getattribute__(self, name: str):
        # __dict__ lookup keeps deepcopy working: it reconstructs instances
        # attribute-by-attribute, probing before _per_layer exists.
        per_layer = object.__getattribute__(self, "__dict__").get("_per_layer", {})
        if name in per_layer:
            raise _AmbiguousAccessError(f"'{name}' is a per-layer attribute")
        return object.__getattribute__(self, name)

    @property
    def is_heterogeneous(self) -> bool:
        return True

    @property
    def per_layer_attributes(self) -> set:
        return set(self._per_layer)

    @property
    def per_layer_config(self) -> list:
        return [
            SimpleNamespace(**{key: values[i] for key, values in self._per_layer.items()})
            for i in range(self.num_hidden_layers)
        ]


_GEMMA4_GLOBALS: dict[str, object] = {
    "hidden_size": 16,
    "num_hidden_layers": 6,
    "vocab_size": 32,
    "max_position_embeddings": 64,
    "intermediate_size": 32,
    "sliding_window": 8,
}


def test_heterogeneous_head_dim_does_not_read_global() -> None:
    """Gemma-4-E2B shape: per-layer head_dim (sliding 256 / full-attention 512)."""
    config = _HeterogeneousConfig(
        per_layer={"head_dim": [256, 256, 256, 256, 256, 512]},
        num_attention_heads=8,
        num_key_value_heads=1,
        **_GEMMA4_GLOBALS,
    )
    mapped = map_default_transformer_lens_config(config)
    assert mapped.d_head == 256  # majority-layer value, matching pre-5.15 behavior
    assert mapped.per_layer_head_dim == [256, 256, 256, 256, 256, 512]
    assert mapped.n_key_value_heads == 1  # uniform, still read globally


def test_heterogeneous_kv_heads_does_not_read_global() -> None:
    """Gemma-4-31B shape: head_dim and num_key_value_heads both vary per layer."""
    config = _HeterogeneousConfig(
        per_layer={
            "head_dim": [256, 256, 256, 256, 256, 512],
            "num_key_value_heads": [16, 16, 16, 16, 16, 4],
        },
        num_attention_heads=32,
        **_GEMMA4_GLOBALS,
    )
    mapped = map_default_transformer_lens_config(config)
    assert mapped.d_head == 256
    assert mapped.n_key_value_heads == 16
    assert mapped.per_layer_head_dim == [256, 256, 256, 256, 256, 512]
    assert mapped.per_layer_num_key_value_heads == [16, 16, 16, 16, 16, 4]


def test_legacy_global_fields_reconstruct_per_layer_geometry() -> None:
    """Pre-5.15 Gemma 4: <field> holds the sliding value, global_<field> the
    full-attention value; the per-layer view is rebuilt from layer_types."""
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    mapped = map_default_transformer_lens_config(
        _hf_config(
            num_attention_heads=32,
            head_dim=256,
            global_head_dim=512,
            num_key_value_heads=16,
            num_global_key_value_heads=4,
            layer_types=layer_types,
            num_hidden_layers=6,
        )
    )
    assert mapped.d_head == 256  # unchanged from the pre-fix scalar
    assert mapped.n_key_value_heads == 16
    assert mapped.per_layer_head_dim == [256] * 5 + [512]
    assert mapped.per_layer_num_key_value_heads == [16] * 5 + [4]


def test_legacy_kv_reconstruction_gated_on_attention_k_eq_v() -> None:
    """HF applies num_global_key_value_heads only when attention_k_eq_v is set;
    with it False, every layer runs the base KV-head count."""
    mapped = map_default_transformer_lens_config(
        _hf_config(
            num_attention_heads=32,
            head_dim=256,
            global_head_dim=512,
            num_key_value_heads=16,
            num_global_key_value_heads=4,
            attention_k_eq_v=False,
            layer_types=["sliding_attention"] * 5 + ["full_attention"],
            num_hidden_layers=6,
        )
    )
    assert not hasattr(mapped, "per_layer_num_key_value_heads")
    assert mapped.n_key_value_heads == 16
    # head_dim reconstruction is ungated — global_head_dim applies regardless.
    assert mapped.per_layer_head_dim == [256] * 5 + [512]


def test_unanticipated_per_layer_field_resolves_to_majority() -> None:
    """Any registered per-layer field — not just the ones Gemma 4 registers today —
    must resolve to its majority value instead of crashing the probe."""
    config = _HeterogeneousConfig(
        per_layer={"sliding_window": [8, 8, 16, 8]},
        num_attention_heads=4,
        hidden_size=16,
        num_hidden_layers=4,
        vocab_size=32,
        max_position_embeddings=64,
        intermediate_size=32,
    )
    mapped = map_default_transformer_lens_config(config)
    # Read via __dict__: boot consumes the mapped config that way, and attribute
    # access on the (copied) heterogeneous config itself still raises.
    assert mapped.__dict__["sliding_window"] == 8


def test_legacy_global_field_equal_to_base_stays_scalar() -> None:
    """No per-layer view when the global_* value matches the base field."""
    mapped = map_default_transformer_lens_config(
        _hf_config(
            head_dim=8,
            global_head_dim=8,
            layer_types=["sliding_attention", "full_attention"],
        )
    )
    assert mapped.d_head == 8
    assert not hasattr(mapped, "per_layer_head_dim")


def test_homogeneous_config_gets_no_per_layer_fields() -> None:
    mapped = map_default_transformer_lens_config(_hf_config(head_dim=8))
    assert not hasattr(mapped, "per_layer_head_dim")
    assert not hasattr(mapped, "per_layer_num_key_value_heads")


def test_majority_value_semantics() -> None:
    """The scalar collapse is the most-common value — not first, min, or max."""
    assert majority_value([512, 256, 256]) == 256  # majority beats first element
    assert majority_value([64, 32, 64]) == 64  # majority beats min
    assert majority_value([512, 256, 512, 256]) == 512  # tie breaks to earliest layer


def test_heterogeneous_config_through_bridge_config_build() -> None:
    """Full boot pipeline (map → from_dict → passthrough) on a heterogeneous config.

    intermediate_size is both per-layer here and in _HF_PASSTHROUGH_ATTRS, so the
    passthrough loop must skip it rather than perform the raising global read.
    """
    config = _HeterogeneousConfig(
        per_layer={
            "head_dim": [32, 32, 32, 64],
            "intermediate_size": [128, 64, 64, 64],
            "num_attention_heads": [8, 8, 8, 4],
        },
        num_key_value_heads=2,
        hidden_size=16,
        num_hidden_layers=4,
        vocab_size=32,
        max_position_embeddings=64,
    )
    bridge_config = build_bridge_config_from_hf(config, "TestArch", "test-model", torch.float32)
    assert bridge_config.n_heads == 8  # majority of per-layer num_attention_heads
    assert bridge_config.d_head == 32
    assert bridge_config.per_layer_head_dim == [32, 32, 32, 64]
    assert bridge_config.n_key_value_heads == 2  # uniform, still read globally
    assert bridge_config.d_mlp == 128  # per-layer intermediate_size collapses to max
    # The passthrough loop must skip the per-layer-registered attr, not copy it.
    assert not hasattr(bridge_config, "intermediate_size")


def test_bridge_config_retains_per_layer_fields() -> None:
    """from_dict filters to signature params; the per-layer fields must survive."""
    bridge_config = TransformerBridgeConfig.from_dict(
        {
            "d_model": 16,
            "d_head": 256,
            "n_layers": 6,
            "n_ctx": 64,
            "n_heads": 8,
            "per_layer_head_dim": [256] * 5 + [512],
            "per_layer_num_key_value_heads": [16] * 5 + [4],
        }
    )
    assert bridge_config.per_layer_head_dim == [256] * 5 + [512]
    assert bridge_config.per_layer_num_key_value_heads == [16] * 5 + [4]
