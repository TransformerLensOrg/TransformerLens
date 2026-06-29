"""Unit tests for Olmo3ArchitectureAdapter.

Olmo3 inherits its config, component mapping, and weight conversions unchanged
from Olmo2 (the class body is `pass`). These smoke tests pin that contract
behaviorally: the factory resolves the Olmo3 arch string to this adapter, and a
constructed Olmo3 adapter produces the same component surface as Olmo2 — so an
accidental override (or a dropped factory registration) is caught here. The
inherited surface itself is covered by test_olmo2_adapter.py.
"""

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.olmo3 import (
    Olmo3ArchitectureAdapter,
)


def _cfg(architecture: str) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=2,
        d_vocab=256,
        architecture=architecture,
    )


def test_factory_resolves_olmo3_arch_string() -> None:
    """The Olmo3 arch string must stay wired to this adapter, or loading breaks."""
    assert SUPPORTED_ARCHITECTURES["Olmo3ForCausalLM"] is Olmo3ArchitectureAdapter


def test_builds_same_component_surface_as_olmo2() -> None:
    """A constructed Olmo3 adapter yields Olmo2's component mapping (body is `pass`)."""
    olmo3 = Olmo3ArchitectureAdapter(_cfg("Olmo3ForCausalLM"))
    olmo2 = Olmo2ArchitectureAdapter(_cfg("Olmo2ForCausalLM"))
    assert isinstance(olmo3, Olmo2ArchitectureAdapter)
    assert set(olmo3.component_mapping) == set(olmo2.component_mapping)
