"""Unit tests for the Tier-2 thin-subclass adapters.

Each wraps an architecture whose module tree matches an already-verified
parent: Youtu = dense-MLA DeepSeek-V2, Jais2 = Nemotron (plain LayerNorm,
so folding stays enabled), Ministral3 = Mistral (plus a llama-4 positional
query scale the attention bridge applies from rope_parameters), VaultGemma
= Gemma 2 minus the post-norms, MusicFlamingo = Audio Flamingo 3 plus a
delegated rotary temporal embedding.
"""
from typing import Any

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekV2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gemma2 import (
    Gemma2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.jais2 import (
    Jais2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.ministral3 import (
    Ministral3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.music_flamingo import (
    MusicFlamingoArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.nemotron import (
    NemotronArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.vaultgemma import (
    VaultGemmaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.youtu import (
    YoutuArchitectureAdapter,
)


def _cfg(arch: str, **overrides: Any):
    overrides.setdefault("n_key_value_heads", 4)
    return make_bridge_cfg(arch, **overrides)


def test_youtu_is_dense_mla_deepseek_v2():
    adapter = YoutuArchitectureAdapter(_cfg("YoutuForCausalLM"))
    assert isinstance(adapter, DeepSeekV2ArchitectureAdapter)
    attn = adapter.component_mapping["blocks"].submodules["attn"]
    assert attn.submodules["kv_a_proj_with_mqa"].name == "kv_a_proj_with_mqa"


def test_jais2_is_nemotron_shaped():
    adapter = Jais2ArchitectureAdapter(_cfg("Jais2ForCausalLM"))
    assert isinstance(adapter, NemotronArchitectureAdapter)
    mlp = adapter.component_mapping["blocks"].submodules["mlp"]
    assert mlp.submodules["in"].name == "up_proj"  # ungated


def test_ministral3_is_mistral_shaped():
    adapter = Ministral3ArchitectureAdapter(_cfg("Ministral3ForCausalLM"))
    assert isinstance(adapter, MistralArchitectureAdapter)


def test_vaultgemma_drops_post_norms():
    adapter = VaultGemmaArchitectureAdapter(_cfg("VaultGemmaForCausalLM"))
    assert isinstance(adapter, Gemma2ArchitectureAdapter)
    blocks = adapter.component_mapping["blocks"]
    assert "ln1_post" not in blocks.submodules
    assert "ln2_post" not in blocks.submodules
    assert blocks.submodules["ln2"].name == "pre_feedforward_layernorm"


def test_factory_registrations():
    assert SUPPORTED_ARCHITECTURES["YoutuForCausalLM"] is YoutuArchitectureAdapter
    assert SUPPORTED_ARCHITECTURES["Jais2ForCausalLM"] is Jais2ArchitectureAdapter
    assert SUPPORTED_ARCHITECTURES["Ministral3ForCausalLM"] is Ministral3ArchitectureAdapter
    assert SUPPORTED_ARCHITECTURES["VaultGemmaForCausalLM"] is VaultGemmaArchitectureAdapter
    assert (
        SUPPORTED_ARCHITECTURES["MusicFlamingoForConditionalGeneration"]
        is MusicFlamingoArchitectureAdapter
    )


# Internal-only adapters (never on the Hub; the class-level counterpart of
# INTENTIONAL_EXCLUDES group 1 in tests/unit/tools/test_model_registry.py) are
# exempt from the per-adapter unit-coverage requirement. Keyed by adapter
# module basename so this file's own source never contains the class names the
# scanner looks for.
_COVERAGE_EXEMPT_MODULES = {"mingpt", "nanogpt", "native", "neel_solu_old"}


def _adapter_coverage() -> list[tuple[str, str, bool]]:
    """(class name, module basename, referenced-by-any-test-file) for every
    registered adapter class."""
    from pathlib import Path

    sources = [p.read_text() for p in Path(__file__).parent.glob("test_*.py")]
    rows = []
    for cls in sorted(set(SUPPORTED_ARCHITECTURES.values()), key=lambda c: c.__name__):
        module = cls.__module__.rsplit(".", 1)[-1]
        rows.append((cls.__name__, module, any(cls.__name__ in s for s in sources)))
    return rows


def test_every_registered_adapter_has_unit_coverage():
    """Derived from the factory so a new adapter — thin subclass or not — cannot
    ship without a unit test referencing it (the silent gap this file's hand
    enumeration used to allow)."""
    missing = [
        name
        for name, module, covered in _adapter_coverage()
        if not covered and module not in _COVERAGE_EXEMPT_MODULES
    ]
    assert not missing, (
        f"Registered adapters with no unit test coverage: {missing}. Add a "
        "test file (copy a sibling) or a bespoke test in this file."
    )


def test_coverage_exemptions_are_still_uncovered():
    """Self-cleaning: once an exempt adapter gains coverage, drop the exemption."""
    stale = [
        module
        for name, module, covered in _adapter_coverage()
        if covered and module in _COVERAGE_EXEMPT_MODULES
    ]
    assert (
        not stale
    ), f"Exempt adapter modules now have coverage; remove from _COVERAGE_EXEMPT_MODULES: {stale}"
