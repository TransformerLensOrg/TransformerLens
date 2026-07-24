"""A default verification run must be able to reach "verified".

Vision architectures gate on phase 7 and audio on phase 8, but the default
phase set was the text one for everybody. Every default run on a multimodal
model therefore wrote all-pass scores, found phase 7 missing, and left the
status at 0 — silently, while reporting the model as verified.
"""

import pytest

from transformer_lens.tools.model_registry.verify_models import (
    _default_phases_for_architecture,
    _full_and_core_phases,
)

# One representative per class the runner distinguishes.
ARCHITECTURES = [
    "GPT2LMHeadModel",
    "LlamaForCausalLM",
    "Mistral3ForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "HubertForCTC",
]


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_default_run_satisfies_core_verification(arch: str) -> None:
    """The invariant that was violated: defaults must cover the core set."""
    default = set(_default_phases_for_architecture(arch))
    _, core = _full_and_core_phases(arch)
    missing = core - default
    assert not missing, f"{arch}: default run omits core phase(s) {sorted(missing)}"


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_default_run_is_not_treated_as_partial(arch: str) -> None:
    """`is_partial_run` compares against the full set; a default run must
    equal it, or the status-writing branch never executes."""
    full, _ = _full_and_core_phases(arch)
    assert set(_default_phases_for_architecture(arch)) == full


def test_vision_and_audio_defaults_differ_from_text() -> None:
    """Guards the regression directly: were these to collapse back to the text
    set, both families would silently become unverifiable again."""
    assert 7 in _default_phases_for_architecture("Mistral3ForConditionalGeneration")
    assert 8 in _default_phases_for_architecture("HubertForCTC")
    assert _default_phases_for_architecture("GPT2LMHeadModel") == [1, 2, 3, 4]
