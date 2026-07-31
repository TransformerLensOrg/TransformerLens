"""Vision encoders must route to the Phase-1-only verification set.

Vision architectures have no tokenizer and no HookedTransformer counterpart, so the
default text phase set ({1,2,3,4}) cannot run against them. They previously opted out
of verification entirely via ``applicable_phases = []``, which made them unverifiable
rather than verified — Phase 1 (HF parity on pixel input) does apply.
"""

import pytest

from transformer_lens.tools.model_registry.verify_models import (
    _full_and_core_phases,
    _phases_to_run,
)
from transformer_lens.utilities.architectures import (
    NO_HT_COMPARISON_ARCHITECTURES,
    classify_architecture,
)

VISION_ARCHS = [
    "ViTForImageClassification",
    "ViTModel",
    "DeiTForImageClassification",
]


@pytest.mark.parametrize("architecture", VISION_ARCHS)
def test_vision_architectures_classify_as_vision(architecture: str) -> None:
    assert classify_architecture(architecture) == "vision"


@pytest.mark.parametrize("architecture", VISION_ARCHS)
def test_vision_runs_phase_one_only(architecture: str) -> None:
    """Phase 1 is the whole verification story for a vision encoder."""
    full, core = _full_and_core_phases(architecture)
    assert full == {1}
    assert core == {1}


@pytest.mark.parametrize("architecture", VISION_ARCHS)
def test_vision_phase_one_survives_adapter_filter(architecture: str) -> None:
    """The adapter's applicable_phases must not filter Phase 1 back out.

    An adapter declaring applicable_phases=[] leaves _phases_to_run empty, which
    verify_models treats as "coverage deferred" and skips the model outright.
    """
    assert _phases_to_run(architecture, [1]) == [1]


@pytest.mark.parametrize("architecture", VISION_ARCHS)
def test_vision_skips_hooked_transformer_comparison(architecture: str) -> None:
    """There is no HookedTransformer vision encoder to compare against."""
    assert architecture in NO_HT_COMPARISON_ARCHITECTURES


def test_audio_still_routes_to_phase_eight() -> None:
    """Vision routing must not disturb the existing audio phase set."""
    assert _full_and_core_phases("ASTForAudioClassification") == ({1, 8}, {1, 8})
    assert _full_and_core_phases("HubertForCTC") == ({1, 8}, {1, 8})


def test_text_still_routes_to_all_four_phases() -> None:
    """Nor the default text phase set."""
    full, core = _full_and_core_phases("GPT2LMHeadModel")
    assert full == {1, 2, 3, 4}
    assert core == {1, 4}
