"""Shared fixtures for model_bridge acceptance tests.

Session-scoped fixtures avoid redundant model loads across test files.
All models used here must be in the CI cache (see .github/workflows/checks.yml).
"""

import pytest

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="session")
def gpt2_bridge():
    """TransformerBridge wrapping gpt2 (no compatibility mode)."""
    return TransformerBridge.boot_transformers("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_bridge_compat():
    """TransformerBridge wrapping gpt2 with compatibility mode enabled."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode()
    return bridge


@pytest.fixture(scope="session")
def gpt2_bridge_compat_no_processing():
    """TransformerBridge wrapping gpt2 with compatibility mode but no weight processing."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    return bridge


@pytest.fixture(scope="session")
def gpt2_hooked_processed():
    """HookedTransformer gpt2 with default weight processing."""
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_hooked_unprocessed():
    """HookedTransformer gpt2 without weight processing."""
    return HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
