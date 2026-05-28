"""Shared fixtures for model_bridge acceptance tests.

Session-scoped fixtures avoid redundant model loads across test files.
All models used here must be in the CI cache (see .github/workflows/checks.yml).

Imports of ``transformer_lens`` are deferred into the fixtures: this conftest is
loaded during pytest's initial conftest collection (before plugins' pytest_configure),
and a module-level TL import would trip jaxtyping's "package already imported" guard
when a single test file under this directory is run in isolation.
"""

import pytest


@pytest.fixture(scope="session")
def gpt2_bridge():
    """TransformerBridge wrapping gpt2 (no compatibility mode)."""
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_bridge_compat():
    """TransformerBridge wrapping gpt2 with compatibility mode enabled."""
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode()
    return bridge


@pytest.fixture(scope="session")
def gpt2_bridge_compat_no_processing():
    """TransformerBridge wrapping gpt2 with compatibility mode but no weight processing."""
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    return bridge


@pytest.fixture(scope="session")
def gpt2_hooked_processed():
    """HookedTransformer gpt2 with default weight processing."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_hooked_unprocessed():
    """HookedTransformer gpt2 without weight processing."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
