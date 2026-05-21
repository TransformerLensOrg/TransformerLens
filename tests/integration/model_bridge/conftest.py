"""Shared fixtures for model_bridge integration tests.

Session-scoped fixtures avoid redundant model loads across test files.
All models used here must be in the CI cache (see .github/workflows/checks.yml).

NB: imports of ``transformer_lens`` are deferred into fixture bodies so that
jaxtyping's pytest_configure import hook can install before the package is
first imported. Module-level imports here break running these tests in
isolation (RuntimeError: "jaxtyping cannot check these packages because they
are already imported").
"""

import pytest


@pytest.fixture(scope="session")
def distilgpt2_bridge():
    """TransformerBridge wrapping distilgpt2 (no compatibility mode)."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    return TransformerBridge.boot_transformers("distilgpt2", device="cpu")


@pytest.fixture(scope="session")
def distilgpt2_bridge_compat():
    """TransformerBridge wrapping distilgpt2 with compatibility mode enabled."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode()
    return bridge


@pytest.fixture(scope="session")
def gpt2_bridge():
    """TransformerBridge wrapping gpt2 (no compatibility mode)."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    return TransformerBridge.boot_transformers("gpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_bridge_compat():
    """TransformerBridge wrapping gpt2 with compatibility mode enabled."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode()
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


@pytest.fixture(scope="session")
def distilgpt2_hooked_processed():
    """HookedTransformer distilgpt2 with default weight processing."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("distilgpt2", device="cpu")


@pytest.fixture(scope="session")
def distilgpt2_hooked_unprocessed():
    """HookedTransformer distilgpt2 without weight processing."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained_no_processing("distilgpt2", device="cpu")


@pytest.fixture(scope="session")
def gpt2_bridge_compat_no_processing():
    """TransformerBridge wrapping gpt2 with compat mode, no weight processing."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    return bridge


@pytest.fixture(scope="session")
def distilgpt2_bridge_compat_no_processing():
    """TransformerBridge wrapping distilgpt2 with compat mode, no weight processing."""
    from transformer_lens.model_bridge.bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True, disable_warnings=True)
    return bridge
