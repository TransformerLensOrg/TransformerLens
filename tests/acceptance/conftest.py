"""Shared fixtures for acceptance tests.

Session-scoped fixtures avoid redundant model loads across test files.
All models used here must be in the CI cache (see .github/workflows/checks.yml).
"""

import pytest

from transformer_lens import HookedTransformer


@pytest.fixture(scope="session")
def gpt2_model():
    """Session-scoped HookedTransformer gpt2 with default weight processing."""
    return HookedTransformer.from_pretrained("gpt2", device="cpu")
