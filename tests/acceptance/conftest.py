"""Shared fixtures for acceptance tests.

Session-scoped fixtures avoid redundant model loads across test files.
All models used here must be in the CI cache (see .github/workflows/checks.yml).

NB: imports of ``transformer_lens`` are deferred into fixture bodies so that
jaxtyping's pytest_configure import hook can install before the package is
first imported.
"""

import pytest


@pytest.fixture(scope="session")
def gpt2_model():
    """Session-scoped HookedTransformer gpt2 with default weight processing."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="session")
def bloom_560m_hooked():
    """Session-scoped HookedTransformer for bigscience/bloom-560m.

    Loaded with ``default_prepend_bos=False`` to match what the bloom-similarity
    tests expect. Bloom-560m is ~1.2 GB so sharing is meaningful.
    """
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        "bigscience/bloom-560m", default_prepend_bos=False, device="cpu"
    )


@pytest.fixture(scope="session")
def bloom_560m_hf_model():
    """Session-scoped raw HuggingFace bloom-560m model (for parity checks)."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")


@pytest.fixture(scope="session")
def bloom_560m_hf_tokenizer():
    """Session-scoped bloom-560m tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bigscience/bloom-560m")
