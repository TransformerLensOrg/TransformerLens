"""Global pytest configuration for memory management and test optimization."""

import gc
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True, scope="function")
def cleanup_memory():
    """Automatically clean up memory after each test."""
    yield
    # Clear torch cache for all accelerators
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # Force garbage collection for cleanup
    gc.collect()


@pytest.fixture(autouse=True, scope="class")
def cleanup_class_memory():
    """Clean up memory after each test class."""
    yield
    # More aggressive cleanup after test classes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


# Configure pytest to be more memory-efficient
def pytest_configure(config):
    """Configure pytest for better memory usage and reproducible randomness."""
    # Configure garbage collection to be more aggressive
    gc.set_threshold(700, 10, 10)

    # Set random seeds for consistent test parametrization across parallel workers
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True, scope="session")
def _enable_hf_retry_for_tests():
    """Wrap HuggingFace Auto*.from_pretrained with retry-on-429 for the entire
    test session.

    Deferred to fixture (rather than pytest_configure) so jaxtyping's import
    hook can instrument transformer_lens before we import the helper.
    """
    from transformer_lens.utilities.hf_utils import enable_hf_retry

    enable_hf_retry()
    yield


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    """Session-scoped GPT-2 tokenizer (no add_bos_token).

    Shared across the unit, integration, and acceptance test trees to avoid
    repeated AutoTokenizer.from_pretrained("gpt2") calls — each one triggers
    a HuggingFace Hub freshness check even when the tokenizer is cached.
    Tokenizers are immutable for read-only use, so session scope is safe.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def gpt2_hooked_processed():
    """Session-scoped HookedTransformer gpt2 with default weight processing.

    Top-level fixture for unit tests and any other consumer without a closer
    fixture in scope. Sub-conftests in tests/acceptance/model_bridge/ and
    tests/integration/model_bridge/ define their own same-named fixture,
    which shadows this one within those subtrees.

    Safe for read-only use: ``.parameters()``, ``.state_dict()``,
    ``.to_tokens()``, ``.cfg``. Do NOT mutate (no ``.process_weights_()``,
    no permanent hooks, no ``.train()``/``.eval()`` that you don't restore).
    """
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


def pytest_sessionfinish(session, exitstatus):
    """Clean up at the end of test session."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
